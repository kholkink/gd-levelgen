"""Trains the level model on pretokenized windows (gdgen/model/data.py).

Single GPU:  python scripts/train.py --data data/tokens/v1 --preset m --out runs/m1
Multi GPU:   torchrun --nproc_per_node 8 scripts/train.py ...
CPU smoke:   python scripts/train.py --preset test --ctx 512 --batch 2 --steps 20 --out runs/smoke

Logs one JSON line per log interval to <out>/log.jsonl; checkpoints to <out>/ckpt.pt (resumed automatically).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.model.data import TokenData, WindowDataset, WindowSampler, token_dirs  # noqa: E402
from gdgen.model.gpt import GPT, GPTConfig  # noqa: E402


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


def val_batches(args, n: int) -> dict[str, list]:
    """Fixed validation windows per kind, so the numbers are comparable across steps and runs."""
    sampler = WindowSampler(TokenData(args.data, "val"), args.ctx, p_layout=0.5, seed=12345)
    out = {"layout": [], "deco": []}
    for kind in out:
        tries = 0
        while len(out[kind]) < n and tries < 20 * n:
            x, y, k = sampler.sample(kind)
            tries += 1
            if k == kind:
                out[kind].append((torch.from_numpy(x), torch.from_numpy(y)))
    return out


@torch.no_grad()
def evaluate(model, batches, device, amp) -> dict[str, float]:
    model.eval()
    res = {}
    for kind, items in batches.items():
        tot, cnt = 0.0, 0
        for i in range(0, len(items), 4):
            x = torch.stack([a for a, _ in items[i:i + 4]]).to(device)
            y = torch.stack([b for _, b in items[i:i + 4]]).to(device)
            with amp:
                loss, c = model(x, y)
            tot, cnt = tot + float(loss), cnt + int(c)
        res[f"val_{kind}"] = tot / max(1, cnt)
    model.train()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "tokens" / "v1"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="m")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=4, help="micro-batch per GPU")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--p-layout", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-windows", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--save-minutes", type=float, default=15, help="also save when this long since the last save")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="auto: bf16 on Ampere+ (A100/H100/RTX 6000), fp16 with loss scaling on T4/P100")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = args.dtype
    if dtype == "auto":
        dtype = ("bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16") if device.startswith("cuda") \
            else "fp32"
    amp = torch.autocast("cuda", dtype=torch.bfloat16 if dtype == "bf16" else torch.float16) \
        if dtype in ("bf16", "fp16") and device.startswith("cuda") else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == "fp16")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab_size = len(json.loads((token_dirs(args.data)[0] / "vocab.json").read_text())["tokens"])
    cfg = GPTConfig.preset(args.preset, vocab_size=128 * math.ceil(vocab_size / 128), ctx_len=args.ctx)
    model = GPT(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=device.startswith("cuda"))
    step = 0
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        if master:
            print(f"resumed from step {step}")
    if master:
        print(f"model: {model.n_params() / 1e6:.1f}M non-embedding params, {model.n_params(False) / 1e6:.1f}M total; "
              f"vocab {vocab_size}; ctx {args.ctx}; tokens/step {args.batch * args.accum * world * args.ctx:,}; "
              f"{dtype} on {torch.cuda.get_device_name(device) if device.startswith('cuda') else 'cpu'} x{world}")
        (out / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "args": vars(args)}, indent=1))

    def wrap(m):
        return torch.nn.parallel.DistributedDataParallel(m, device_ids=[int(os.environ["LOCAL_RANK"])]) if ddp else m

    fwd = wrap(torch.compile(model) if args.compile else model)
    compiled = args.compile

    ds = WindowDataset(args.data, args.ctx, "train", seed=args.seed + step, rank=rank, p_layout=args.p_layout)
    loader = iter(torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                              pin_memory=device.startswith("cuda"), persistent_workers=args.workers > 0))
    vb = val_batches(args, args.eval_windows) if master else None

    t0, tok_seen, log_loss, log_cnt = time.time(), 0, 0.0, 0
    last_save = time.time()
    model.train()
    while step < args.steps:
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        for micro in range(args.accum):
            x, y = next(loader)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            sync = (not ddp) or micro == args.accum - 1
            try:
                with (nullcontext() if sync else fwd.no_sync()), amp:
                    loss_sum, cnt = fwd(x, y, grad_ckpt=args.grad_ckpt)
            except Exception as e:  # noqa: BLE001  (torch.compile failing on this machine must not end the run)
                if not compiled:
                    raise
                print(f"torch.compile failed ({type(e).__name__}: {str(e)[:300]}); training without it", flush=True)
                fwd, compiled = wrap(model), False
                with (nullcontext() if sync else fwd.no_sync()), amp:
                    loss_sum, cnt = fwd(x, y, grad_ckpt=args.grad_ckpt)
            with amp:
                loss = loss_sum / cnt.clamp(min=1) / args.accum
            scaler.scale(loss).backward()
            log_loss += float(loss_sum.detach())
            log_cnt += int(cnt)
            tok_seen += x.numel()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if master and step % args.log_every == 0:
            dt = time.time() - t0
            rec = {"step": step, "loss": log_loss / max(1, log_cnt), "lr": lr, "gnorm": float(gnorm),
                   "tok_per_s": tok_seen * world / dt, "time": time.time()}
            print(json.dumps(rec), flush=True)
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            t0, tok_seen, log_loss, log_cnt = time.time(), 0, 0.0, 0
        if master and (step % args.eval_every == 0 or step == args.steps):
            rec = {"step": step, **evaluate(model, vb, device, amp), "time": time.time()}
            print(json.dumps(rec), flush=True)
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        due = step % args.save_every == 0 or step == args.steps or time.time() - last_save >= 60 * args.save_minutes
        if ddp:                                  # every rank must agree, the time check is per process
            flag = torch.tensor([int(due)], device=device)
            dist.all_reduce(flag)
            due = bool(flag.item())
        if due:
            last_save = time.time()
        if master and due:
            tmp = out / "ckpt.tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "config": cfg.to_dict(), "args": vars(args)}, tmp)
            tmp.replace(ckpt_path)
    if ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
