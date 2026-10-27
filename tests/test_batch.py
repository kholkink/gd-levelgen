"""Batched decoding (gdgen/model/gpt.py BatchDecoder) gives what GPT.generate gives one request at a time."""
import torch

from gdgen.model.gpt import GPT, BatchDecoder, GPTConfig, RowCache, SampleRequest, run_sync


def model():
    torch.manual_seed(0)
    return GPT(GPTConfig.preset("test", vocab_size=128, ctx_len=96)).eval()


def test_step_logits_match_full_forward():
    m = model()
    seqs = [[1, 5, 9, 3], [2, 7, 7, 7, 7, 8, 1]]
    cache = RowCache(m, 2)
    last = [m.prefill_row(s[:-1], cache, r) for r, s in enumerate(seqs)]
    full_prev = [m(torch.tensor([s[:-1]]))[0, -1] for s in seqs]
    for a, b in zip(last, full_prev):
        assert torch.allclose(a, b, atol=1e-4)
    logits = m.step(torch.tensor([s[-1] for s in seqs]), torch.tensor([len(s) - 1 for s in seqs]), cache)
    for r, s in enumerate(seqs):
        assert torch.allclose(logits[r], m(torch.tensor([s]))[0, -1], atol=1e-4)


def requests():
    def inject(toks):                      # after every token 7 force 100, 101 (like music after a SEG)
        return [100, 101] if toks[-1] == 7 else []

    def until(toks):
        return len(toks) >= 30

    return [SampleRequest([1, 2, 3], 20, temperature=0, stop_ids={5}),
            SampleRequest([4] * 11, 25, temperature=0, inject=inject),
            SampleRequest([9, 8], 40, temperature=0, until=until),
            SampleRequest([3, 3, 3, 3, 3], 90, temperature=0),          # runs into the context limit
            SampleRequest([6], 15, temperature=0, inject=inject, stop_ids={11})]


def task(reqs):
    out = []
    for req in reqs:
        out.append((yield req))
    return out


def test_batch_decoder_matches_generate():
    m = model()
    single = [run_sync(task([r]), m)[0] for r in requests()]
    batched = BatchDecoder(m, rows=3).run([task([r]) for r in requests()])        # 5 requests, 3 rows
    assert [b[0] for b in batched] == single
    chained = BatchDecoder(m, rows=2).run([task(requests()[:3]), task(requests()[3:])])
    assert chained[0] + chained[1] == single                                     # a task's requests in order
