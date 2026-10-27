"""Downloads songs and extracts the music features the level model is conditioned on (runs on Kaggle CPU kernels).

For every song of song_urls.jsonl in this part (id % n == i): download, decode to mono 22.05 kHz, and save
<out>/<id>.npz with frame-level curves at sr/hop = 22050/512 (~43 fps):
    onset   onset strength (spectral flux), divided by the song's 99th percentile
    rms     loudness, dB relative to the song's maximum
    low/mid/high   mel-band energies (< 200 Hz, 200 Hz - 3 kHz, > 3 kHz), dB relative to the song's maximum
    beats   beat times in seconds (librosa beat tracker); tempo in BPM; duration in seconds; fps
The audio is deleted right after; index.jsonl lists every song with ok/error.

Usage: python scripts/song_features.py --urls song_urls.jsonl --out song_features/part_0of4 --part 0/4 [--limit N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

SR, HOP = 22050, 512
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one thread per worker process


def load_audio(path: str) -> np.ndarray:
    """Mono float32 at SR: libsndfile (mp3 since 1.1) or ffmpeg, never audioread, whose ffmpeg can block on an
    inherited stdin inside a notebook's subprocess."""
    import librosa
    import soundfile as sf

    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001  (old libsndfile without mp3)
        raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
                              "-ar", str(SR), "-"], stdin=subprocess.DEVNULL, capture_output=True, timeout=180,
                             check=True).stdout
        return np.frombuffer(raw, dtype=np.float32)
    y = y.mean(axis=1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def features(path: str) -> dict:
    import librosa

    y = load_audio(path)
    if len(y) < SR:
        raise ValueError("audio shorter than a second")
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    mel = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=64, fmax=SR / 2)
    freqs = librosa.mel_frequencies(n_mels=64, fmax=SR / 2)
    bands = [mel[freqs < 200].sum(0), mel[(freqs >= 200) & (freqs < 3000)].sum(0), mel[freqs >= 3000].sum(0)]

    def db(power):
        power = np.asarray(power, dtype=np.float64)
        return (10 * np.log10(np.maximum(power, 1e-10) / max(power.max(), 1e-10))).astype(np.float16)

    n = min(len(onset), len(rms), mel.shape[1])
    scale = max(float(np.percentile(onset, 99)), 1e-6)
    return {
        "onset": np.clip(onset[:n] / scale, 0, 4).astype(np.float16),
        "rms": db(rms[:n] ** 2), "low": db(bands[0][:n]), "mid": db(bands[1][:n]), "high": db(bands[2][:n]),
        "beats": librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP).astype(np.float32),
        "tempo": np.float32(np.atleast_1d(tempo)[0]), "duration": np.float32(len(y) / SR),
        "fps": np.float32(SR / HOP),
    }


def download(url: str, path: str) -> None:
    import requests

    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            return
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def process(job: tuple[dict, str]) -> dict:
    rec, out = job
    sid, url = rec["id"], rec["url"]
    target = Path(out) / f"{sid}.npz"
    if target.exists():
        return {"id": sid, "ok": True, "cached": True}
    fd, tmp = tempfile.mkstemp(suffix=".ogg" if url.split("?")[0].endswith(".ogg") else ".mp3")
    os.close(fd)
    try:
        download(url, tmp)
        feats = features(tmp)
        np.savez_compressed(target, **feats)
        return {"id": sid, "ok": True, "duration": float(feats["duration"]), "tempo": float(feats["tempo"]),
                "frames": int(len(feats["onset"]))}
    except Exception as e:  # noqa: BLE001  (one bad file must not stop the part)
        return {"id": sid, "ok": False, "error": f"{type(e).__name__}: {e}"[:300],
                "trace": traceback.format_exc(limit=2)[-300:]}
    finally:
        os.remove(tmp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    i, n = map(int, args.part.split("/"))
    recs = [json.loads(line) for line in Path(args.urls).read_text().splitlines() if line]
    recs = [r for r in recs if r.get("url") and r["id"] % n == i]
    if args.limit:
        recs = recs[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"part {args.part}: {len(recs)} songs", flush=True)
    t0, ok = time.time(), 0
    # spawn, not fork: forking after numba/BLAS threads exist can deadlock the workers
    with open(out / "index.jsonl", "a") as index, multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(process, [(r, str(out)) for r in recs]), 1):
            index.write(json.dumps(res) + "\n")
            index.flush()
            ok += res["ok"]
            if not res["ok"] and k - ok <= 5:
                print("error:", res["id"], res["error"], res.get("trace", ""), flush=True)
            if k % 50 == 0 or k == len(recs):
                print(f"{k}/{len(recs)} songs, {ok} ok, {(time.time() - t0) / k:.1f} s/song", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
