#!/usr/bin/env python3
"""
fire_ambience.py - synthesizes a crackling fireplace with wind outside the window.

Layers (each its own function so levels are easy to tune):
  roar     low, breathing rumble of the flames
  hiss     airy gas hiss that swells with the fire
  crackle  clustered small snaps from burning wood
  pops     occasional loud pops with a woody knock
  settle   a log shifting now and then (thump + burst of crackles)
  wind     muffled gusts and a faint whistle outside the glass

Usage:
    python fire_ambience.py --seconds 45 --out fire_sample
    python fire_ambience.py --seconds 3600 --wind 0 --out fire_1h_nowind
"""
import argparse
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44100


def lowpass(x, f, order=2):
    return sosfilt(butter(order, f, "lowpass", fs=SR, output="sos"), x, axis=-1)


def bandpass(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x, axis=-1)


def slow_noise(n, rng, hz, lo=0.0, hi=1.0):
    """A smooth random curve that wanders between lo and hi about `hz` times a second."""
    pts = max(4, int(n / SR * hz) + 4)
    knots = rng.random(pts)
    curve = np.interp(np.linspace(0, pts - 3, n), np.arange(pts), knots)
    curve = lowpass(curve, max(hz, 0.05), 1)
    curve = (curve - curve.min()) / (np.ptp(curve) + 1e-9)
    return lo + (hi - lo) * curve


def pan(mono, p):
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)]) * np.sqrt(2)


def roar(n, rng, intensity):
    x = lowpass(rng.standard_normal((2, n)), 220, 4)
    x += 0.4 * bandpass(rng.standard_normal((2, n)), 220, 600)
    return x * (0.5 + 0.5 * intensity)


def hiss(n, rng, intensity):
    x = bandpass(rng.standard_normal((2, n)), 1800, 7000)
    flicker = slow_noise(n, rng, 3, 0.6, 1.0)
    return x * intensity ** 2 * flicker


def crackle(n, rng, intensity):
    """Snaps arrive in clusters: a fast-varying rate driven by the fire's intensity."""
    out = np.zeros((2, n))
    rate = 4 + 26 * intensity ** 2 * slow_noise(n, rng, 1.5)       # snaps per second
    times = np.flatnonzero(rng.random(n) < rate / SR)
    for t in times:
        size = rng.lognormal(-1.3, 0.8)
        length = int(SR * rng.uniform(0.0006, 0.004))
        if t + length >= n:
            continue
        burst = rng.standard_normal(length) * np.exp(-np.arange(length) / (length / 4)) * size
        out[:, t:t + length] += pan(burst, rng.uniform(-0.5, 0.5))
    # Shape the snaps: woody mids + a little sparkle, and tame the harshest highs.
    return lowpass(bandpass(out, 700, 9000) + 0.3 * bandpass(out, 300, 900), 8000)


def knock(freq, rng, length=0.05):
    t = np.arange(int(length * SR)) / SR
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * 90) + 0.6 * rng.standard_normal(len(t)) * np.exp(-t * 400)


def pops(n, rng):
    out = np.zeros((2, n))
    k = rng.poisson(n / SR * 0.35)                                 # a loud pop every ~3 s
    for t in rng.integers(0, n - SR // 10, k):
        sig = knock(rng.uniform(350, 1100), rng) * rng.lognormal(-0.6, 0.5)
        out[:, t:t + len(sig)] += pan(sig, rng.uniform(-0.4, 0.4))
    return lowpass(out, 7000)


def settle(n, rng, out_crackle):
    """A log shifting: soft thump plus a burst of extra crackle, every 20-45 s."""
    out = np.zeros((2, n))
    t = int(rng.uniform(6, 20) * SR)
    while t < n - SR:
        th = np.arange(int(0.4 * SR)) / SR
        thump = np.sin(2 * np.pi * 70 * th * (1 - 0.3 * th)) * np.exp(-th * 9)
        thump += 0.5 * lowpass(rng.standard_normal(len(th)), 900) * np.exp(-th * 14)
        out[:, t:t + len(th)] += pan(thump, rng.uniform(-0.2, 0.2)) * 0.5
        burst_len = int(rng.uniform(1.5, 3.0) * SR)
        e = min(n, t + burst_len)
        env = np.exp(-np.arange(e - t) / (0.6 * SR))
        out_crackle[:, t:e] *= 1 + 2.5 * env
        t += int(rng.uniform(20, 45) * SR)
    return out


def wind(n, rng):
    """Gusts outside, muffled by the window, with a faint moving whistle."""
    gust = slow_noise(n, rng, 0.12, 0.15, 1.0) ** 1.8
    base = lowpass(rng.standard_normal((2, n)), 500, 4)
    # moving band for the whistle: filter blocks with a slowly changing center
    whistle = np.zeros((2, n))
    noise = rng.standard_normal((2, n))
    center = slow_noise(n, rng, 0.08, 380, 750)
    blk = SR // 4
    for s in range(0, n, blk):
        e = min(n, s + blk + 512)
        c = float(center[s])
        whistle[:, s:e] += bandpass(noise[:, s:e], c * 0.94, c * 1.06) * np.hanning(e - s)
    return lowpass(base + 0.5 * whistle, 1200) * gust


def render(seconds, seed, wind_db):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    intensity = slow_noise(n, rng, 0.07, 0.45, 1.0)                # the fire breathes slowly
    c = crackle(n, rng, intensity)
    s = settle(n, rng, c)
    layers = {
        "roar": (roar(n, rng, intensity), -14),
        "hiss": (hiss(n, rng, intensity), -30),
        "crackle": (c, -6),
        "pops": (pops(n, rng), -11),
        "settle": (s, -12),
    }
    mix = np.zeros((2, n))
    for sig, db in layers.values():
        mix += sig / (np.sqrt(np.mean(sig ** 2)) + 1e-12) * 10 ** (db / 20)
    if wind_db > -60:
        w = wind(n, rng)
        mix += w / (np.sqrt(np.mean(w ** 2)) + 1e-12) * 10 ** (wind_db / 20)
    fade = np.minimum(1, np.arange(n) / (2 * SR)) * np.minimum(1, (n - np.arange(n)) / (2 * SR))
    return mix * fade


def master(x, target=-20.0):
    try:
        import pyloudnorm
        lufs = pyloudnorm.Meter(SR).integrated_loudness(x.T)
    except ImportError:
        lufs = 20 * np.log10(np.sqrt(np.mean(x ** 2))) - 0.7
    x = x * 10 ** ((target - lufs) / 20)
    peak = np.abs(x).max()
    if peak > 0.89:   # pops are spiky: soft-limit only the tips
        knee = 0.6
        a = np.abs(x)
        over = a > knee
        x[over] = np.sign(x[over]) * (knee + 0.29 * np.tanh((a[over] - knee) / 0.29))
    return x


def write_wav24(path, x):
    ints = np.clip(np.round(x * 8388607), -8388608, 8388607).astype("<i4")
    frames = ints.T.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wind", type=float, default=-17, help="wind level in dB vs the fire; -99 = off")
    ap.add_argument("--lufs", type=float, default=-20, help="ambience sits quieter than music")
    ap.add_argument("--out", default="fire_sample")
    args = ap.parse_args()

    x = master(render(args.seconds, args.seed, args.wind), args.lufs)
    wav = Path(args.out).with_suffix(".wav")
    write_wav24(wav, x)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "320k",
                        str(wav.with_suffix(".mp3"))], check=True)
    print(f"Wrote {wav} ({args.seconds:.0f}s)")


if __name__ == "__main__":
    main()
