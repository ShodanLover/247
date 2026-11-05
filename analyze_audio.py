"""analyze_audio.py - objective checks on rendered tracks (used by verify_album.py).

    python analyze_audio.py file.wav [key_pc mode] ...
Prints loudness, peak, and how much of the pitched energy falls inside the song's key.
"""
import sys
import wave

import numpy as np

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]   # natural minor; dorian 6th shows up as a small leak


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = np.frombuffer(w.readframes(n), dtype=np.uint8)
    if width == 3:
        raw = raw.reshape(-1, 3)
        ints = raw[:, 0].astype(np.int32) | raw[:, 1].astype(np.int32) << 8 | raw[:, 2].astype(np.int32) << 16
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        x = ints / 8388608.0
    else:
        x = raw.view("<i2") / 32768.0
    return x.reshape(-1, ch).T, sr


def chroma(x, sr, lo=80, hi=2000):
    """Energy per pitch class from tonal peaks of a long-window spectrum (80 Hz - 2 kHz).

    Only bins standing well above the local noise floor count, so drum noise,
    vinyl hiss and sampler crunch don't register as notes."""
    from scipy.ndimage import median_filter
    mono = x.mean(axis=0)
    seg = 1 << 15
    hop = seg // 2
    win = np.hanning(seg)
    spec = np.zeros(seg // 2 + 1)
    for s in range(0, len(mono) - seg, hop):
        spec += np.abs(np.fft.rfft(mono[s:s + seg] * win)) ** 2
    f = np.fft.rfftfreq(seg, 1 / sr)
    floor = median_filter(spec, size=61)
    tonal = np.where(spec > 4 * floor, spec - floor, 0)   # >6 dB above the local floor
    keep = (f > lo) & (f < hi)
    midi = 69 + 12 * np.log2(f[keep] / 440)
    near = np.abs(midi - np.round(midi)) < 0.25        # in-tune bins only
    pcs = np.round(midi[near]).astype(int) % 12
    c = np.bincount(pcs, weights=tonal[keep][near], minlength=12)
    tuned = tonal[keep][near].sum() / (tonal[keep].sum() + 1e-12)
    return c / c.sum(), tuned


def in_key(c, key, mode):
    scale = MAJOR if mode == "major" else MINOR
    return float(sum(c[(key + p) % 12] for p in scale))


if __name__ == "__main__":
    import pyloudnorm
    args = sys.argv[1:]
    for path, key, mode in zip(args[0::3], args[1::3], args[2::3]):
        x, sr = read_wav(path)
        lufs = pyloudnorm.Meter(sr).integrated_loudness(x.T)
        c, tuned = chroma(x, sr)
        print(f"{path}: {lufs:.1f} LUFS, peak {20 * np.log10(np.abs(x).max()):.2f} dBFS, "
              f"in-key energy {in_key(c, int(key), mode):.0%}, tuned bins {tuned:.0%}")
