#!/usr/bin/env python3
"""
lofi_beat.py - synthesizes original lo-fi hip-hop tracks from scratch.

Every sound is generated from oscillators and noise with numpy/scipy - no
samples, plugins or downloaded audio. Each seed is a different song: key,
chord progression, tempo, swing, instruments, comping style, drum style,
song structure, melodies and textures (vinyl, rain, tape, sampler crunch) vary.

Usage:
    python lofi_beat.py --seed 12             # one track -> lofi_beat.wav/.mp3
    python lofi_beat.py --seed 12 --out mytrack
    python make_album.py --count 30           # a whole playlist (see README)
"""
import argparse
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
LEAD_IN = 1.0         # seconds of texture before the music starts
TAIL = 8.0            # seconds after the last bar for the final chord to ring
TARGET_LUFS = -15.0   # every track lands at the same loudness for back-to-back play


# ---------------------------------------------------------------- DSP helpers
def lowpass(x, f, order=2):
    return sosfilt(butter(order, f, "lowpass", fs=SR, output="sos"), x, axis=-1)


def highpass(x, f, order=2):
    return sosfilt(butter(order, f, "highpass", fs=SR, output="sos"), x, axis=-1)


def bandpass(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x, axis=-1)


def midi_hz(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def active_rms(x):
    """RMS of the louder half of 0.5 s blocks, so silent sections don't skew levels."""
    mono = np.abs(x).mean(axis=0) if x.ndim == 2 else np.abs(x)
    blk = SR // 2
    n = len(mono) // blk
    if n == 0:
        return rms(x)
    b = np.sqrt(np.mean(mono[: n * blk].reshape(n, blk) ** 2, axis=1))
    b = np.sort(b)[n // 2:]
    return float(np.mean(b) + 1e-12)


def env(n, dur, attack=0.004, decay=1.0, rel=0.3):
    t = np.arange(n) / SR
    return (np.minimum(1, t / attack) * np.exp(-t * decay)
            * np.clip(1 - (t - dur) / rel, 0, 1)), t


def karplus(freq, n, rng, damp=0.996, bright=0.5):
    """Plucked string (Karplus-Strong), vectorised one period at a time."""
    N = max(2, int(round(SR / freq - 0.5)))
    exc = rng.uniform(-1, 1, N)
    exc = lowpass(exc, 1500 + 6000 * bright) if N > 12 else exc
    y = np.zeros(n + N + 1)
    y[1:N + 1] = exc
    for s in range(N + 1, n + N + 1, N):
        e = min(s + N, n + N + 1)
        y[s:e] = damp * 0.5 * (y[s - N:e - N] + y[s - N - 1:e - N - 1])
    return y[1:n + 1]


class Bus:
    """A stereo track that one-shot sounds are mixed into."""

    def __init__(self, n):
        self.buf = np.zeros((2, n))

    def add(self, sig, t, gain=1.0, pan=0.0):
        start = int(round(t * SR))
        if start < 0:
            sig, start = sig[-start:], 0
        n = min(len(sig), self.buf.shape[1] - start)
        if n <= 0:
            return
        a = (pan + 1) * np.pi / 4  # equal-power pan
        self.buf[0, start:start + n] += sig[:n] * gain * np.cos(a) * np.sqrt(2)
        self.buf[1, start:start + n] += sig[:n] * gain * np.sin(a) * np.sqrt(2)


# ---------------------------------------------------------------- chord instruments
def epiano(freq, dur, vel, rng):
    """FM electric piano: decaying modulator gives the Rhodes-style bark."""
    e, t = env(int((dur + 0.6) * SR), dur, 0.003, 0.9 + freq / 600, 0.6)
    index = (0.6 + 2.2 * vel) * np.exp(-t * 7) + 0.25
    ph = rng.uniform(0, 2 * np.pi)
    tone = np.sin(2 * np.pi * freq * t + index * np.sin(2 * np.pi * freq * t + ph))
    tine = 0.06 * vel * np.sin(2 * np.pi * freq * 7 * t) * np.exp(-t * 25)
    trem = 1 + 0.08 * np.sin(2 * np.pi * 4.2 * t + ph)
    return (tone + tine) * e * trem * (0.35 + 0.65 * vel)


def felt(freq, dur, vel, rng):
    """Felt piano: slightly inharmonic partials, highs decaying first, soft hammer."""
    e, t = env(int((dur + 0.5) * SR), dur, 0.002, 0.0, 0.5)
    x = np.zeros_like(t)
    for k in range(1, 9):
        fk = k * freq * np.sqrt(1 + 0.0004 * k * k)
        if fk > 9000:
            break
        amp = (0.5 + 0.5 * vel) ** (k - 1) / k ** 1.4
        x += amp * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28)) * np.exp(-t * (0.6 + 0.5 * k + freq / 700))
    hammer = lowpass(rng.standard_normal(len(t)) * np.exp(-t * 200), 1200) * 0.08
    return lowpass(x * e + hammer, 3500) * (0.4 + 0.6 * vel)


def nylon(freq, dur, vel, rng):
    """Nylon-string guitar via Karplus-Strong."""
    n = int((dur + 0.3) * SR)
    x = karplus(freq, n, rng, damp=0.9985 - freq / 400000, bright=0.3 + 0.4 * vel)
    e, _ = env(n, dur, 0.001, 0.4, 0.3)
    return lowpass(x * e, 3200) * vel * 1.4


def vibes(freq, dur, vel, rng):
    """Vibraphone: harmonic FM bell with motor tremolo."""
    e, t = env(int((dur + 1.0) * SR), dur, 0.002, 1.1, 1.0)
    index = 1.2 * vel * np.exp(-t * 9)
    tone = np.sin(2 * np.pi * freq * t + index * np.sin(2 * np.pi * freq * 4 * t))
    tone += 0.15 * np.sin(2 * np.pi * freq * 4 * t) * np.exp(-t * 6)
    trem = 1 + 0.25 * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6.28))
    return tone * e * trem * (0.3 + 0.7 * vel)


def pad(freq, dur, vel, rng):
    """Warm pad: three detuned, filtered saws with a slow swell."""
    e, t = env(int((dur + 1.2) * SR), dur, 0.9, 0.0, 1.2)
    x = np.zeros_like(t)
    for d in (-6, 0, 7):
        ph = (freq * 2 ** (d / 1200) * t + rng.random()) % 1
        x += 2 * ph - 1
    return lowpass(x * e, 1100) * vel / 3


KEYS = {"epiano": epiano, "felt": felt, "nylon": nylon, "vibes": vibes}


# ---------------------------------------------------------------- bass
def bass_round(freq, dur, vel, rng):
    e, t = env(int((dur + 0.08) * SR), dur, 0.008, 1.2, 0.08)
    w = 2 * np.pi * freq * t
    x = np.tanh(1.5 * (np.sin(w) + 0.25 * np.sin(2 * w) + 0.08 * np.sin(3 * w)))
    return lowpass(x * e, 800) * vel


def bass_upright(freq, dur, vel, rng):
    n = int((dur + 0.1) * SR)
    x = karplus(freq, n, rng, damp=0.997, bright=0.15)
    e, t = env(n, dur, 0.002, 2.0, 0.1)
    thump = np.sin(2 * np.pi * freq * t) * np.exp(-t * 10) * 0.6
    return lowpass((x + thump) * e, 700) * vel * 1.5


def bass_sub(freq, dur, vel, rng):
    e, t = env(int((dur + 0.15) * SR), dur, 0.005, 0.5, 0.15)
    f = freq * (1 + 0.12 * np.exp(-t * 40))   # short downward glide
    x = np.tanh(1.3 * np.sin(2 * np.pi * np.cumsum(f) / SR))
    return lowpass(x * e, 500) * vel


BASS = {"round": bass_round, "upright": bass_upright, "sub": bass_sub}


# ---------------------------------------------------------------- leads
def flute(freq, dur, vel, rng):
    e, t = env(int((dur + 0.3) * SR), dur, 0.04, 0.8, 0.3)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5 * t) * np.minimum(1, t / 0.4)
    ph = 2 * np.pi * np.cumsum(freq * vib) / SR
    x = np.sin(ph) + 0.15 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)
    breath = bandpass(rng.standard_normal(len(t)), 1500, 5000) * 0.03
    return lowpass((x + breath) * e, 3000) * vel


def horn(freq, dur, vel, rng):
    """Muted horn: band-limited saw, darker, with vibrato."""
    e, t = env(int((dur + 0.2) * SR), dur, 0.06, 0.5, 0.2)
    vib = 1 + 0.005 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.3)
    ph = 2 * np.pi * np.cumsum(freq * vib) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 8) if k * freq < 6000)
    return lowpass(x * e, 1400) * vel * 0.8


def pluck(freq, dur, vel, rng):
    n = int((dur + 0.4) * SR)
    x = karplus(freq, n, rng, damp=0.998, bright=0.6)
    e, _ = env(n, dur, 0.001, 0.6, 0.4)
    return lowpass(x * e, 4500) * vel * 1.3


def whistle(freq, dur, vel, rng):
    e, t = env(int((dur + 0.15) * SR), dur, 0.05, 0.6, 0.15)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 6 * t) * np.minimum(1, t / 0.25)
    return np.sin(2 * np.pi * np.cumsum(freq * vib) / SR) * e * vel


LEADS = {"flute": (flute, 67, 86), "horn": (horn, 60, 79), "pluck": (pluck, 62, 81),
         "whistle": (whistle, 72, 88), "vibes": (vibes, 67, 86)}


# ---------------------------------------------------------------- drums
def kick(vel, rng, tune=44, punch=0.0):
    """Pitch-swept sine kick; `punch` (0-1) adds attack, click and drive for harder styles."""
    t = np.arange(int(0.5 * SR)) / SR
    f = tune + (85 + 40 * punch) * np.exp(-t * (32 + 18 * punch))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * (7 + 2 * punch))
    click = lowpass(rng.standard_normal(len(t)) * np.exp(-t * 300), 2500 + 2500 * punch) * (0.15 + 0.2 * punch)
    return lowpass(np.tanh((1.6 + 1.2 * punch) * (body + click)), 3500 + 3000 * punch) * vel


def snare(vel, rng, tone=185.0, ratio=1.784):
    t = np.arange(int(0.35 * SR)) / SR
    body = (0.5 * np.sin(2 * np.pi * tone * t) * np.exp(-t * 28)
            + 0.2 * np.sin(2 * np.pi * tone * ratio * t) * np.exp(-t * 40))
    noise = bandpass(rng.standard_normal(len(t)), 1200, 6500) * np.exp(-t * 18)
    return lowpass(body + 0.9 * noise, 5000) * vel


def rim(vel, rng, tone=1700.0):
    t = np.arange(int(0.08 * SR)) / SR
    x = bandpass(rng.standard_normal(len(t)), 1400, 2600) * np.exp(-t * 70)
    x += 0.4 * np.sin(2 * np.pi * tone * t) * np.exp(-t * 90)
    return x * vel * 1.3


def clap(vel, rng):
    t = np.arange(int(0.3 * SR)) / SR
    e = sum(np.where(t >= d, np.exp(-(t - d) * 120), 0) for d in (0, 0.009, 0.019))
    e += np.where(t >= 0.028, np.exp(-(t - 0.028) * 14), 0) * 0.6
    return lowpass(bandpass(rng.standard_normal(len(t)), 900, 5000) * e, 6000) * vel * 0.9


def hat(vel, rng, open_=False):
    dur, decay = (0.35, 9) if open_ else (0.06, 70)
    t = np.arange(int(dur * SR)) / SR
    x = highpass(rng.standard_normal(len(t)), 7000, order=4) * np.exp(-t * decay)
    return lowpass(x, 11000) * vel


def shaker(vel, rng):
    t = np.arange(int(0.09 * SR)) / SR
    e = np.minimum(1, t / 0.015) * np.exp(-t * 45)
    return bandpass(rng.standard_normal(len(t)), 4500, 10000) * e * vel


SNARES = {"snare": snare, "rim": rim, "clap": clap}


# ---------------------------------------------------------------- textures + fx
def vinyl(n, rng):
    """Hiss, fine crackle, occasional pops and a 33 1/3 rpm loudness wobble."""
    secs = n / SR
    t = np.arange(n) / SR
    hiss = lowpass(highpass(rng.standard_normal((2, n)), 800), 7000) * 0.1

    def impulses(rate, scale):
        imp = np.zeros((2, n))
        k = rng.poisson(secs * rate)
        pos = rng.integers(0, n, k)
        amp = rng.exponential(scale, k) * rng.choice([-1, 1], k)
        side = rng.random(k)
        imp[0, pos] += amp * (1 - 0.6 * side)
        imp[1, pos] += amp * (0.4 + 0.6 * side)
        return imp

    clicks = bandpass(impulses(9, 1.0), 1200, 9000)
    pops = lowpass(impulses(0.35, 4.0), 1800)
    return (hiss + clicks + pops) * (1 + 0.25 * np.sin(2 * np.pi * 0.555 * t))


def rain(n, rng):
    """Rain on a window: a soft noise bed that slowly swells, plus droplet ticks."""
    bed = bandpass(rng.standard_normal((2, n)), 300, 5000)
    swell = lowpass(rng.standard_normal(n), 0.3)
    swell = 1 + 0.4 * swell / (np.abs(swell).max() + 1e-9)
    imp = np.zeros((2, n))
    k = rng.poisson(n / SR * 35)
    imp[rng.integers(0, 2, k), rng.integers(0, n, k)] = rng.exponential(1.0, k)
    ticks = bandpass(imp, 2500, 9000) * 3
    return lowpass(bed * swell * 0.4 + ticks, 7000)


def reverb(x, rng, decay=1.8, size=2.4):
    """Convolution reverb with a synthetic impulse response (decaying noise)."""
    t = np.arange(int(size * SR)) / SR
    ir = lowpass(rng.standard_normal((2, len(t))) * np.exp(-t * 6.9 / decay), 4500)
    ir = np.concatenate([np.zeros((2, int(0.02 * SR))), ir], axis=1)
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
    return np.stack([fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])


def tape_wow(x, phase, depth=1.0):
    """Slow pitch drift (wow) and faint flutter via a modulated delay line."""
    n = x.shape[1]
    t = np.arange(n) / SR
    d = (0.004 + 0.0011 * depth * np.sin(2 * np.pi * 0.45 * t + phase)
         + 0.00006 * np.sin(2 * np.pi * 5.3 * t))
    idx = np.arange(n)
    return np.stack([np.interp(idx - d * SR, idx, ch) for ch in x])


def crunch(x, rate=26000, bits=12):
    """Old-sampler character: sample-and-hold at a lower rate, fewer bits."""
    step = SR / rate
    idx = (np.floor(np.arange(x.shape[1]) / step) * step).astype(int)
    q = 2 ** (bits - 1)
    peak = np.abs(x).max() + 1e-9
    return lowpass(np.round(x[:, idx] / peak * q) / q * peak, rate * 0.45)


# ---------------------------------------------------------------- harmony
QUAL = {
    "maj7": [0, 4, 7, 11], "maj9": [0, 4, 7, 11, 14], "6/9": [0, 4, 7, 9, 14],
    "maj7#11": [0, 4, 7, 11, 18], "m7": [0, 3, 7, 10], "m9": [0, 3, 7, 10, 14],
    "m11": [0, 3, 7, 10, 14, 17], "7": [0, 4, 7, 10], "9": [0, 4, 7, 10, 14],
    "13": [0, 4, 10, 14, 21], "7b9": [0, 4, 7, 10, 13], "m7b5": [0, 3, 6, 10],
    "sus9": [0, 5, 7, 10, 14], "susb9": [0, 5, 7, 10, 13],
}

# (name, mode, bars); each bar is a list of (semitones above the tonic, quality, beats)
PROGRESSIONS = [
    ("falling", "major", [[(5, "maj9", 4)], [(4, "m7", 4)], [(2, "m9", 4)], [(0, "maj9", 4)]]),
    ("two-five", "major", [[(2, "m9", 4)], [(7, "13", 4)], [(0, "maj9", 4)], [(9, "m9", 4)]]),
    ("sunday", "major", [[(0, "maj9", 4)], [(9, "m9", 4)], [(2, "m9", 4)], [(7, "13", 4)]]),
    ("royal road", "major", [[(5, "maj9", 4)], [(7, "9", 4)], [(4, "m7", 4)], [(9, "m9", 4)]]),
    ("drifting", "minor", [[(0, "m9", 4)], [(5, "m9", 4)], [(10, "13", 4)], [(3, "maj7", 4)]]),
    ("midnight", "minor", [[(0, "m9", 4)], [(8, "maj7", 4)], [(3, "maj9", 4)], [(10, "13", 4)]]),
    ("dorian vamp", "minor", [[(0, "m9", 4)], [(0, "m11", 4)], [(5, "9", 4)], [(5, "13", 4)]]),
    ("floating", "major", [[(0, "maj9", 4)], [(5, "maj7#11", 4)], [(0, "6/9", 4)], [(5, "maj7#11", 4)]]),
    ("circle", "major", [[(4, "m7", 4)], [(9, "m9", 4)], [(2, "m9", 4)], [(7, "13", 4)]]),
    ("rainy minor", "minor", [[(0, "m9", 4)], [(5, "m11", 4)],
                              [(2, "m7b5", 2), (7, "7b9", 2)], [(0, "m9", 4)]]),
]
PENTA = {"major": [0, 2, 4, 7, 9], "minor": [0, 3, 5, 7, 10]}
KEY_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def voice(pcs, prev, lo=50, hi=76):
    """Place pitch classes as a close voicing near the previous chord (smooth voice leading)."""
    best, best_cost = None, 1e9
    for rot in range(len(pcs)):
        order = pcs[rot:] + pcs[:rot]
        for start in range(lo, lo + 12):
            notes, cur = [], start - 1
            for pc in order:
                cur += (pc - cur) % 12 or 12
                notes.append(cur)
            if notes[-1] > hi:
                continue
            if prev:
                cost = sum(abs(a - b) for a, b in zip(sorted(notes), sorted(prev)))
                cost += 3 * abs(len(notes) - len(prev))
            else:
                cost = abs(np.mean(notes) - 62)
            if cost < best_cost:
                best, best_cost = notes, cost
    return best or voice(pcs, prev, lo, hi + 6)


def turnaround(spec):
    """The chord that ends each section and leads back to the top of the loop."""
    _, mode, prog = PROGRESSIONS[spec["progression"]]
    if spec.get("diatonic_turnaround"):
        # V sus (major) or V sus b9 (minor): every note is in the key
        return 7, "sus9" if mode == "major" else "susb9"
    return (prog[0][0][0] + 7) % 12, "sus9"   # dominant of the first chord (can be chromatic)


def pitch_classes_used(spec):
    """Every pitch class the song's scale and chords use (for checking it stays in key)."""
    _, mode, prog = PROGRESSIONS[spec["progression"]]
    key = spec["key"]
    scale = [0, 2, 4, 5, 7, 9, 11] if mode == "major" else [0, 2, 3, 5, 7, 8, 10]
    used = {(key + p) % 12 for p in scale}
    chords = [(deg, qual) for bar in prog for deg, qual, _ in bar] + [turnaround(spec)]
    for deg, qual in chords:
        used |= {(key + deg + iv) % 12 for iv in QUAL[qual]}
    return used


def chord(key, deg, qual):
    root = (key + deg) % 12
    ivs = QUAL[qual]
    pcs = [(root + i) % 12 for i in ivs[1:]]   # rootless: the bass plays the root
    pcs = list(dict.fromkeys(pcs))[:5]
    return dict(root=root, qual=qual, pcs=pcs, tones={(root + i) % 12 for i in ivs},
                bass=36 + root if root <= 7 else 24 + root)


# ---------------------------------------------------------------- song spec
ARRANGEMENTS = [
    [("intro", 4), ("A", 8), ("B", 8), ("A", 8), ("break", 4), ("B", 8), ("A", 8), ("outro", 4)],
    [("intro", 4), ("A", 8), ("A", 8), ("B", 8), ("break", 4), ("A", 8), ("B", 8), ("outro", 4)],
    [("intro", 8), ("A", 8), ("B", 8), ("break", 4), ("B", 8), ("A", 8), ("outro", 4)],
    [("intro", 4), ("A", 8), ("B", 8), ("B", 8), ("break", 8), ("A", 8), ("B", 8), ("outro", 4)],
    [("intro", 4), ("A", 8), ("B", 8), ("break", 4), ("A", 8), ("B", 8), ("outro", 4)],
    [("intro", 4), ("B", 8), ("A", 8), ("break", 4), ("B", 8), ("A", 8), ("B", 8), ("outro", 4)],
]
WORDS_A = ["Rainy", "Midnight", "Sleepy", "Velvet", "Paper", "Amber", "Quiet", "Soft", "Hazy",
           "Neon", "Faded", "Warm", "Lazy", "Cloudy", "Late", "Dusty", "Blue", "Golden", "Foggy",
           "Lunar", "Slow", "Cozy", "Distant", "Mellow", "Silver", "Polaroid", "Static", "Evening",
           "Window", "Autumn", "Gentle", "Hidden", "Morning", "Old", "Pastel"]
WORDS_B = ["Notes", "Streets", "Tea", "Pages", "Rooftops", "Lanterns", "Daydream", "Commute",
           "Library", "Puddles", "Radio", "Sunday", "Study", "Café", "Cassette", "Lamplight",
           "Drizzle", "Moon", "Corners", "Letters", "Echoes", "Hours", "Thoughts", "Clouds",
           "Trains", "Steam", "Blankets", "Kites", "Ink", "Postcards", "Windows", "Balcony",
           "Sweaters", "Fireflies", "Crosswalk"]


WORDS_HIPHOP_A = ["Concrete", "Brick", "Corner", "Subway", "Basement", "Block", "Crate", "Uptown",
                  "Backpack", "Bodega", "Boombox", "Midnight", "Stoop", "Smoky", "Gritty", "Chrome",
                  "Rooftop", "Alley", "Graffiti", "Late Train", "Night Shift", "Golden Era", "Cassette",
                  "Brownstone", "Fire Escape"]
WORDS_HIPHOP_B = ["Cypher", "Loops", "Crates", "Breaks", "Kicks", "Sessions", "Tapes", "Walk",
                  "Hustle", "Grooves", "Avenue", "Cuts", "Flow", "Diggin'", "Dreams", "Static",
                  "Freestyle", "Rhymes", "Headnod", "Swing", "Chops", "Dust", "Bounce", "Blocks"]

# Per-style choices: (options, weights) or (low, high) ranges, plus mix levels in dB vs the kick.
STYLES = {
    "lofi": dict(
        bpm=(70, 91), swing=(0.54, 0.64),
        keys=(["epiano", "felt", "nylon", "vibes"], [4, 3, 2, 1.5]),
        comp=(["sustain", "rhythm", "arp"], [5, 3, 2]),
        pad=0.45, bass=(["round", "upright", "sub"], [4, 2.5, 2]), walking=0.5,
        lead=(["flute", "horn", "pluck", "whistle", "vibes", "none"], [3, 2, 2.5, 1.2, 1.5, 0.8]),
        drums=(["boombap", "laidback", "halftime", "bouncy"], [4, 3, 1.5, 2]),
        snare=(["snare", "rim", "clap"], [5, 2, 2]), hats=(["hats", "shaker"], [3, 1]),
        hat_busy=(0.05, 0.4), kick_tune=(40, 54), punch=(0.0, 0.0), ghost=0.2,
        vinyl_db=(-32, -24), rain=0.3, crunch=0.3, tone_hz=(8000, 12500), duck=0.3,
        mix=dict(keys=-2.5, pad=-13, bass=-2.0, snare=-4.0, hats=-13.0, lead=-5.0),
        words=(WORDS_A, WORDS_B)),
    # Head-nod tempos, harder swing, punchy drums up front, chopped stabs,
    # 808s and sampler crunch; melodies take a back seat.
    "hiphop": dict(
        bpm=(84, 97), swing=(0.57, 0.66),
        keys=(["epiano", "felt", "nylon", "vibes"], [4, 2.5, 1.5, 2]),
        comp=(["sustain", "rhythm", "arp"], [3, 5, 1.5]),
        pad=0.25, bass=(["round", "upright", "sub"], [3, 1.5, 4]), walking=0.25,
        lead=(["flute", "horn", "pluck", "whistle", "vibes", "none"], [1.5, 2.5, 2, 0.8, 1.5, 3]),
        drums=(["boombap", "laidback", "halftime", "bouncy"], [6, 2, 0.8, 3]),
        snare=(["snare", "rim", "clap"], [6, 0.8, 3]), hats=(["hats", "shaker"], [5, 1]),
        hat_busy=(0.2, 0.55), kick_tune=(45, 56), punch=(0.5, 1.0), ghost=0.45,
        vinyl_db=(-30, -23), rain=0.12, crunch=0.65, tone_hz=(9000, 13500), duck=0.4, drive=0.6, tuned_drums=True,
        mix=dict(keys=-6.5, pad=-16, bass=-2.0, snare=-0.5, hats=-9.5, lead=-8.0),
        words=(WORDS_HIPHOP_A, WORDS_HIPHOP_B)),
}


def pick(rng, options, weights=None):
    w = None if weights is None else np.array(weights, float) / sum(weights)
    return options[int(rng.choice(len(options), p=w))]


def make_spec(seed, style="lofi", **overrides):
    st = STYLES[style]
    rng = np.random.default_rng(seed)
    s = dict(seed=seed, style=style)

    def uni(name, nd=2):
        return round(float(rng.uniform(*st[name])), nd)

    s["progression"] = int(rng.integers(len(PROGRESSIONS)))
    s["key"] = int(rng.integers(12))
    s["bpm"] = int(rng.integers(*st["bpm"]))
    s["swing"] = uni("swing", 3)
    s["keys"] = pick(rng, *st["keys"])
    s["comp"] = pick(rng, *st["comp"])
    if s["comp"] == "arp" and s["keys"] == "epiano" and rng.random() < 0.5:
        s["comp"] = "sustain"
    s["pad"] = bool(rng.random() < st["pad"])
    s["bass"] = pick(rng, *st["bass"])
    s["walking"] = bool(s["bass"] == "upright" and rng.random() < st["walking"])
    s["lead"] = pick(rng, *st["lead"])
    if s["lead"] == "vibes" and s["keys"] == "vibes":
        s["lead"] = "flute"
    s["drums"] = pick(rng, *st["drums"])
    s["snare"] = pick(rng, *st["snare"])
    s["hats"] = pick(rng, *st["hats"])
    s["hat_busy"] = uni("hat_busy")
    s["kick_tune"] = uni("kick_tune", 1)
    s["punch"] = uni("punch")
    s["ghost"] = st["ghost"]
    s["vinyl_db"] = uni("vinyl_db", 1)
    s["rain"] = bool(rng.random() < st["rain"])
    s["wow"] = round(float(rng.uniform(0.4, 1.4)), 2)
    s["crunch"] = bool(rng.random() < st["crunch"])
    s["tone_hz"] = int(rng.uniform(*st["tone_hz"]))
    s["duck"] = st["duck"]
    s["drive"] = st.get("drive", 1.3)
    s["tuned_drums"] = st.get("tuned_drums", False)
    s["diatonic_turnaround"] = st.get("diatonic_turnaround", False)
    s["mix"] = dict(st["mix"])
    s["break"] = pick(rng, ["filter", "drums"], [3, 2] if style == "lofi" else [2, 3])
    s.update(overrides)

    bar = 240.0 / s["bpm"]
    lengths = [sum(b for _, b in a) * bar + LEAD_IN + TAIL for a in ARRANGEMENTS]
    ok = [i for i, d in enumerate(lengths) if 130 <= d <= 215]
    s.setdefault("arrangement", int(rng.choice(ok)) if ok else int(np.argmin(np.abs(np.array(lengths) - 170))))
    words_a, words_b = st["words"]
    s.setdefault("title", f"{pick(rng, words_a)} {pick(rng, words_b)}")
    s["key_name"] = KEY_NAMES[s["key"]] + ("m" if PROGRESSIONS[s["progression"]][1] == "minor" else "")
    return s


# ---------------------------------------------------------------- composition
def bar_parts(stype, i, nbars, spec):
    off = dict(keys=False, pad=False, bass=False, kick=False, snare=False, hats=False,
               rim=False, lead=False)
    if stype == "intro":
        return {**off, "keys": True, "pad": spec["pad"], "hats": i == nbars - 1}
    if stype == "A":
        return {**off, "keys": True, "bass": True, "kick": True, "snare": True, "hats": True}
    if stype == "B":
        return {**off, "keys": True, "pad": spec["pad"], "bass": True, "kick": True,
                "snare": True, "hats": True, "lead": spec["lead"] != "none"}
    if stype == "break":
        if spec["break"] == "drums":   # keys drop out for the first half
            return {**off, "keys": i >= nbars // 2, "bass": True, "kick": True, "snare": True, "hats": True}
        return {**off, "keys": True, "pad": spec["pad"], "hats": True, "rim": True}
    return {**off, "keys": True, "pad": spec["pad"], "bass": i < nbars - 1, "kick": i < nbars - 2,
            "snare": i < nbars - 2, "hats": i < nbars - 1}


def make_motif(rng, scale_len):
    rhythms = [
        [(0.5, 1.0), (1.5, 0.5), (2, 1.5), (4.5, 0.5), (5, 0.75), (6, 1.75)],
        [(0, 1.5), (2, 0.5), (2.75, 1.0), (4, 2.5), (7, 0.75)],
        [(1, 0.5), (1.5, 0.5), (2, 2.0), (5, 0.5), (5.5, 0.5), (6, 1.75)],
        [(0, 0.75), (0.75, 0.75), (1.5, 2.5), (4.5, 0.5), (5, 0.5), (5.5, 2.0)],
        [(2, 0.5), (2.5, 0.5), (3, 1.0), (4, 3.0)],
        [(0, 0.5), (0.5, 0.5), (1, 0.5), (1.5, 1.5), (4, 0.5), (4.75, 0.75), (5.5, 2.0)],
    ]
    rhythm = rhythms[int(rng.integers(len(rhythms)))]
    idx, notes = int(rng.integers(2, scale_len - 2)), []
    for pos, dur in rhythm:
        notes.append([pos, dur, idx])
        idx = int(np.clip(idx + rng.choice([-2, -1, -1, 1, 1, 2]), 0, scale_len - 1))
    return notes


def vary_motif(motif, rng, scale_len, resolve=False):
    m = [list(n) for n in motif]
    if resolve:
        m[-1][2] = max(0, m[-1][2] - 1)
        m[-1][1] += 1.0
    else:
        m[-1][2] = int(np.clip(m[-1][2] + rng.choice([-1, 1]), 0, scale_len - 1))
        if len(m) > 3 and rng.random() < 0.4:
            m.pop(int(rng.integers(1, len(m) - 1)))
    return m


def render(spec):
    rng = np.random.default_rng(spec["seed"] * 7919 + 17)
    bpm, swing, key = spec["bpm"], spec["swing"], spec["key"]
    beat = 60.0 / bpm
    bar = 4 * beat
    step = beat / 4
    _, mode, prog = PROGRESSIONS[spec["progression"]]
    sections = ARRANGEMENTS[spec["arrangement"]]
    tonic = chord(key, 0, "maj9" if mode == "major" else "m9")

    # ---- timeline: every bar's chords (with voice leading) and active parts
    timeline, prev = [], None
    for stype, nbars in sections:
        for i in range(nbars):
            bars = [list(seg) for seg in prog[i % len(prog)]]
            if stype == "outro" and i == nbars - 1:
                bars = [[0, "maj9" if mode == "major" else "m9", 4]]
            elif i == nbars - 1 and stype != "outro" and len(bars) == 1:
                bars = [[bars[0][0], bars[0][1], 2], [*turnaround(spec), 2]]
            segs, pos = [], 0
            for deg, qual, beats in bars:
                c = chord(key, deg, qual)
                c["notes"] = prev = voice(c["pcs"], prev)
                segs.append((c, pos, beats))
                pos += beats
            timeline.append(dict(stype=stype, i=i, nbars=nbars, segs=segs,
                                 parts=bar_parts(stype, i, nbars, spec)))
    total_bars = len(timeline)
    n = int((LEAD_IN + total_bars * bar + TAIL) * SR)

    def at(b, pos):
        """Seconds at beat position `pos` of bar `b`, with 16th-note swing."""
        s = pos * 4
        pair = np.floor(s / 2)
        w = s - 2 * pair
        m = w * 2 * swing if w <= 1 else 2 * swing + (w - 1) * 2 * (1 - swing)
        return LEAD_IN + b * bar + (pair * 2 + m) * step

    def human(sd):
        return rng.normal(0, sd)

    def hv(v, sd=0.07):
        return float(np.clip(v * (1 + rng.normal(0, sd)), 0.05, 1.0))

    keys_fn = KEYS[spec["keys"]]
    bass_fn = BASS[spec["bass"]]
    snare_fn = SNARES[spec["snare"]]
    rim_fn = rim
    kick_tune = spec["kick_tune"]
    if spec.get("tuned_drums"):
        # Tune the drums' pitched tones to the song's root or fifth, like a producer
        # tuning samples, so loud drums don't clash with the chords.
        def nearest(hz):
            cands = [midi_hz(m) for m in range(12, 110) if (m - key) % 12 in (0, 7)]
            return min(cands, key=lambda c: abs(np.log(c / hz)))
        kick_tune = nearest(spec["kick_tune"])
        if spec["snare"] == "snare":
            tone = nearest(200.0)
            snare_fn = lambda v, r, tone=tone: snare(v, r, tone, 1.5)
        rim_tone = nearest(1700.0)
        rim_fn = lambda v, r: rim(v, r, rim_tone)
        if spec["snare"] == "rim":
            snare_fn = rim_fn
    lead_fn, lead_lo, lead_hi = LEADS.get(spec["lead"], LEADS["flute"])
    scale = [p for p in range(lead_lo, lead_hi + 1) if (p - key) % 12 in PENTA[mode]]

    keys, pads, bss, kik, snr, hts, mel = (Bus(n) for _ in range(7))
    kick_times = []
    motif = None

    for b, info in enumerate(timeline):
        stype, i, parts = info["stype"], info["i"], info["parts"]
        last_bar = b == total_bars - 1
        phrase_end = i == info["nbars"] - 1

        for seg_idx, (c, sb, lb) in enumerate(info["segs"]):
            notes = c["notes"]
            seg_end = at(b, sb + lb)

            # ---- chords
            if parts["keys"]:
                base_vel = hv(0.55, 0.1) * (0.85 if stype in ("intro", "break") else 1.0)
                comp = spec["comp"] if not last_bar else "sustain"

                def strum(t0, dur, vel, voices):
                    order = voices if rng.random() < 0.75 else voices[::-1]
                    spread = rng.uniform(0.006, 0.018)
                    for k, note in enumerate(order):
                        f = midi_hz(note) * 2 ** (rng.normal(0, 3) / 1200)
                        v = hv(vel * (1.1 if note == max(voices) else 1.0), 0.06)
                        keys.add(keys_fn(f, dur, v, rng), t0 + k * spread + human(0.004),
                                 pan=float(np.interp(note, [48, 76], [-0.35, 0.35])))

                if comp == "sustain":
                    t0 = at(b, sb)
                    dur = lb * beat - 0.06
                    if sb == 0 and b > 0 and not last_bar and rng.random() < 0.3:
                        t0 = at(b - 1, 3.75)   # push the downbeat a swung 16th early
                        dur += at(b, 0) - t0
                    if last_bar:
                        dur = bar + TAIL - 2.0
                    strum(t0, dur, base_vel, notes)
                    if lb == 4 and not last_bar and rng.random() < 0.35:
                        strum(at(b, 2.5), 1.3 * beat, base_vel * 0.55, notes[-3:])
                elif comp == "rhythm":
                    hits = [(0, 1.4), (2.5, 1.2)] if lb == 4 else [(0, 1.6)]
                    if lb == 4 and rng.random() < 0.4:
                        hits = [(0, 1.0), (1.5, 0.8), (3, 0.9)]
                    for pos, d in hits:
                        strum(at(b, sb + pos), d * beat, base_vel * (1 if pos == 0 else 0.75), notes)
                else:  # arpeggio over swung 8ths
                    strum(at(b, sb), lb * beat, base_vel * 0.6, notes)
                    seq = notes + notes[-2:0:-1]
                    for k in range(int(lb * 2)):
                        note = seq[k % len(seq)] + 12 * (k % 4 == 3)
                        keys.add(keys_fn(midi_hz(note), 1.4 * beat, hv(base_vel * 0.7), rng),
                                 at(b, sb + k * 0.5) + human(0.006),
                                 pan=float(np.interp(note, [48, 88], [-0.4, 0.4])))
            if parts["pad"]:
                for note in notes[-3:]:
                    pads.add(pad(midi_hz(note), seg_end - at(b, sb), 0.5, rng), at(b, sb),
                             pan=float(rng.uniform(-0.5, 0.5)))

            # ---- bass
            if parts["bass"]:
                root = c["bass"]
                if seg_idx + 1 < len(info["segs"]):
                    nxt = info["segs"][seg_idx + 1][0]
                elif b + 1 < total_bars:
                    nxt = timeline[b + 1]["segs"][0][0]
                else:
                    nxt = tonic
                approach = nxt["bass"] + int(rng.choice([-1, 1, 2])) - root
                if lb == 2:
                    pattern = [(0, 0, 1.4, 1.0), (1.5, approach, 0.45, 0.6)]
                elif spec["walking"]:
                    third = 3 if (c["root"] + 3) % 12 in c["tones"] else 4
                    pattern = [(0, 0, 0.9, 1.0), (1, 7, 0.9, 0.8),
                               (2, int(rng.choice([12, third + 12, 10])), 0.9, 0.8), (3, approach, 0.9, 0.75)]
                elif spec["bass"] == "sub":
                    pattern = [[(0, 0, 2.5, 1.0), (2.75, 0, 0.5, 0.6), (3.5, 12, 0.45, 0.5)],
                               [(0, 0, 3.4, 1.0), (3.5, approach, 0.45, 0.6)]][int(rng.integers(2))]
                else:
                    pattern = [
                        [(0, 0, 1.4, 1.0), (1.75, 0, 0.25, 0.55), (2.5, 7, 0.9, 0.8), (3.5, approach, 0.45, 0.6)],
                        [(0, 0, 2.3, 1.0), (2.5, 0, 0.4, 0.6), (3, 7, 0.4, 0.7), (3.5, approach, 0.45, 0.6)],
                        [(0, 0, 3.3, 1.0), (3.5, approach, 0.45, 0.65)],
                    ][int(rng.integers(3))]
                for pos, iv, d, v in pattern:
                    bss.add(bass_fn(midi_hz(root + iv), d * beat, hv(v), rng),
                            at(b, sb + pos) + human(0.006))
        if last_bar:
            bss.add(bass_fn(midi_hz(tonic["bass"]), 3.5 * beat, 0.8, rng), at(b, 0))

        # ---- drums (16 steps per bar)
        style = spec["drums"]
        if parts["kick"]:
            hits = {"boombap": {0: 1.0, 10: 0.85}, "laidback": {0: 1.0, 11: 0.75},
                    "halftime": {0: 1.0, 14: 0.6}, "bouncy": {0: 1.0, 7: 0.7, 10: 0.8}}[style].copy()
            if style == "boombap" and rng.random() < 0.4:
                hits[7] = 0.6
            if style in ("boombap", "halftime") and rng.random() < 0.2:
                hits[3] = 0.5
            if style == "laidback" and rng.random() < 0.35:
                hits[6] = 0.55
            if style == "bouncy" and rng.random() < 0.3:
                hits[15] = 0.4
            if spec.get("style") == "hiphop" and style == "boombap" and rng.random() < 0.35:
                hits[2] = 0.55   # the classic doubled kick early in the bar
            if phrase_end and rng.random() < 0.6:
                hits[13] = 0.55
            for s, v in hits.items():
                t = at(b, s / 4) + human(0.004)
                kik.add(kick(hv(v), rng, kick_tune, spec.get("punch", 0.0)), t)
                kick_times.append(t)
        elif last_bar:
            t = at(b, 0)
            kik.add(kick(0.7, rng, kick_tune, spec.get("punch", 0.0)), t)
            kick_times.append(t)

        lag = 0.025 if style == "laidback" else 0.015   # snares sit behind the beat
        if parts["snare"]:
            for s in ((8,) if style == "halftime" else (4, 12)):
                snr.add(snare_fn(hv(0.8), rng), at(b, s / 4) + lag + human(0.005), pan=0.05)
            if rng.random() < spec.get("ghost", 0.2):
                ghosts = [7, 9, 15] if spec.get("style") == "hiphop" else [9, 15]
                snr.add(snare_fn(hv(0.18), rng), at(b, int(rng.choice(ghosts)) / 4) + lag * 0.7)
            if phrase_end and stype != "outro":
                for s, v in ((14, 0.3), (15, 0.45)):
                    snr.add(snare_fn(hv(v), rng), at(b, s / 4) + lag * 0.8 + human(0.005))
        if parts["rim"]:
            for s in (4, 12):
                snr.add(rim_fn(hv(0.5), rng), at(b, s / 4) + lag + human(0.005), pan=0.1)

        if parts["hats"]:
            busy = spec["hat_busy"] * (1.6 if stype in ("break",) or (stype == "A" and b > 24) else 1)
            for s in range(16):
                t = at(b, s / 4) + human(0.006)
                if spec["hats"] == "shaker":
                    v = (0.5 if s % 4 == 2 else 0.3 if s % 2 == 0 else 0.2) * (0.6 if stype == "intro" else 1)
                    hts.add(shaker(hv(v, 0.15), rng), t, pan=-0.25)
                elif phrase_end and s == 14 and stype not in ("intro", "outro"):
                    hts.add(hat(0.45, rng, open_=True), t, pan=0.3)
                elif s % 2 == 0:
                    v = (0.55 if s % 4 == 0 else 0.38) * (0.6 if stype == "intro" else 1)
                    hts.add(hat(hv(v, 0.12), rng), t, pan=0.3)
                elif rng.random() < busy:
                    hts.add(hat(hv(0.2, 0.15), rng), t, pan=0.3)

        # ---- lead melody: a 2-bar motif with variations, new motif each section
        if parts["lead"]:
            if i == 0:
                motif = make_motif(rng, len(scale))
            if i % 2 == 0:
                half = (i // 2) % 4
                m = motif if half in (0, 2) else vary_motif(motif, rng, len(scale), resolve=(half == 3))
                for pos, d, idx in m:
                    bb = b + int(pos // 4)
                    if bb >= total_bars:
                        continue
                    segs = timeline[bb]["segs"]
                    c = segs[-1][0] if (pos % 4) >= segs[-1][1] else segs[0][0]
                    note = scale[idx]
                    # avoid a semitone rub against the chord (a b9 above a chord tone)
                    if note % 12 not in c["tones"] and (note - 1) % 12 in c["tones"]:
                        note = scale[min(idx + 1, len(scale) - 1)]
                    mel.add(lead_fn(midi_hz(note), d * beat, hv(0.5, 0.1), rng),
                            at(b, pos) + human(0.01), pan=0.15)

    # ---------------------------------------------------------------- mixing
    times = np.arange(n) / SR

    # Filter sweep on the keys: muffled intro opening up, darker filter break, closing outro.
    curve, t_acc = [(0.0, 0.0)], LEAD_IN
    for stype, nbars in sections:
        start, end = t_acc, t_acc + nbars * bar
        if stype == "intro":
            curve.append((end - 0.2 * bar, 1.0))
        elif stype == "break" and spec["break"] == "filter":
            curve += [(start, 1.0), (start + 0.5 * bar, 0.35), (end - 0.5 * bar, 0.4), (end, 1.0)]
        elif stype == "outro":
            curve.append((start + bar, 1.0))
        t_acc = end
    curve.append((t_acc + TAIL, 0.0))
    xs, ys = zip(*curve)
    bright = np.interp(times, np.maximum.accumulate(np.array(xs)), ys)
    keys_mix = keys.buf
    if spec["crunch"]:
        keys_mix = crunch(keys_mix)
    keys_mix = lowpass(keys_mix, 550) * (1 - bright) + keys_mix * bright

    # Gentle sidechain "pump": keys, pad, bass and lead dip slightly on every kick.
    duck = np.ones(n)
    seg = 1 - spec.get("duck", 0.3) * np.exp(-np.arange(int(0.6 * SR)) / SR / 0.15)
    for t in kick_times:
        s = int(t * SR)
        e = min(n, s + len(seg))
        duck[s:e] = np.minimum(duck[s:e], seg[: e - s])

    # Balance each part to a fixed level relative to the kick, so every
    # instrument combination comes out mixed the same way.
    ref = active_rms(kik.buf) if kick_times else 0.1

    def level(x, db):
        return x * (ref * 10 ** (db / 20) / active_rms(x)) if np.any(x) else x

    mx = {**STYLES["lofi"]["mix"], **spec.get("mix", {})}
    keys_l = level(keys_mix, mx["keys"])
    pad_l = level(lowpass(pads.buf, 1500) * bright, mx["pad"])
    bass_l = level(bss.buf, mx["bass"])
    snr_l = level(snr.buf, mx["snare"])
    hts_l = level(hts.buf, mx["hats"])
    mel_l = level(mel.buf, mx["lead"])
    music = (keys_l * duck + pad_l * duck + bass_l * duck + kik.buf + snr_l + hts_l + mel_l * duck)
    send = highpass(keys_l * 0.35 + snr_l * 0.25 + mel_l * 0.6 + pad_l * 0.5, 250)
    music += reverb(send, rng) * 0.35

    music = tape_wow(music, rng.uniform(0, 2 * np.pi), spec["wow"])
    music = highpass(music, 30)
    music /= np.max(np.abs(music))
    drive = spec.get("drive", 1.3)
    music = np.tanh(drive * music) / np.tanh(drive)   # soft tape saturation (less for punchier drums)
    music = lowpass(music, spec["tone_hz"])

    texture = vinyl(n, rng)
    texture *= rms(music) * 10 ** (spec["vinyl_db"] / 20) / rms(texture)
    if spec["rain"]:
        r = rain(n, rng)
        texture += r * rms(music) * 10 ** (-27 / 20) / rms(r)
    mix = music + texture

    fade_in = np.minimum(1, times / 0.6)
    fade_out = np.clip((times[-1] - times) / 6.0, 0, 1) ** 1.5
    return master(mix * fade_in * fade_out)


def master(x, target=TARGET_LUFS, ceiling_db=-1.0):
    """Normalize to a target loudness with a gentle soft-knee peak limiter."""
    try:
        import pyloudnorm
        meter = pyloudnorm.Meter(SR)
        loudness = lambda y: meter.integrated_loudness(y.T)
    except ImportError:  # fallback: RMS approximation
        loudness = lambda y: 20 * np.log10(rms(y)) - 0.7
    ceiling = 10 ** (ceiling_db / 20)
    knee = 0.6 * ceiling

    def limit(y):
        a = np.abs(y)
        over = a > knee
        out = y.copy()
        out[over] = np.sign(y[over]) * (knee + (ceiling - knee) * np.tanh((a[over] - knee) / (ceiling - knee)))
        return out

    for _ in range(3):   # limiting lowers loudness a little, so iterate
        x = limit(x * 10 ** ((target - loudness(x)) / 20))
    return x


# ---------------------------------------------------------------- export
def write_wav24(path, x, rng):
    dither = (rng.random(x.shape) - rng.random(x.shape)) / 8388607  # TPDF dither
    ints = np.clip(np.round((x + dither) * 8388607), -8388608, 8388607).astype("<i4")
    frames = ints.T.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(frames)


def write_mp3(wav_path, mp3_path, tags=None):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found, skipping MP3. Install it (free) with:\n"
              "  winget install Gyan.FFmpeg     (Windows)\n"
              "  brew install ffmpeg            (macOS)\n"
              "  sudo apt install ffmpeg        (Linux)")
        return False
    meta = []
    for k, v in (tags or {}).items():
        meta += ["-metadata", f"{k}={v}"]
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path),
                    "-codec:a", "libmp3lame", "-b:a", "320k", *meta, str(mp3_path)], check=True)
    return True


def build_track(spec, wav_path, mp3_path=None, tags=None):
    mix = render(spec)
    write_wav24(wav_path, mix, np.random.default_rng(spec["seed"] + 1))
    if mp3_path:
        write_mp3(wav_path, mp3_path, tags)
    return mix.shape[1] / SR


def describe(s):
    return (f"{s['title']}: {s['key_name']}, {s['bpm']} BPM, '{PROGRESSIONS[s['progression']][0]}' "
            f"progression, {s['keys']} ({s['comp']}){' + pad' if s['pad'] else ''}, {s['bass']} bass"
            f"{' (walking)' if s['walking'] else ''}, {s['lead']} lead, {s['drums']} drums with {s['snare']}"
            f"{', rain' if s['rain'] else ''}{', sampler crunch' if s['crunch'] else ''}")


def main():
    ap = argparse.ArgumentParser(description="Synthesize an original lo-fi hip-hop track.")
    ap.add_argument("--seed", type=int, default=7, help="each seed is a different song")
    ap.add_argument("--bpm", type=int, help="override the tempo")
    ap.add_argument("--style", default="lofi", choices=sorted(STYLES))
    ap.add_argument("--out", default="lofi_beat", help="output file name without extension")
    ap.add_argument("--no-mp3", action="store_true")
    args = ap.parse_args()

    spec = make_spec(args.seed, args.style, **({"bpm": args.bpm} if args.bpm else {}))
    print("Rendering", describe(spec))
    wav = Path(args.out).with_suffix(".wav")
    secs = build_track(spec, wav, None if args.no_mp3 else wav.with_suffix(".mp3"),
                       {"title": spec["title"], "artist": "lofi_beat.py"})
    print(f"Wrote {wav} ({int(secs // 60)}:{secs % 60:04.1f})")
    Path(args.out).with_suffix(".json").write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
