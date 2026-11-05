#!/usr/bin/env python3
"""verify_album.py - checks every rendered track, the playlist, and that tracks really differ.

Usage: python verify_album.py [tracks_dir]     (default: tracks)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyloudnorm

import lofi_beat as lb
from analyze_audio import chroma, read_wav

out = Path(sys.argv[1] if len(sys.argv) > 1 else "tracks")
specs = json.loads((out / "album.json").read_text(encoding="utf-8"))
ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
        print("FAIL ", msg)
    return cond


rows, chromas = [], []
for s in specs:
    name = f"{s['track']:02d} - {s['title']}"
    wav, mp3 = out / "wav" / f"{name}.wav", out / f"{name}.mp3"
    if not check(wav.exists() and mp3.exists(), f"{name}: files missing"):
        continue
    x, sr = read_wav(wav)
    dur = x.shape[1] / sr
    lufs = pyloudnorm.Meter(sr).integrated_loudness(x.T)
    peak = 20 * np.log10(np.abs(x).max())
    mode = lb.PROGRESSIONS[s["progression"]][1]
    c, _ = chroma(x, sr)
    chromas.append(c)
    # "in key" = the scale plus any borrowed notes the song's chords use on purpose (e.g. dorian 6th)
    key_fit = float(sum(c[p] for p in lb.pitch_classes_used(s)))

    # no dropouts: every 2 s block between intro and outro has signal
    blk = 2 * sr
    body = x[:, 4 * sr: -12 * sr]
    nb = body.shape[1] // blk
    block_db = 20 * np.log10(np.sqrt(np.mean(body[:, : nb * blk].reshape(2, nb, blk) ** 2, axis=(0, 2))) + 1e-9)

    probe = json.loads(subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration,bit_rate:format_tags=title,track",
         "-of", "json", str(mp3)], capture_output=True, text=True, check=True).stdout)["format"]
    dec = subprocess.run([ffmpeg, "-v", "error", "-i", str(mp3), "-f", "null", "-"], capture_output=True, text=True)

    check(120 <= dur <= 240, f"{name}: duration {dur:.0f}s")
    check(abs(lufs - lb.TARGET_LUFS) < 0.5, f"{name}: loudness {lufs:.1f} LUFS")
    check(peak < -0.9, f"{name}: peak {peak:.2f} dBFS")
    # Not 100%: a note's 5th harmonic sounds a (slightly flat) major third above it, which can fall
    # outside the key; bright plucked/struck timbres (e.g. nylon arpeggios) reach ~94% from that alone.
    check(key_fit > 0.92, f"{name}: only {key_fit:.0%} of pitched energy in key")
    check(block_db.min() > -45, f"{name}: dropout ({block_db.min():.0f} dB block)")
    check(abs(float(probe["duration"]) - dur) < 0.2, f"{name}: mp3 duration mismatch")
    check(int(probe["bit_rate"]) > 300000, f"{name}: mp3 bitrate {probe['bit_rate']}")
    check(probe.get("tags", {}).get("title") == s["title"], f"{name}: mp3 title tag")
    check(dec.returncode == 0 and not dec.stderr.strip(), f"{name}: mp3 decode errors")
    rows.append((s, dur, lufs, peak, key_fit))

print(f"\n{'#':>2}  {'title':<22}{'key':<5}{'bpm':>4}  {'len':>5}  {'LUFS':>5}  {'peak':>6}  {'in key':>6}  sound")
for s, dur, lufs, peak, fit in rows:
    print(f"{s['track']:>2}  {s['title']:<22}{s['key_name']:<5}{s['bpm']:>4}  {int(dur // 60)}:{int(dur % 60):02d}  "
          f"{lufs:5.1f}  {peak:6.2f}  {fit:6.0%}  {s['keys']}/{s['bass']}/{s['lead']}/{s['drums']}")

# Variety: unique titles and key+progression pairs, spread of tempo and instruments,
# and no two tracks with near-identical pitch content AND tempo AND instruments.
check(len({s["title"] for s in specs}) == len(specs), "titles are not unique")
check(len({(s["key"], s["progression"]) for s in specs}) == len(specs), "a key+progression pair repeats")
sig = [(s["keys"], s["bass"], s["lead"], s["drums"], s["comp"]) for s in specs]
check(len(set(sig)) == len(specs), "two tracks share the exact same instrumentation")
C = np.array(chromas)
sim = C @ C.T / np.outer(np.linalg.norm(C, axis=1), np.linalg.norm(C, axis=1))
np.fill_diagonal(sim, 0)
i, j = np.unravel_index(np.argmax(sim), sim.shape)
print(f"\nTempo range {min(s['bpm'] for s in specs)}-{max(s['bpm'] for s in specs)} BPM, "
      f"{len({s['bpm'] for s in specs})} distinct tempos; "
      f"{len(set(sig))} distinct instrument setups; "
      f"most similar harmony: #{i + 1} and #{j + 1} (cosine {sim[i, j]:.2f})")

# Playlist resolves
m3u = (out / "playlist.m3u").read_text(encoding="utf-8").splitlines()
entries = [l for l in m3u if l and not l.startswith("#")]
check(len(entries) == len(specs) and all((out / e).exists() for e in entries), "playlist entries missing")
total = sum(r[1] for r in rows)
print(f"Playlist: {len(entries)} entries, {int(total // 60)} min total")

print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
