#!/usr/bin/env python3
"""
build_playlist.py - prepares your downloaded music for the stream.

1. Finds every audio file under music/ (subfolders are fine; folders starting with _ are skipped).
2. Reads each track's title and artist from its tags (or from "Artist - Title.mp3" file names).
   You can override or add details in music/credits.csv (see below).
3. Evens out the volume: every track is loudness-normalized to the same level (-16 LUFS,
   ffmpeg's two-pass loudnorm), so nothing jumps out between songs on a 24/7 stream.
4. Writes music_ready/*.mp3, playlist.js (read by the scene's player), and credits.txt
   (paste into your stream description; CC-BY tracks require it).

Usage:
    python build_playlist.py
    python build_playlist.py --music music --lufs -16

music/credits.csv (optional, any subset of columns), keyed by file name:
    file,title,artist,credit
    Jingle Bells.mp3,Jingle Bells,Kevin MacLeod,"Kevin MacLeod (incompetech.com), CC BY 4.0"
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIO = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
if not (ffmpeg and ffprobe):
    sys.exit("ffmpeg not found. Install it (free): winget install Gyan.FFmpeg")


def probe(path):
    out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration:format_tags",
                          "-of", "json", str(path)], capture_output=True, text=True)
    if out.returncode:
        return None
    fmt = json.loads(out.stdout)["format"]
    tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}
    return float(fmt.get("duration", 0)), tags


def guess_from_name(stem):
    stem = re.sub(r"^\d+\s*[-.]\s*", "", stem)          # drop "01 - " track numbers
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return title.strip(), artist.strip()
    return stem.replace("_", " ").strip(), ""


def normalize(src, dst, lufs):
    """Measure loudness (EBU R128), apply one fixed gain to hit the target, and catch any
    peaks that would clip with a limiter. A fixed gain keeps each song's own dynamics."""
    first = subprocess.run([ffmpeg, "-hide_banner", "-i", str(src), "-af",
                            "loudnorm=print_format=json", "-f", "null", "-"], capture_output=True, text=True)
    err = first.stderr
    m = json.loads(err[err.rindex("{"):err.rindex("}") + 1])   # the measurement block ffmpeg prints last
    gain = lufs - float(m["input_i"])
    af = f"volume={gain:.2f}dB,alimiter=limit=0.79:level=disabled"     # 0.79 = -2 dBFS: headroom for MP3 encoding
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(src), "-af", af, "-ar", "44100",
                    "-c:a", "libmp3lame", "-b:a", "256k", "-map_metadata", "-1", str(dst)], check=True)
    return float(m["input_i"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", default=str(HERE / "music"))
    ap.add_argument("--out", default=str(HERE / "music_ready"))
    ap.add_argument("--lufs", type=float, default=-16.0)
    args = ap.parse_args()

    music, out = Path(args.music), Path(args.out)
    music.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    overrides = {}
    if (music / "credits.csv").exists():
        with open(music / "credits.csv", newline="", encoding="utf-8-sig") as f:
            overrides = {row["file"].strip(): row for row in csv.DictReader(f) if row.get("file")}

    files = sorted(p for p in music.rglob("*") if p.suffix.lower() in AUDIO
                   and not any(part.startswith("_") for part in p.relative_to(music).parts))
    tracks, keep = [], set()
    for p in files:
        info = probe(p)
        if not info or info[0] < 20:
            print(f"skip (unreadable or too short): {p.name}")
            continue
        dur, tags = info
        title, artist = tags.get("title", ""), tags.get("artist", "")
        if not title:
            title, guessed = guess_from_name(p.stem)
            artist = artist or guessed
        o = overrides.get(p.name, {})
        title, artist = o.get("title") or title, o.get("artist") or artist
        credit = o.get("credit") or ""
        # cache key: re-normalize only when the source file or target changes
        key = hashlib.sha1(f"{p.relative_to(music)}|{p.stat().st_size}|{p.stat().st_mtime}|{args.lufs}".encode()).hexdigest()[:10]
        dst = out / f"{re.sub(r'[^A-Za-z0-9]+', '_', p.stem)[:60]}_{key}.mp3"
        keep.add(dst.name)
        if not dst.exists():
            src_lufs = normalize(p, dst, args.lufs)
            print(f"normalized {p.name}  ({src_lufs:.1f} -> {args.lufs:.0f} LUFS)")
        rel = Path(os.path.relpath(dst, HERE)).as_posix()       # relative to playlist.js, which the scene loads
        tracks.append(dict(file=rel, title=title, artist=artist or "Unknown artist",
                           credit=credit, seconds=round(dur, 1), source=str(p.relative_to(music))))
    for old in out.glob("*.mp3"):            # drop files for tracks that were removed
        if old.name not in keep:
            old.unlink()

    (HERE / "playlist.js").write_text(
        "// Generated by build_playlist.py; do not edit by hand.\nwindow.PLAYLIST = "
        + json.dumps(tracks, indent=1, ensure_ascii=False) + ";\n", encoding="utf-8")
    lines = ["Music in this stream:", ""]
    for t in sorted(tracks, key=lambda t: (t["artist"].lower(), t["title"].lower())):
        lines.append(f"\"{t['title']}\" by {t['artist']}" + (f" | {t['credit']}" if t["credit"] else ""))
    (HERE / "credits.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hours = sum(t["seconds"] for t in tracks) / 3600
    print(f"\n{len(tracks)} tracks, {hours:.1f} hours. Wrote playlist.js and credits.txt")


if __name__ == "__main__":
    main()
