#!/usr/bin/env python3
"""
make_album.py - renders a batch of distinct lo-fi tracks plus a VLC playlist.

Each track cycles through the chord progressions and spreads keys around the
circle of fifths, so no two tracks share a key + progression; tempo,
instruments, drums, structure and textures come from the track's seed.

Usage:
    python make_album.py                      # 30 tracks -> tracks/
    python make_album.py --count 10 --start-seed 5000 --out tracks_vol2
Outputs:
    tracks/NN - Title.mp3      320 kbps, tagged (title / album / track number)
    tracks/wav/NN - Title.wav  24-bit masters
    tracks/playlist.m3u        open in VLC (enable loop for a 24/7 stream)
    tracks/tracklist.txt       timestamps, ready for a stream description
    tracks/album.json          every track's full recipe (re-render any one exactly)
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lofi_beat as lb


def render_one(job):
    spec, wav, mp3, tags = job
    secs = lb.build_track(spec, wav, mp3, tags)
    return spec, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--out", default="tracks")
    ap.add_argument("--album", default="Rainy Night Lofi Vol. 1")
    ap.add_argument("--style", default="lofi", choices=sorted(lb.STYLES))
    ap.add_argument("--workers", type=int, default=3, help="parallel renders (~2 GB RAM each)")
    ap.add_argument("--from-recipes", action="store_true",
                    help="re-render the exact songs in <out>/album.json instead of new ones")
    ap.add_argument("--tuned-drums", action="store_true", help="tune drum tones to each song's key")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    n_prog = len(lb.PROGRESSIONS)
    specs, titles = [], set()
    if args.from_recipes:
        specs = json.loads((out / "album.json").read_text(encoding="utf-8"))
        args.count = len(specs)
    for i in range(0 if args.from_recipes else args.count):
        seed = args.start_seed + i
        spec = lb.make_spec(seed, args.style, progression=i % n_prog, key=(i * 7 + 3) % 12)
        bump = 0
        while spec["title"] in titles:   # keep titles unique
            bump += 1
            spec = lb.make_spec(seed, args.style, progression=i % n_prog, key=(i * 7 + 3) % 12,
                                title=lb.make_spec(seed + 7919 * bump, args.style)["title"])
        titles.add(spec["title"])
        spec["track"] = i + 1
        specs.append(spec)

    if args.tuned_drums:
        for s in specs:
            s["tuned_drums"] = True

    jobs = []
    for s in specs:
        name = f"{s['track']:02d} - {s['title']}"
        tags = {"title": s["title"], "album": args.album, "artist": "lofi_beat.py",
                "track": f"{s['track']}/{args.count}", "genre": "Lo-Fi Hip Hop"}
        jobs.append((s, out / "wav" / f"{name}.wav", out / f"{name}.mp3", tags))

    done = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render_one, j) for j in jobs]
        for f in as_completed(futures):
            spec, secs = f.result()
            done[spec["track"]] = secs
            print(f"[{len(done):2d}/{args.count}] {spec['track']:02d} {lb.describe(spec)} "
                  f"({int(secs // 60)}:{secs % 60:04.1f})", flush=True)

    # Playlist, tracklist and recipes
    m3u, lines, t = ["#EXTM3U"], [], 0.0
    for s in specs:
        secs = done[s["track"]]
        name = f"{s['track']:02d} - {s['title']}"
        m3u += [f"#EXTINF:{int(round(secs))},{s['title']}", f"{name}.mp3"]
        h, m, sec = int(t // 3600), int(t % 3600 // 60), int(t % 60)
        lines.append(f"{h}:{m:02d}:{sec:02d}  {s['title']}" if h else f"{m:02d}:{sec:02d}  {s['title']}")
        s["seconds"] = round(secs, 2)
        t += secs
    (out / "playlist.m3u").write_text("\n".join(m3u) + "\n", encoding="utf-8")
    (out / "tracklist.txt").write_text(
        f"{args.album}\n{len(specs)} tracks, {int(t // 3600)}h {int(t % 3600 // 60)}m\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    (out / "album.json").write_text(json.dumps(specs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {len(specs)} tracks, total {int(t // 60)} min. Playlist: {out / 'playlist.m3u'}")


if __name__ == "__main__":
    main()
