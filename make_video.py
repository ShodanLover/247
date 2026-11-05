#!/usr/bin/env python3
"""
make_video.py - turns the rendered loop frames into a YouTube-ready MP4.

1. render/frames/f_####.png -> render/loop.mp4   (bloom, vignette, film grain)
2. loop.mp4 repeated for the length of the audio + the beat -> <out>.mp4
   (fades in from black and out at the end, 1080p H.264 + AAC 320k)

Usage:
    python make_video.py                                   # lofi_beat.wav -> rainy_night_lofi.mp4
    python make_video.py --audio mix.wav --out my_video
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default=str(HERE / "render" / "frames"))
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--audio", default=str(HERE / "lofi_beat.wav"))
ap.add_argument("--out", default=str(HERE / "rainy_night_lofi"))
ap.add_argument("--crf", type=int, default=21, help="lower = higher quality, bigger file")
args = ap.parse_args()

ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
if not ffmpeg:
    sys.exit("ffmpeg not found. Install it (free): winget install Gyan.FFmpeg")


def run(cmd):
    print(">", " ".join(str(c) for c in cmd[:6]), "...")
    subprocess.run([str(c) for c in cmd], check=True)


# 1. Encode the loop with a soft bloom (blurred highlights screened on top),
#    a vignette and animated film grain.
loop = HERE / "render" / "loop.mp4"
post = ("format=gbrp,split[base][hi];"   # blend in RGB; in YUV the screen blend tints everything
        "[hi]colorlevels=rimin=0.55:gimin=0.55:bimin=0.55,gblur=sigma=28[glow];"
        "[base][glow]blend=all_mode=screen:all_opacity=0.45,"
        "vignette=angle=PI/4.5,noise=alls=3:allf=t,format=yuv420p")
run([ffmpeg, "-y", "-loglevel", "error", "-framerate", args.fps,
     "-i", Path(args.frames) / "f_%04d.png", "-vf", post,
     "-c:v", "libx264", "-preset", "slow", "-crf", args.crf, "-g", args.fps * 2, loop])

# 2. Repeat the loop for the whole track and add the audio.
dur = float(json.loads(subprocess.run(
    [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", args.audio],
    capture_output=True, text=True, check=True).stdout)["format"]["duration"])
out = Path(args.out).with_suffix(".mp4")
run([ffmpeg, "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", loop, "-i", args.audio,
     "-map", "0:v", "-map", "1:a", "-t", f"{dur:.3f}",
     "-vf", f"fade=in:st=0:d=2,fade=out:st={dur - 3:.3f}:d=3",
     "-c:v", "libx264", "-preset", "medium", "-crf", args.crf, "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", out])
print(f"Wrote {loop}\nWrote {out}  ({int(dur // 60)}:{dur % 60:04.1f})")
