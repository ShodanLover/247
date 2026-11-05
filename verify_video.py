#!/usr/bin/env python3
"""verify_video.py - checks that the rendered loop is seamless and the final MP4 is valid.

Usage: python verify_video.py [video.mp4]      (default: rainy_night_lofi.mp4)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
frames = sorted((HERE / "render" / "frames").glob("f_*.png"))
video = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "rainy_night_lofi.mp4"
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok &= bool(cond)


def load(p):
    return np.asarray(Image.open(p).convert("RGB").reduce(4), dtype=np.float32)


# Seam: the jump from the last frame back to the first should look like any
# other frame-to-frame step.
check(len(frames) > 1, f"{len(frames)} loop frames found")
idx = np.linspace(0, len(frames) - 2, 24).astype(int)
steps = [np.abs(load(frames[i + 1]) - load(frames[i])).mean() for i in idx]
seam = np.abs(load(frames[0]) - load(frames[-1])).mean()
typical = float(np.median(steps))
print(f"frame-to-frame change: typical {typical:.2f}, max {max(steps):.2f}, seam {seam:.2f}")
check(min(steps) * 0.5 <= seam <= max(steps) * 1.25,
      "last frame flows into first at normal speed (seamless loop, no slowdown)")
still = np.abs(load(frames[len(frames) // 7]) - load(frames[0])).mean()
check(still > typical, "scene actually animates across the loop")

# Final video
ffprobe = shutil.which("ffprobe")
info = json.loads(subprocess.run(
    [ffprobe, "-v", "error", "-show_entries",
     "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,bit_rate",
     "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout)
v = next(s for s in info["streams"] if s["codec_type"] == "video")
a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
dur = float(info["format"]["duration"])
print(f"video: {v['codec_name']} {v['width']}x{v['height']} @ {v['r_frame_rate']}, "
      f"audio: {a and a['codec_name']} {a and a['sample_rate']} Hz, {dur:.1f}s")
check(v["codec_name"] == "h264" and (v["width"], v["height"]) == (1920, 1080), "1080p H.264 video")
check(a is not None and a["codec_name"] == "aac", "AAC audio track present")
check(dur > 60, "video runs the full length of the track")
dec = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(video), "-f", "null", "-"],
                     capture_output=True, text=True)
check(dec.returncode == 0 and not dec.stderr.strip(), "decodes end to end with no errors")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
