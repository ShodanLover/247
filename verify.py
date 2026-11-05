#!/usr/bin/env python3
"""verify.py - checks that lofi_beat.wav / .mp3 are valid, decodable and sane.

Usage: python verify.py [basename]     (default: lofi_beat)
"""
import json
import shutil
import subprocess
import sys
import wave

import numpy as np
from scipy.signal import spectrogram

base = sys.argv[1] if len(sys.argv) > 1 else "lofi_beat"
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok &= bool(cond)


# ---- WAV
with wave.open(f"{base}.wav", "rb") as w:
    ch, width, sr, nframes = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    raw = np.frombuffer(w.readframes(nframes), dtype=np.uint8).reshape(-1, 3)
ints = (raw[:, 0].astype(np.int32) | raw[:, 1].astype(np.int32) << 8
        | raw[:, 2].astype(np.int32) << 16)
ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
x = (ints / 8388608.0).reshape(-1, ch).T
dur = nframes / sr
peak_db = 20 * np.log10(np.max(np.abs(x)))
rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)))

print(f"WAV: {ch} ch, {width * 8}-bit, {sr} Hz, {int(dur // 60)}:{dur % 60:04.1f}")
check(ch == 2 and sr == 44100 and width == 3, "WAV format is 24-bit / 44.1 kHz stereo")
check(120 <= dur <= 240, f"duration {dur:.1f}s is within 2-4 minutes")
check(np.all(np.isfinite(x)), "no NaN/inf samples")
check(-3 < peak_db < 0, f"peak {peak_db:.2f} dBFS (no clipping)")
check(-24 < rms_db < -10, f"overall RMS {rms_db:.1f} dBFS")
check(abs(np.mean(x)) < 1e-3, "no DC offset")
corr = np.corrcoef(x[0], x[1])[0, 1]
check(0.5 < corr < 0.999, f"stereo L/R correlation {corr:.2f} (wide but mono-safe)")

# Loudness envelope: intro should be quieter than the body, ending fades to ~silence.
env = np.sqrt(np.mean(x.reshape(2, -1, sr // 2)[:, : int(dur * 2)] ** 2, axis=(0)))  # 0.5s blocks
db = 20 * np.log10(env + 1e-9)
intro, body, last = db[2:20].mean(), db[40:300].mean(), db[-2:].mean()
check(intro < body - 3, f"intro ({intro:.1f} dB) softer than body ({body:.1f} dB)")
check(last < body - 30, f"ending fades out ({last:.1f} dB)")

# ---- MP3
mp3 = f"{base}.mp3"
ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
if ffprobe and ffmpeg:
    info = json.loads(subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration,bit_rate:stream=codec_name,sample_rate,channels",
         "-of", "json", mp3], capture_output=True, text=True, check=True).stdout)
    st, fmt = info["streams"][0], info["format"]
    print(f"MP3: {st['codec_name']}, {st['sample_rate']} Hz, {st['channels']} ch, "
          f"{int(fmt['bit_rate']) // 1000} kbps, {float(fmt['duration']):.1f}s")
    check(st["codec_name"] == "mp3", "MP3 codec")
    check(abs(float(fmt["duration"]) - dur) < 0.2, "MP3 duration matches WAV")
    dec = subprocess.run([ffmpeg, "-v", "error", "-i", mp3, "-f", "f32le", "-ac", "2", "-"],
                         capture_output=True, check=True)
    check(dec.stderr == b"", "MP3 decodes with no errors")
    y = np.frombuffer(dec.stdout, dtype="<f4").reshape(-1, 2).T
    y_rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)))
    check(abs(y_rms - rms_db) < 0.5, f"decoded MP3 level {y_rms:.1f} dBFS matches WAV")
else:
    check(False, "ffmpeg/ffprobe not found - cannot verify MP3")

# ---- waveform + spectrogram image for a visual check
try:
    from PIL import Image

    W, H = 1400, 500
    mono = x.mean(axis=0)
    cols = np.array_split(np.abs(mono), W)
    amp = np.array([c.max() for c in cols])
    img = np.full((H, W, 3), 18, np.uint8)
    half = 100
    for i, a in enumerate(amp):
        h = int(a * half)
        img[half - h:half + h, i] = (120, 200, 255)
    f, t, S = spectrogram(mono[::2], fs=sr / 2, nperseg=2048, noverlap=1024)
    S = 10 * np.log10(S[f < 8000] + 1e-12)
    S = np.clip((S - (S.max() - 80)) / 80, 0, 1)[::-1]
    spec = np.array(Image.fromarray((S * 255).astype(np.uint8)).resize((W, H - 2 * half - 10)))
    img[2 * half + 10:, :, 0] = spec
    img[2 * half + 10:, :, 1] = (spec * 0.6).astype(np.uint8)
    img[2 * half + 10:, :, 2] = (255 - spec) // 3
    Image.fromarray(img).save(f"{base}_overview.png")
    print(f"Saved {base}_overview.png (waveform + 0-8 kHz spectrogram)")
except ImportError:
    print("Pillow not installed - skipping overview image")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
