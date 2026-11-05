# 247

A 24/7 Christmas fireplace livestream for YouTube: a live-rendered cozy room, a countdown to
Christmas in light strings, and a music player with a "now playing" card.

Everything lives in [`christmas/`](christmas/). Its [README](christmas/README.md) covers where to
get safe music, how to build the playlist, and how to set up OBS.

## Setting up on a new PC

Clone the repo into `D:\247`:

```bash
git clone https://github.com/ShodanLover/247.git D:\247
```

Install Python 3 and ffmpeg (`winget install Gyan.FFmpeg`). Then, from `D:\247`, create the Python
environment:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install numpy scipy pyloudnorm
```

Music isn't stored in the repo, so download your tracks into `christmas\music\` on that machine.
