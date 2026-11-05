# 24/7 Christmas Fireplace Stream

A live-rendered cozy Christmas room: a crackling fire, a twinkling tree, and snow falling outside
the window. A countdown to Christmas is spelled out in real-looking light strings across the top
of the wall, and a "now playing" card in the bottom-right corner shows the current song and artist.
The scene is a web page, so OBS streams it directly. Nothing visibly loops, because the fire, snow
and lights are generated live.

| File | What it is |
|---|---|
| `scene/index.html` | The scene, the countdown and the music player, all in one page |
| `music/` | Put your downloaded songs here (subfolders are fine) |
| `music/credits.csv` | Optional: fix a song's title or artist, or add its required credit line |
| `build_playlist.py` | Prepares the music, evens out volume, and writes `playlist.js` and `credits.txt` |
| `credits.txt` | Generated credits to paste into your stream description |

## 1. Get music that is safe to stream

Only use music whose license you can point to. These sources are free; check the license on each
track's own page before you use it.

| Source | What's there | Rules |
|---|---|---|
| [YouTube Audio Library](https://studio.youtube.com) (YouTube Studio → Audio Library) | Search "Christmas", "holiday", "jingle"; some lo-fi | Made for YouTube. Filter for "Attribution not required" to keep it simple; otherwise copy the credit line it gives you. YouTube's help pages don't clearly say whether these licenses cover 24/7 livestreams, so check before you rely on it. |
| [Kevin MacLeod / incompetech.com](https://incompetech.com) | Classic carols, instrumental | CC BY 4.0: you **must** credit, e.g. `"Deck the Halls" Kevin MacLeod (incompetech.com), Licensed under CC BY 4.0`. Put this in `credits.csv`. |
| [Pixabay Music](https://pixabay.com/music/) | Lots of "lofi christmas" and holiday tracks | Free for commercial use. **Skip any track marked "Content ID registered"**, since those can still get claimed. Keep the license certificate it offers when you're logged in. |
| [Musopen](https://musopen.org) | Public-domain recordings of classical music (Nutcracker, etc.) | Use only recordings whose page says Public Domain. An old composition doesn't free a modern recording of it. |

**Avoid:**
- Songs ripped from YouTube or Spotify
- "No copyright music" channels, which re-upload other people's work
- Anything from a subscription library you might cancel
- Modern Christmas hits like Mariah Carey, Wham!, Rudolph, Frosty, or "Let It Snow"

**How much music:** for a 24/7 stream, aim for 4–8 hours, roughly 60–120 songs, so repeats aren't
obvious. The player shuffles, never repeats a song until most of the others have played, and
crossfades between them.

## 2. Build the playlist

```bash
python build_playlist.py
```

The script:
- reads each song's title and artist from its tags, or from file names like `Artist - Title.mp3`
- normalizes every track to the same loudness (-16 LUFS) so nothing jumps out
- writes `playlist.js`, which the scene reads, and `credits.txt`

Run it again whenever you add or remove songs. Songs it has already processed are skipped, so
re-running is quick.

If a title or artist comes out wrong, or a song needs a credit line, add a row to `music/credits.csv`:

```
file,title,artist,credit
Deck the Halls.mp3,Deck the Halls,Kevin MacLeod,"Kevin MacLeod (incompetech.com), CC BY 4.0"
```

Paste `credits.txt` into your stream description. That's how CC-BY credit requirements get met.

## 3. Set up OBS

1. Sources → **+** → **Browser**.
2. Tick **Local file**, then pick `scene/index.html`.
3. Set the width to **1920**, the height to **1080**, and FPS to **30** (or 60).
4. Tick **Control audio via OBS**. The music then goes into the stream and shows in OBS's audio mixer.
5. Untick **Shutdown source when not visible** so the music never stops.

To preview the page in a normal browser, click once to start the sound. Browsers block autoplay;
OBS doesn't.

## Scene options

Add these to the end of the page address in OBS, for example `index.html?seconds=0&volume=0.8`:

| Option | Effect |
|---|---|
| `seconds=0` | Countdown shows `DD:HH:MM` instead of `DD:HH:MM:SS` |
| `volume=0.8` | Music volume, from 0 to 1 (default 0.9) |
| `card=0` | Hide the "now playing" card |
| `demo=fast` | Run the countdown 60× fast, to watch the digits change |
| `date=2026-12-25T00:00:00-05:00` | Count to an exact moment. By default it counts to midnight on December 25 in this PC's time zone. |

On Christmas Day the countdown hits zero and the light strings switch to a chasing pattern. Then it
starts counting toward next Christmas.
