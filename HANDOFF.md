# Handoff: where this project stands

Notes for picking the work up on another machine, or in a new Claude session. Read this first.

## Goal

A 24/7 Christmas livestream on YouTube, hosted from the desktop PC with OBS. The stream shows a
live-rendered cozy room with a fireplace, a countdown to Christmas, and music with a "now playing"
card.

## Status (as of 2026-11-05)

**Done and approved by the user:**
- `christmas/scene/index.html`, the live 2D scene, drawn on a canvas:
  - **Room:** fireplace with particle fire and embers, a Christmas tree drawn from about 4,600 needle
    strokes with ornaments, a bead garland and 150 lights, stockings, a wreath, a poinsettia,
    candles, and a window with snowfall over a moonlit valley.
  - **Countdown:** `DD:HH:MM:SS` across the top of the wall. Each digit is its own string of C9 bulbs
    bent into the numeral's shape and hung from a ceiling nail, in classic red, green, blue, gold and
    orange. When a digit changes, its string blinks off, on, off, then lights in the new shape.
  - **Tree lights:** about 65% blink at random; the rest stay steady.
  - **User feedback:** "Looks great." Earlier they asked to brighten the room, improve the fire and
    tree, and make the countdown look like real string lights. All of that is done. They said the
    window and the outside view look great, so keep those as they are.
- **Music player, inside the same page:** shuffle with no repeats until most songs have played,
  3-second crossfades, a skip on any file that fails to load, and a "now playing" card in the
  bottom right with a spinning record, holly, title and artist, and "…" for long names.
  - **Tested:** the crossfade, the card, and long-name shortening, using dummy tracks.
  - **Preview gotcha:** the Python test server can't seek inside audio files, so seeking fails in the
    preview. Normal playback, and OBS (local files), don't need seeking.
- `christmas/build_playlist.py`:
  - Reads titles and artists from tags, from `Artist - Title` file names, or from
    `music/credits.csv` overrides.
  - Normalizes every track to -16 LUFS with a fixed gain plus a limiter. A very quiet and a loud
    test track both came out at exactly -16.00.
  - Writes `playlist.js` (read by the scene) and `credits.txt` (for the stream description).
- `christmas/README.md`: safe music sources, the playlist build, OBS setup, and scene URL options.

**Next steps:**
1. **Set up the desktop:** Python, a `.venv` with numpy, scipy and pyloudnorm, ffmpeg, and OBS.
   Commands are in the root README.
2. **Music:** the user downloads tracks into `christmas/music/`. Sources and license rules are in
   `christmas/README.md`:
   - YouTube Audio Library
   - Kevin MacLeod / incompetech (CC BY 4.0, must credit)
   - Pixabay (skip "Content ID registered" tracks)
   - Musopen (public-domain recordings only)

   Aim for 4–8 hours. Then run `build_playlist.py`. Music is git-ignored on purpose: licenses forbid
   redistributing it, and the repo may be public.
3. **OBS:** add a Browser source pointing at local file `christmas/scene/index.html`, 1920×1080,
   30 or 60 fps. Tick "Control audio via OBS" and untick "Shutdown source when not visible". Then
   verify in OBS that the music plays and the scene renders smoothly. This hasn't been tested in
   OBS yet.
4. **Open question:** the user never gave a verdict on the two fire sound samples
   (`christmas/fire_sample*.mp3`, made by `fire_ambience.py`). If they like one, mix a long
   fire-crackle bed quietly under the music.
5. **Unverified:** whether YouTube Audio Library licenses cover 24/7 livestreams. YouTube's help
   pages don't say clearly. The user should check before relying on it.

## Things to know about how the user works

- They listen and look carefully, and they give direct feedback. Show small samples or previews and
  get their reaction before producing anything in bulk.
- **Background:** an earlier lo-fi project (procedurally generated music, root-level `lofi_beat.py`,
  `make_album.py` and so on) was **cancelled**. The user listened and said the tracks all sounded the
  same, "copy and pastes of each other", with too much rain sound. Every automated check had passed,
  so metrics don't prove creative quality. Don't revive that approach; the scripts are kept only for
  reference.
- The user prefers Christmas-themed, cozy, readable visuals, and liked the countdown being part of
  the scene rather than a floating overlay.

## Testing the scene

- `python -m http.server 8247 --directory christmas`, then open `http://localhost:8247/scene/index.html`.
  `.claude/launch.json` already defines this server.
- `?demo=fast` runs the countdown 60× fast. `?seconds=0` shows `DD:HH:MM`.
- **Performance:** about 2 ms per frame of drawing. Each countdown digit is cached as an image and
  redrawn only when it changes.
