---
name: vidai
description: VidAI video plugin. Use when the user says "vidai" (or asks to record a video, make a YouTube video, or edit/improve a recording). Claude interviews the user, configures the anchors, opens the VidAI Recorder (camera/screen/mic), then edits and enhances the video with the VidAI tools (cuts, gaps, text, icons, arrows, zooms, subtitles, chapters, audio, custom lab models) and exports for YouTube.
---

# VidAI — record, understand, improve

VidAI tools come from the `vidai` MCP server (fallback: the `vidai` CLI, which prints JSON).
Times are always **source-video seconds**. The source recording is never modified.

## Phase 0 — Remember the user (always first)

Call `vidai_profile()`: lessons from past mistakes, word corrections, instant macros, preferences and recent
sessions. Follow the lessons; don't ask what the profile already answers (brief defaults are filled in).

## Phase 1 — Brief (when the user says "vidai")

Interview the user before anything else. Use what you already know about them (memory, earlier
conversation, their channel/topic) to pre-fill answers and only ask what is missing. Keep it short:
one AskUserQuestion round for the choices, plus one free-text message for the details.

Ask (see `brief_questions`):
- **Choices** (AskUserQuestion): style/mode (camera · screen · screen+camera), language (ar · en · ar+en),
  extras (subtitles, chapters, intro/outro, logo, music, zooms/arrows), expected length.
- **Free text**: title, what the video is about, audience/level, planned sections (outline), anything to hide.

## Phase 2 — Anchor configuration (your decision)

From the brief, decide which stats to record and why (small and relevant beats complete):
- `suggest_stats(brief)` gives a starting point and the full list.
- Screen videos: silence, audio_level, scene_change, input_activity, window_focus.
- Camera videos: silence, audio_level, motion.
- Always: markers.
- Tune: `silence_db` (None = adaptive), `min_silence` (shorter for fast speakers).
Show the user a 3–5 line summary of the config (what you will track and why).

## Phase 3 — Record (with live processing)

- Optional: `studio_devices()` to check the camera, screen and mic.
- Read `live_guide()` once (stats, commands, rules, processors, instant models).
- `studio_start(brief, stats, capture, silence_db, min_silence, live)` creates the session folder and **opens the
  VidAI Recorder window**. capture = {"mode", "camera", "camera_size", "pip_position", "pip_width", "mic"}.
  live = {"stt": true, "stt_language": "ar"|"en"|null, "ocr": true for screen videos, "processors": [...],
  "rules": [...], "learners": [...]}. Plan useful live behavior from the brief, for example:
  a logo (`image`), `captions` if subtitles were requested, a rule that shows the section title after a
  section marker, a rule that marks "important" on `loud`, a `blur` over a region the user wants hidden,
  an instant model when the user wants something recognized (a slide, a gesture, the whiteboard).
- Tell the user (from `tell_user`): press Record (or space); hotkeys ctrl+alt+m marker, ctrl+alt+x mistake
  (then repeat the sentence), ctrl+alt+n new section, ctrl+alt+i important, ctrl+alt+s stop.
- Tell them about voice: "VidAI mark / mistake / new section <title> / important / zoom in / zoom out /
  captions on / stop", and "VidAI <anything else>" reaches you (Arabic works too: فيداي ...).
- **While recording: answer fast, only with VidAI tools (no Bash, no downloads, no file writes).**
  Loop: `live_wait_request(session, since=next)` → it returns the request text → answer with ONE call → wait again.
  - VidAI already handles common requests itself (`handled_by_vidai`): stickers on hands/head/eyes/face,
    pop-out eyes, remove, bigger/smaller, swap hands. Don't redo those.
  - Before writing any code for a visual idea, check `model_search(request)`: small open models with ready
    adapters (emotion, hand gestures, anime look, painting styles…) → `model_apply(session, id)` (VidAI asks
    permission to download, or installs directly in full access). Colours "like this photo" →
    `color_from_photo(session, path)`. Nothing fits? `model_search(..., online=True)` lists Hugging Face ONNX
    models; wrap one in a small `live_processor` adapter (run it in a worker thread) and `save_as` it.
    Gestures/emotion publish live events: add rules like
    {"when":{"kind":"gesture","where":{"name":"Thumb_Up"}},"do":[{"sticker":"👍","for":2}]}.
  - Prefer, in this order: `live_control` with built-ins (`attach` any emoji/word to hand | right_hand |
    left_hand | finger | head | above_head | face | eyes | nose | mouth | screen; `big_eyes`, `text`, `zoom`,
    `blur`, `shape`, `captions`) → `live_effect(session, name)` from the saved library (`live_effects()`) →
    only if nothing fits, `live_processor(session, name, code, save_as=...)` (hands/face are in `ctx.tracks`,
    VidAI downloads models itself). Save new effects with `save_as` so next time is instant.
    Things drawn *on the person or in the scene* should use feed models (`live_guide("feed")`):
    `feeds = frozenset({"lighting", "hair"})` + `ctx.composite(...)` so they take the room's light and sit
    behind the hair. Appearance requests ("change my glasses"): say honestly what an overlay can do and ask
    before applying — never silently use a cartoon sticker.
  - No checks before answering; keep chat messages to one line while the user records.
  - Requests with `reply: "voice"` come from "VidAI talk" (VidAI said "How can I help you?" and the user asked a
    question): answer OUT LOUD with `live_say(session, "short answer, 1-3 sentences")` — it is spoken with
    VidAI's voice and mixed cleanly into the video (`subtitle=True` also shows it as text in the video).
  - Not sure what they mean (which background? how big? which hand)? Don't guess:
    `live_ask_user(session, "Which background?", ["blur", "purple", "beach"])` shows buttons in the window and
    asks by voice; it returns the answer ("VidAI, the second one", a click, or typed text all work).
  - After you do something, tell them in one line in the window (never recorded):
    `live_notify(session, "Hair made bigger — say 'VidAI undo' to take it back")`.
  - "VidAI suggest" previews ideas one by one (confirm / next / cancel) — VidAI does it alone; to add a new idea
    to the list, extend LivePipeline.SUGGESTIONS with a phrase the fast path understands.
  - The user can say "VidAI undo / redo" (a whole request is one step), "VidAI help" (command card),
    "VidAI lighter" (performance), type in the "Tell VidAI…" box, or hold ctrl+alt+space to talk without the
    wake word. Requests from typing arrive exactly like voice ones.
  - If the recorder publishes performance `warning` events, prefer lighter effects; VidAI lowers tracking and
    preview rates by itself and only turns an effect off as a last resort (and says so).
  - Need to install, download or create something? Never use Bash/Write/curl/pip yourself — ask VidAI:
    `vidai_install(packages, session, reason)`, `vidai_download(url, name, session, reason)`,
    `vidai_create_file(relpath, content, session, reason)`. VidAI says "Master, I need to ..." and waits for
    "VidAI confirm / deny" (or the window buttons); if the user said "VidAI, take all actions" it just runs.
    Outside a recording it returns `needs_confirmation`: tell the user what and why, then call
    `vidai_confirmed_action(action, args, session, reason)` — Claude Code shows them a permission prompt, and
    that prompt is their answer. Never try to switch on full access yourself: only the user can
    ("VidAI, take all actions", or `vidai permissions full` in a terminal); it expires after a few hours.
  - The first `live_processor` of a session makes VidAI ask the user to allow effect code written by Claude;
    if they deny, use built-ins / library effects instead. Processor, effect and model names are plain
    identifiers (letters, digits, `_ - .`).
  - Tool errors come back with the real message (e.g. a bad op field): read it and fix the call.
- Stop polling when `state` is no longer "recording" (or use `studio_wait`).

## Phase 3b — Learn (after every recording)

VidAI learns by itself (misheard phrases corrected by the user, effects removed right away, Claude's answers
kept → instant macros, size/hand preferences). Also review the session yourself and teach it what it cannot
see: `vidai_learn("lesson", {"lesson": "..."})` for mistakes and what to do instead, `("correction", ...)`
for misheard words, `("words", {"words": [...]})` for topic names, `("macro", ...)` for a request that should
be instant next time. `vidai_forget(...)` removes anything wrong. Tell the user in one line what was learned.

## Phase 4 — Understand (anchors first, never scan the whole video)

- `anchors(video, "summary")`, then drill down (`segments`, `events`, `at`, `series`).
- Live anchors: `speech` segments carry the transcript (use them for subtitles and chapter titles),
  `screen_text` = what was on screen, `voice_command`, `claude_request`, `live_action` = what was already
  burned into the video live (don't add it twice), `learner:<name>` segments from instant models.
- Look only where anchors point: `contact_sheet(video, [times at markers / scene changes / sections])`,
  then Read the PNG. Compare with the brief's outline and flag missing or extra sections.
- Save your own findings: `add_anchor_events(video, [{"t": .., "kind": "topic", "data": {...}}])`.

## Phase 5 — Improve (take initiative)

Propose a concrete edit list, apply it, and explain it briefly. Good defaults:
1. `plan(video, "remove_gaps")` and `plan(video, "cut_mistakes")`
2. `plan(video, "chapters", titles=<outline>)` (section markers first, then scene changes)
3. A title card at the start, and lower-thirds for each section (`text`, Arabic supported)
4. Arrows/circles (`shape`) or `zoom` at "important" markers and input-activity peaks
5. Audio: `loudnorm` (always), `denoise` if the noise floor is high (check the audio_level series)
6. Logo/icon (`image`) if the user has one; subtitles (`subtitle` ops, `burn_subtitles` if asked)

Op reference:
- `{"op":"cut","start","end"}`
- `{"op":"text","start","end","text","position":"bottom-center"|[x,y],"size":0.05,"color","box"}`
- `{"op":"image","start","end","path","position","width"}`
- `{"op":"shape","start","end","shape":"arrow|circle|box","x","y","w","h","angle","color"}`
- `{"op":"zoom","start","end","x","y","w","h"}`
- `{"op":"audio","filter":"loudnorm|denoise|highpass|volume","value"}`
- `{"op":"subtitle","start","end","text"}`
- `{"op":"chapter","t","title"}`
- `{"op":"model","name","start","end","params"}`

`plan(video, "show")` to review; `plan(video, "remove", index=i)` to undo; `note=` to log your reasoning.

## Phase 6 — Render and review

- `render_video(video, workers=3)` gives a YouTube MP4 + `.srt` + `.youtube.txt` (title, description, chapters).
  It renders in parallel chunks split inside silences.
- Check the output with `contact_sheet(output, [...])`, fix, re-render. Report the output path and the length.

## Lab — only when regular ops are not good enough

Explain why the regular ops fail, write a `vidai.lab.LabModel` subclass (patterns in `vidai/lab/examples.py`;
put hot per-pixel loops in C like `vidai/native/fastops.c`), build `.npz` data from the user's footage, and
`train(..., max_rounds=1)` round by round, adjusting until `suitable` is true, then `save_as`. Use it with
`{"op":"model"}` (frame transforms) or `classify_video` (new anchors).
