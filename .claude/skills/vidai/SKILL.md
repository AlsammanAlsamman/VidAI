---
name: vidai
description: VidAI video plugin. Use when the user says "vidai" (or asks to record a video, make a YouTube video, or edit/improve a recording). Claude interviews the user, configures the anchors, opens the VidAI Recorder (camera/screen/mic), then edits and enhances the video with the VidAI tools (cuts, gaps, text, icons, arrows, zooms, subtitles, chapters, audio, custom lab models) and exports for YouTube.
---

# VidAI — record, understand, improve

VidAI tools come from the `vidai` MCP server (fallback: the `vidai` CLI, which prints JSON).
Times are always **source-video seconds**. The source recording is never modified.

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
- **While recording (slow loop):** poll `live_stats(session, since=next)` every ~10 s:
  - handle `claude_requests` (the user asked you something by voice) with `live_control` or `live_processor`
  - react to stats (long silence, screen_text, learner changes, errors) and adjust processors/rules
  - write a new processor with `live_processor(session, name, code)` when commands/rules are not enough;
    if it returns `runtime_error`, fix the code and call it again
  - keep it light: effects should help the video, not distract
- Stop polling when `state` is no longer "recording" (or use `studio_wait`).

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
