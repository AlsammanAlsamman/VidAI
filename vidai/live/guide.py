"""The live-processing guide Claude reads (tool: live_guide). Keep it short, concrete and complete."""

GUIDE = {
"overview": """VidAI live processing — two loops while recording:
FAST LOOP (every frame, ms): processors change the video, rules react to live stats, voice commands act,
instant models classify frames. Runs inside the recorder; never waits for you.
SLOW LOOP (you, seconds): read the stats stream with live_stats(session, since=<next>), decide, then send
commands with live_control(session, [...]) or write new processors with live_processor(session, name, code).
Poll every ~5-15 s while the user records; react to `claude` events (the user said "VidAI <request>").
While a request is open, a funny animated VidAI icon shows in the corner of the video ("Claude is thinking").
live_control(..., done=True) (the default) answers the request and hides it; use done=False for
intermediate steps, and {"cmd":"thinking","on":true} to show it yourself during longer work.
Everything live is also saved as anchors (live_action, speech, screen_text, markers...) for editing later.""",

"stats": """Events on the live stream (kind: data):
silence_start {} / silence_end {duration,start}     speech_start {} / speech_end {duration,start}
loud {db}                                           scene_change {score}
transcript {text,start,end,lang,is_command}         voice_command {command,args,text}
claude {message,source}  <- the user asked YOU something by voice ("VidAI make the title bigger")
screen_text {text,lines,new_lines}  (OCR, if on)    marker {type,note,source}
learner {name,label,confidence}                     window_focus {title}
action {...} what the recorder did                  ack/error {command,...} results of your commands
perf {fps,frame_ms,active_processors,level,encoder}   warning {what:"performance", level, text}
notify {text,kind}   help {lines}   question {question,text,options}   answer {question,answer,said,by}
Voice commands built in: "VidAI" + mark | mistake | new section <title> | important | zoom in | zoom out |
captions on/off | learn <name> | label <value> | wrong | stop | anything else -> claude event. Arabic works too
(فيداي علامة / خطأ / قسم جديد ... / مهم / تكبير / تصغير / إيقاف).""",

"commands": """live_control commands (JSON objects, "for": seconds makes anything temporary):
{"cmd":"text","text":"...","position":"top-left|bottom-center|[x,y]","size":0.05,"color":"#fff","box":"#000A","for":4}
{"cmd":"shape","shape":"arrow|circle|box","x":.5,"y":.5,"w":.15,"h":.15,"angle":225,"color":"#FF3B30","for":3}
{"cmd":"image","path":"/abs/logo.png","position":"top-right","width":.12}
{"cmd":"zoom","x":.25,"y":.25,"w":.5,"h":.5,"for":6}          {"cmd":"blur","x":0,"y":.9,"w":.4,"h":.1}
{"cmd":"add","name":"captions","type":"captions"}  types: text shape image zoom blur captions model attach big_eyes
{"cmd":"add","name":"fx_crown","type":"attach","params":{"what":"👑","to":"head","scale":1.0}}
   attach to: hand right_hand left_hand other_hand finger head above_head face eyes nose mouth screen
   what: any emoji, a word ("apple","horns","sunglasses"...), or an image path
{"cmd":"add","name":"fx_big_eyes","type":"big_eyes","params":{"zoom":1.8}}
{"cmd":"add","name":"fx_adjust","type":"adjust","params":{"auto":true,"brightness":0.1,"contrast":1.1,"saturation":1.2,
   "warmth":0.3,"gamma":0.9,"sharpen":0.3}}   picture: light/contrast/colours/warmth/sharpness (one C pass)
{"cmd":"add","name":"fx_background","type":"background","params":{"mode":"blur|color|image|animated","image":"/path.jpg"}}
   (the user can say these themselves: "fix the light", "brighter", "blur the background", "beach behind me"...)
{"cmd":"add","name":"x","file":"/abs/path.py","params":{...}}  (your own processor file)
{"cmd":"set","name":"x","params":{...}}  {"cmd":"enable","name":"x","for":5}  {"cmd":"disable","name":"x"}
{"cmd":"remove","name":"x"}
{"cmd":"rule","rule":{...}}  {"cmd":"unrule","id":"..."}
{"cmd":"mark","type":"section|important|mistake|marker","note":"..."}
{"cmd":"learn","name":"slide","labels":["yes","no"],"region":[x,y,w,h]?}  {"cmd":"label","name":"slide","value":"yes"}
{"cmd":"wrong","name":"slide"}  {"cmd":"forget","name":"slide"}
{"cmd":"undo"}  {"cmd":"redo"}   (one request = one step)   {"cmd":"help"}   {"cmd":"lighter"}
{"cmd":"notify","text":"...","seconds":5}   message in the window only (never recorded) — or live_notify(...)
{"cmd":"listen","seconds":6}   push-to-talk: next utterance is a command without "VidAI"
Questions: live_ask_user(session, "Which background?", ["blur","purple","beach"]) -> {"answer": "purple"}
Talk: "VidAI talk" -> VidAI says "How can I help you?" -> the next sentence arrives with reply="voice" ->
      answer with live_say(session, text, subtitle=False) = {"cmd":"say","text":...}: offline voice (Piper)
      on the speakers and mixed into the video (mic ducked under it).
{"cmd":"stt","on":true,"language":"ar"}  {"cmd":"ocr","on":true,"interval":2}  {"cmd":"status"}  {"cmd":"stop"}""",

"rules": """Rules = instant reflexes (the recorder applies them in ms, no need for you to watch):
{"id":"pause_title","when":{"kind":"silence_end","where":{"duration":{">":2.5}}},
 "do":[{"show_text":"{section}","for":3,"position":"top-left"}],"cooldown":20}
where-ops: > >= < <= == != contains in startswith matches(regex). once:true fires one time.
actions: show_text / zoom{} / shape{} / enable / disable / set+params / mark / notify_claude / label+value
templates: {text} {command} {args} {label} {section} {duration} {t} + any data field of the event.
Examples:
- when screen_text contains "error" -> shape box around the screen + notify_claude "error on screen"
- when learner slide == yes -> enable "slide_zoom"; when == no -> disable it
- when loud -> mark important""",

"processors": """Write a processor when commands/rules are not enough (live_processor(session, name, code)):
from vidai.live.processors import LiveProcessor, register
import numpy as np, cv2
from vidai import native
@register
class Vignette(LiveProcessor):
    defaults = {"strength": 0.5}
    stage = 0                      # 0 = changes the picture (runs first), 1 = overlay on top (default)
    listens = {"speech_start"}     # events delivered to on_event (optional)
    def on_event(self, ev, ctx): ...
    def process(self, frame, t, ctx):   # frame: HxWx3 uint8 RGB (1920x1080 by default); return a frame
        ...
Hands and face: set `tracking = True` on the class, then read ctx.tracks.hand("Right"|"Left"|"any") -> palm,
size, tip, points and ctx.tracks.face_now() -> box, eyes, nose, mouth, top (all normalized 0..1). VidAI runs
and downloads the trackers itself. Save your effect with live_processor(..., save_as="name") for reuse.
Rules: budget ~8 ms per frame (set budget_ms if you need more); no Python loops over pixels — use NumPy,
OpenCV (cv2) and vidai.native (alpha_blend, affine_color, frame_mad, rms_db). Cache anything expensive
(ctx.overlay(op) caches rendered text/shape/image overlays). ctx.stats = latest value of every stat,
ctx.bus.publish(kind, data) to emit your own stats (other rules can react to them).
A processor that raises or is too slow is disabled automatically and you get an `error` event with the reason:
read it, fix the code, call live_processor again (same name replaces it).""",

"permissions": """Installing, downloading, creating files: VidAI does it itself (no Claude Code prompts).
vidai_install(["rembg"], session, reason="remove the background") / vidai_download(url, name, session, reason) /
vidai_create_file(relpath, content, session, reason).
- ask mode (default): VidAI says "Master, I need to install rembg to remove the background. Say VidAI confirm,
  or VidAI deny." and shows Confirm / Deny in the window; the call returns done / denied / no_answer.
- full mode: the user said "VidAI, take all actions" -> runs at once ("VidAI, ask me first" switches back).
  Only the user can give it (voice with the wake word, typing, or `vidai permissions full`); it expires.
  Claude's commands can never confirm/deny or switch full access on.
- no recorder open: needs_confirmation -> vidai_confirmed_action (Claude Code asks the user).
- effect code: the first live_processor of a session asks "run effect code written by Claude?" once.
Downloads: https from public hosts only (pass sha256 when known). Installs go only into VidAI's own Python;
files only under ~/.vidai or a real session folder; all logged in ~/.vidai/actions.log.""",

"hub": """Small open models with ready adapters (vidai.hub) — use them before writing code:
model_search("show my mood") -> catalog ids; model_apply(session, "emotion"|"gestures"|"anime"|"style_mosaic"|
"style_candy"|"style_udnie"|"style_rain_princess"|"style_pointilism", params) = {"cmd":"model","model":...}
(asks permission to download unless full access). Live events: emotion {label,confidence}, gesture {name,emoji}
-> rules: {"when":{"kind":"gesture","where":{"name":"Victory"}},"do":[{"sticker":"✌","for":2}]}
Colours like a photo: color_from_photo(session, "/path.jpg") (type "grade", trained in ms).
Styles/anime are slow live (a few paintings per second); they look best applied when editing.""",

"models": """Instant models (learned and corrected during the recording, k-NN on tiny frame features):
1. {"cmd":"learn","name":"whiteboard","labels":["yes","no"],"region":null}
2. Teach: {"cmd":"label","name":"whiteboard","value":"yes"} while the thing is visible, "no" when not
   (the user can say "VidAI label yes/no"). Predictions start after both labels have examples.
3. It publishes learner events (name,label,confidence) when its answer changes -> use them in rules.
4. Correct instantly: {"cmd":"wrong","name":"whiteboard"} (or the user says "VidAI wrong"), or label again.
5. At the end it is saved to the lab registry (reusable with classify_video on other videos).
For heavier needs (a real detector, a transform) use the lab: train/train_until_suitable offline, then run it
live with {"cmd":"add","type":"model","params":{"model":"<name>"}} or your own processor that loads it.""",
"feed": """Feed models: small models trained on THIS video, locally, while the user records (docs/feed-models.md).
They make in-scene effects look filmed: `lighting` (room light, tint, camera softness, grain) and `hair`
(a 1 ms student learned from MediaPipe's hair segmenter on the user's own hair).
- In a processor: feeds = frozenset({"lighting", "hair"}) (started automatically, shared) and draw with
  ctx.composite(frame, rgba, x, y, occlude={"hair"}) instead of native.alpha_blend. Before they are ready it
  is a plain blend; screen graphics (corner titles, captions) keep native.alpha_blend.
- attach stickers use it already: params realistic (default true), behind_hair (things worn on the head).
  "VidAI, make it realistic" puts head effects behind the hair.
- Progress: `feed_model` events {model, ready, score, samples} in live_stats; students are saved to the lab
  as feed_<key> when the recording stops (next session starts trained).
- They cannot generate new realistic pixels (real ears, different glasses): be honest about that, and offer
  a real photo (PNG) through attach instead."""
}


def guide(topic: str | None = None) -> str:
    if topic and topic in GUIDE:
        return GUIDE[topic]
    return "\n\n".join(f"## {k}\n{v}" for k, v in GUIDE.items())
