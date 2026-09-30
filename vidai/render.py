"""Render an edit plan with ffmpeg. Long videos are split at anchor-safe points and rendered in parallel."""
from __future__ import annotations

import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ffmpeg
from .anchors import AnchorFile
from .edit import ApplyModel, Audio, Chapter, Cut, EditPlan, Image, Shape, Subtitle, Text, Zoom
from .overlays import render_overlay

AUDIO_SR = 48000


@dataclass
class RenderResult:
    output: str
    duration: float
    chunks: list[tuple[float, float]]
    seconds: float
    srt: str | None = None
    chapters: list[tuple[float, str]] = field(default_factory=list)


def _between(ranges: list[tuple[float, float]], fps: float) -> str:
    """Frames of the half-open ranges (on the frame grid): exactly round((b - a) * fps) frames each."""
    h = 0.5 / fps
    return "+".join(f"gte(t,{a - h:.6f})*lt(t,{b - h:.6f})" for a, b in ranges)


def _snap(t: float, fps: float) -> float:
    return round(t * fps) / fps


def snap_ranges(ranges: list[tuple[float, float]], fps: float) -> list[tuple[float, float]]:
    """Keep ranges snapped to the video frame grid, so video and audio are cut at the same instants."""
    out = [(_snap(a, fps), _snap(b, fps)) for a, b in ranges]
    return [(a, b) for a, b in out if b - a > 0.5 / fps]


def _meta_escape(s: str) -> str:
    for ch in "\\=;#\n":
        s = s.replace(ch, "\\" + ch)
    return s


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(items: list[tuple[float, float, str]], path: str | Path) -> Path:
    path = Path(path)
    lines = []
    for i, (a, b, text) in enumerate(sorted(items), 1):
        lines += [str(i), f"{_srt_time(a)} --> {_srt_time(b)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _clip(a: float, b: float, cs: float, ce: float) -> tuple[float, float] | None:
    a, b = max(a, cs), min(b, ce)
    return (a - cs, b - cs) if b > a else None


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait()


def _apply_models(src: str, cs: float, ce: float, ops: list[ApplyModel], fps: float, W: int, H: int,
                  out: Path, has_audio: bool) -> Path:
    """Decode the chunk, run lab frame transforms in their ranges, re-encode (near lossless)."""
    from .lab import load_model

    models = [(op, load_model(op.name)[0]) for op in ops]
    dur = ce - cs
    exe = ffmpeg.ffmpeg_exe()
    enc_cmd = [exe, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", f"{fps}", "-i", "-"]
    if has_audio:
        enc_cmd += ["-ss", f"{cs:.3f}", "-t", f"{dur:.3f}", "-i", src, "-map", "0:v", "-map", "1:a",
                    "-c:a", "pcm_s16le"]
    enc_cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", str(out)]
    size = W * H * 3
    i = 0
    with tempfile.TemporaryFile() as dec_err, tempfile.TemporaryFile() as enc_err:
        dec = subprocess.Popen([exe, "-loglevel", "error", "-ss", f"{cs:.3f}", "-i", src, "-t", f"{dur:.3f}",
                                "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               stdout=subprocess.PIPE, stderr=dec_err)
        enc = None
        try:
            enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=enc_err)
            while True:
                buf = dec.stdout.read(size)  # type: ignore[union-attr]
                if len(buf) < size:
                    break
                frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
                t = cs + i / fps
                for op, m in models:
                    if op.start <= t and (op.end is None or t < op.end):
                        frame = m.transform_frame(frame, t=t, **op.params)
                enc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())  # type: ignore[union-attr]
                i += 1
            enc.stdin.close()  # type: ignore[union-attr]
            dec_rc, enc_rc = dec.wait(), enc.wait()
        finally:
            for p in (dec, enc):
                if p is not None:
                    _stop(p)

        def tail(f) -> str:
            f.seek(0)
            return f.read().decode(errors="replace")[-1000:]

        if dec_rc != 0 or i == 0:
            raise ffmpeg.FFmpegError(f"model pre-pass decode failed ({dec_rc}, {i} frames): {tail(dec_err)}")
        if enc_rc != 0:
            raise ffmpeg.FFmpegError(f"model pre-pass encode failed ({enc_rc}): {tail(enc_err)}")
    return out


def _zoom_crop(z: Zoom, W: int, H: int) -> tuple[int, int, int, int]:
    """Crop box (w, h, x, y) covering the zoom region with the output aspect ratio (no stretching),
    inside the frame, even sizes, at least 2 px."""
    cw, ch = max(z.w * W, 2.0), max(z.h * H, 2.0)
    if cw / ch < W / H:
        cw = ch * W / H
    else:
        ch = cw * H / W
    if cw > W or ch > H:
        cw, ch = float(W), float(H)
    cw_, ch_ = max(2, int(round(cw / 2)) * 2), max(2, int(round(ch / 2)) * 2)
    cw_, ch_ = min(cw_, W // 2 * 2), min(ch_, H // 2 * 2)
    mx, my = (z.x + z.w / 2) * W, (z.y + z.h / 2) * H
    cx = int(round(min(max(mx - cw_ / 2, 0), W - cw_)))
    cy = int(round(min(max(my - ch_ / 2, 0), H - ch_)))
    return cw_, ch_, cx, cy


def _render_chunk(plan: EditPlan, info: ffmpeg.MediaInfo, cs: float, ce: float, overlays: list[tuple[Path, float, float]],
                  tmp: Path, idx: int, crf: int, preset: str) -> Path | None:
    W, H, fps = info.width, info.height, info.fps or 30.0
    keeps = [r for a, b in snap_ranges(plan.keep_ranges(), fps) if (r := _clip(a, b, cs, ce))]
    if not keeps:
        return None
    dur = ce - cs
    src, seek = plan.source, cs

    model_ops = [o for o in plan.of(ApplyModel) if o.start < ce and (o.end is None or o.end > cs)]
    if model_ops:
        src = str(_apply_models(plan.source, cs, ce, model_ops, fps, W, H, tmp / f"model_{idx:03d}.mkv", info.has_audio))
        seek = 0.0

    args = []
    if seek:
        args += ["-ss", f"{seek:.3f}"]
    args += ["-t", f"{dur:.3f}", "-i", src]
    graph: list[str] = []
    v = "0:v"
    n = 0

    def lab() -> str:
        nonlocal n
        n += 1
        return f"v{n}"

    for z in plan.of(Zoom):
        r = _clip(z.start, z.end, cs, ce)
        if not r:
            continue
        a_, b_, z_ = lab(), lab(), lab()
        cw, ch, cx, cy = _zoom_crop(z, W, H)
        o = lab()
        graph.append(f"[{v}]split[{a_}][{b_}]")
        graph.append(f"[{b_}]crop={cw}:{ch}:{cx}:{cy},scale={W}:{H},setsar=1[{z_}]")
        graph.append(f"[{a_}][{z_}]overlay=0:0:enable='between(t,{r[0]:.3f},{r[1]:.3f})'[{o}]")
        v = o

    inp = 1
    for png, a, b in overlays:
        r = _clip(a, b, cs, ce)
        if not r:
            continue
        args += ["-i", str(png)]
        o = lab()
        graph.append(f"[{v}][{inp}:v]overlay=0:0:enable='between(t,{r[0]:.3f},{r[1]:.3f})'[{o}]")
        v = o
        inp += 1

    if plan.burn_subtitles:
        subs = [(r[0], r[1], s.text) for s in plan.of(Subtitle) if (r := _clip(s.start, s.end, cs, ce))]
        if subs:
            srt = write_srt(subs, tmp / f"subs_{idx:03d}.srt")
            o = lab()
            graph.append(f"[{v}]subtitles=filename='{srt}':force_style='FontSize=22,Outline=2'[{o}]")
            v = o

    whole = len(keeps) == 1 and keeps[0][0] <= 1e-3 and keeps[0][1] >= dur - 1e-3
    o = lab()
    sel = "" if whole else f"select='{_between(keeps, fps)}',"
    graph.append(f"[{v}]{sel}setpts=N/FRAME_RATE/TB,format=yuv420p[{o}]")
    v = o

    maps = ["-map", f"[{v}]"]
    if info.has_audio:
        af = [f"aresample={AUDIO_SR}", "aformat=channel_layouts=stereo"]
        for au in plan.of(Audio):
            if au.filter == "denoise":
                af.append(f"afftdn=nr={au.value or 12}")
            elif au.filter == "highpass":
                af.append(f"highpass=f={au.value or 80}")
            elif au.filter == "volume":
                af.append(f"volume={au.value or 1.0}")
        trims = [f"atrim=start_sample={round(a * AUDIO_SR)}:end_sample={round(b * AUDIO_SR)},asetpts=PTS-STARTPTS"
                 for a, b in keeps]  # sample-exact cuts at the same (frame-grid) instants as the video
        if whole:
            graph.append(f"[0:a]{','.join(af)}[aout]")
        elif len(trims) == 1:
            graph.append(f"[0:a]{','.join(af + trims)}[aout]")
        else:
            k = len(trims)
            graph.append(f"[0:a]{','.join(af)},asplit={k}" + "".join(f"[as{j}]" for j in range(k)))
            graph += [f"[as{j}]{tr}[at{j}]" for j, tr in enumerate(trims)]
            graph.append("".join(f"[at{j}]" for j in range(k)) + f"concat=n={k}:v=0:a=1[aout]")
        maps += ["-map", "[aout]"]

    out = tmp / f"chunk_{idx:03d}.mkv"
    ffmpeg.run(args + ["-filter_complex", ";".join(graph)] + maps +
               ["-r", f"{fps}", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-c:a", "pcm_s16le", str(out)])
    return out


def _chapters_meta(chapters: list[tuple[float, str]], total: float, path: Path) -> Path:
    lines = [";FFMETADATA1"]
    for i, (t, title) in enumerate(chapters):
        end = chapters[i + 1][0] if i + 1 < len(chapters) else total
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(t * 1000)}", f"END={int(end * 1000)}",
                  f"title={_meta_escape(title)}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _output_chapters(plan: EditPlan, total: float) -> list[tuple[float, str]]:
    """Chapters in output time: one per distinct time, inside the video, the first at 0:00 (YouTube rule)."""
    out: list[tuple[float, str]] = []
    for c in sorted(plan.of(Chapter), key=lambda c: c.t):
        t = round(plan.map_time_after(c.t), 3)
        if t < total - 1e-3 and not (out and t - out[-1][0] < 1e-3):
            out.append((t, c.title))
    if out and out[0][0] > 0:
        out[0] = (0.0, out[0][1])
    return out


def srt_items(plan: EditPlan) -> list[tuple[float, float, str]]:
    """Subtitles in output time. A subtitle partly inside a cut keeps its surviving part (as burned-in ones do)."""
    return [(a, b, s.text) for s in plan.of(Subtitle)
            if (b := plan.map_time_after(s.end)) - (a := plan.map_time_after(s.start)) > 1e-3]


def render(plan: EditPlan, output: str | Path, workers: int = 3, anchors: AnchorFile | None = None,
           crf: int = 20, preset: str = "veryfast", scale_height: int | None = None) -> RenderResult:
    """Render `plan` to `output` (mp4, YouTube-friendly: H.264 + AAC, faststart, chapters).

    The source is split into `workers` chunks at anchor-safe points (silences), each chunk is rendered
    in its own ffmpeg process, then the chunks are joined.
    """
    t0 = time.time()
    output = Path(output)
    info = ffmpeg.probe(plan.source)
    if not plan.duration:
        plan.duration = info.duration
    if anchors is None:
        p = AnchorFile.path_for(plan.source)
        anchors = AnchorFile.load(p) if p.exists() else AnchorFile(video=plan.source, duration=info.duration)
    anchors.duration = anchors.duration or info.duration
    chunks = anchors.chunks(workers) if workers > 1 else [(0.0, info.duration)]
    fps = info.fps or 30.0  # cuts and chunk borders on the frame grid: A/V cut at the same instants, no drift
    grid = [o.model_copy(update={"start": _snap(o.start, fps), "end": _snap(o.end, fps)}) if isinstance(o, Cut) else o
            for o in plan.ops]
    plan = plan.model_copy(update={"ops": grid})
    chunks = [(_snap(a, fps) if i else 0.0, _snap(b, fps) if i + 1 < len(chunks) else b)
              for i, (a, b) in enumerate(chunks)]

    with tempfile.TemporaryDirectory(prefix="vidai_") as td:
        tmp = Path(td)
        overlays = []
        for i, op in enumerate(o for o in plan.ops if isinstance(o, (Text, Image, Shape))):
            png = render_overlay(op, info.width, info.height, tmp / f"ov_{i:03d}.png")
            overlays.append((png, op.start, op.end))

        with ThreadPoolExecutor(max(1, workers)) as ex:
            futs = [ex.submit(_render_chunk, plan, info, cs, ce, overlays, tmp, i, crf, preset)
                    for i, (cs, ce) in enumerate(chunks)]
            parts = [p for f in futs if (p := f.result())]
        if not parts:
            raise ValueError("the plan cuts the whole video")

        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        total = plan.output_duration()
        chapters = _output_chapters(plan, total)
        args = ["-f", "concat", "-safe", "0", "-i", str(lst)]
        if chapters:
            args += ["-i", str(_chapters_meta(chapters, total, tmp / "meta.txt")), "-map_metadata", "1",
                     "-map_chapters", "1"]
        args += ["-map", "0:v"] + (["-map", "0:a"] if info.has_audio else [])
        if scale_height:
            args += ["-vf", f"scale=-2:{scale_height}", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                     "-pix_fmt", "yuv420p"]
        else:
            args += ["-c:v", "copy"]
        if info.has_audio:
            af = ["loudnorm=I=-14:TP=-1.5:LRA=11"] if any(a.filter == "loudnorm" for a in plan.of(Audio)) else []
            args += (["-af", ",".join(af)] if af else []) + ["-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_SR)]
        ffmpeg.run(args + ["-movflags", "+faststart", str(output)])

    srt = None
    subs = srt_items(plan)
    if subs:
        srt = str(write_srt(subs, output.with_suffix(".srt")))
    return RenderResult(str(output), ffmpeg.probe(str(output)).duration, chunks, round(time.time() - t0, 2),
                        srt, chapters)
