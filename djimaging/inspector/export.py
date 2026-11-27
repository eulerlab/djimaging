"""MP4 export of the recording inspector.

Every movie frame in the selected time range becomes one video frame, rendered by the same ``InspectorFigure`` as the
notebook view (field view with outlines and highlights, progressing traces, heatmap, trigger markers and elapsed
time). The time range is mapped to frames with ``timing.frame_range``, the same mapping the viewer uses.

The export speed is independent of the preview speed: ``fps = speed / frame_dt``, i.e. ``speed=1`` replays the
recording in real time (about 15 fps for a 64 x 64 scan at 66.56 ms per frame), ``speed=0.5`` at half speed.

Encoder: the ffmpeg binary shipped with ``imageio-ffmpeg`` (``pip install imageio-ffmpeg``, a few tens of MB, no
system install needed), otherwise ``ffmpeg`` on the ``PATH``. Invocation (via ``imageio_ffmpeg.write_frames``):
``ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r <fps> -i - -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4``.
A JSON sidecar (``out.mp4.json``) records presentation, channel, signal, frame range, times, fps, selection and
filter, so an export can be checked against the viewer.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from djimaging.inspector.render import InspectorFigure, ViewState
from djimaging.inspector.timing import frame_range

DEFAULT_EXPORT_DIR = Path.home() / 'djimaging_inspector_exports'
EXPORT_SIZE_PX = (1280, 736)  # divisible by 16 (H.264 macro blocks)


def find_ffmpeg() -> tuple[Optional[str], str]:
    """Return (path to ffmpeg, how it was found) or (None, explanation)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe(), 'imageio-ffmpeg'
    except Exception as e:  # noqa: BLE001 (ImportError or missing binary)
        reason = f'imageio-ffmpeg unavailable ({type(e).__name__})'
    exe = shutil.which('ffmpeg')
    if exe:
        return exe, 'PATH'
    return None, f"{reason} and no ffmpeg on PATH: install it with `uv pip install imageio-ffmpeg`"


def check_output_path(path: str | Path, forbidden_roots: Iterable[str | Path] = ()) -> Path:
    """Exports must not be written into the data folders (they are treated as read-only)."""
    path = Path(path).resolve()
    for root in forbidden_roots:
        root = Path(root).resolve()
        if path == root or root in path.parents:
            raise ValueError(f"refusing to write {path}: it is inside the data folder {root}, which is read-only")
    if path.suffix.lower() != '.mp4':
        raise ValueError(f"export path must end with .mp4, got {path.name}")
    return path


def export_mp4(state: ViewState, path: str | Path, t_start: Optional[float] = None, t_stop: Optional[float] = None,
               speed: float = 1.0, forbidden_roots: Iterable[str | Path] = (), progress=None, crf: int = 18) -> dict:
    """Render ``state`` for every frame in ``[t_start, t_stop)`` to an MP4 file and return the export metadata."""
    import imageio_ffmpeg

    if not speed > 0:
        raise ValueError(f"speed must be positive, got {speed}")
    exe, how = find_ffmpeg()
    if exe is None:
        raise RuntimeError(how)
    path = check_output_path(path, forbidden_roots)
    path.parent.mkdir(parents=True, exist_ok=True)

    pres = state.pres
    frames = frame_range(pres.frame_times, pres.frame_dt, t_start, t_stop)
    fps = speed / pres.frame_dt
    width, height = EXPORT_SIZE_PX
    dpi = 100
    fig = InspectorFigure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.animated = True

    if how == 'PATH':
        os.environ.setdefault('IMAGEIO_FFMPEG_EXE', exe)
    writer = imageio_ffmpeg.write_frames(
        str(path), size=(width, height), fps=fps, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
        macro_block_size=16, output_params=['-crf', str(crf)])
    writer.send(None)
    t0 = time.time()
    try:
        for k, i in enumerate(frames):
            fig.render(replace(state, frame=i))
            fig.draw()
            img = fig.rgb()
            if img.shape[:2] != (height, width):
                raise RuntimeError(f"rendered frame has shape {img.shape}, expected {(height, width)}")
            writer.send(np.ascontiguousarray(img))
            if progress is not None:
                progress(k + 1, len(frames))
    finally:
        writer.close()

    meta = dict(
        file=str(path), encoder=f'libx264 via {how} ({exe})', fps=fps, speed=speed, frame_dt=pres.frame_dt,
        n_frames=len(frames), frame_start=frames.start, frame_stop=frames.stop,
        t_start=float(pres.frame_times[frames.start]), t_stop=float(pres.frame_times[frames.stop - 1] + pres.frame_dt),
        duration_s=len(frames) / fps, presentation=pres.info.name, key={k: str(v) for k, v in pres.info.key.items()},
        channel=pres.channel, signal=state.signal, view=state.view, display=state.display,
        selected=[int(r) for r in state.shown_selected], eligible=[int(r) for r in state.eligible],
        quality=state.quality_text, view_text=state.view_text, source=pres.source,
        render_seconds=round(time.time() - t0, 1), size=[width, height])
    Path(str(path) + '.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


def read_video(path: str | Path) -> tuple[dict, list]:
    """Decode an exported video (for checks): (metadata reported by ffmpeg, list of RGB frames)."""
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()
    w, h = meta['size']
    frames = [np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3) for buf in reader]
    return meta, frames
