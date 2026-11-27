"""CellViz notebook viewer (read-only).

Select a presentation (and channel) of one field, play back the functional movie with the ROIs of that presentation's
mask, and follow the traces of selected ROIs (or the eligible population) building up in sync with the movie, next to
their trial mean ± SD.

Usage (pipeline, after RoiMask / Traces (/ QI tables) are populated)::

    from djimaging.cellviz import CellViz
    CellViz.list_fields(my_schema)                                    # fields with traces
    CellViz.from_schema(my_schema, field_key=dict(date='2026-09-16', field='GCL0'))

``my_schema`` is your schema module, ``globals()`` after ``from my_schema import *``, or its ``schema`` object.

Usage (offline, no database)::

    from djimaging.cellviz import LocalFieldSource, CellViz
    CellViz(LocalFieldSource('/path/to/20260916/1', field='GCL0'))

Field view overlays (``Overlay`` dropdown, ``set_overlay``), in the Recording and the HighRes view: ROI outlines of the
eligible ROIs, the ROI mask (filled labels, ROIs excluded by the quality filter in grey), the ROI activity (each ROI
filled by its trace value at the current time) or none. The selected ROIs are always outlined in their colour.

MP4 exports go to the folder in the export row, by default ``<processed store>/cellviz/<schema>`` (see ``export``).

Display backends: with ``%matplotlib widget`` (needs the optional ``ipympl`` package) ROIs can be clicked in the field
view or the heatmap and the trace panels can be clicked to seek. Without it, the figure is shown as an image that
scales with the notebook width and ROIs are selected with the list; everything else is identical.

Layout: the widgets wrap to the width of the notebook pane. The figure is ``width`` CSS pixels wide (default 800,
which leaves room for the ROI list beside it in a typical JupyterLab pane at 100 % zoom); the image backend shrinks
it to narrower panes, the ipympl canvas can be resized with its corner handle.

Playback is tied to the recording clock: the play widget advances ``step`` frames every ``step * frame_dt / speed``
seconds of wall-clock time. If rendering is slower than that, the step grows (frames are skipped), and play ticks that
queued up while a frame was rendering are dropped instead of being rendered late. Per frame only the changing artists
are redrawn (blitting, see ``render``). ``playback_stats()`` reports the measured per-frame times.
"""
from __future__ import annotations

import html
import io
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from djimaging.cellviz.data import PresentationData
from djimaging.cellviz.export import export_mp4, find_ffmpeg, processed_export_dir
from djimaging.cellviz.quality import classify
from djimaging.cellviz.registration import map_labels_to_highres, register_highres, rois_outside
from djimaging.cellviz.render import OVERLAYS, CellVizFigure, ViewState
from djimaging.cellviz.timing import frame_index_at

SPEEDS = (0.25, 0.5, 1.0, 2.0)
DEFAULT_THRESHOLDS = {'qidx': 0.35, 'd_qi': 0.6}  # common starting points; the filter itself starts off
SIDEBAR_WIDTH = '230px'


class CellViz:
    """Interactive CellViz viewer for a field source (``LocalFieldSource`` or ``DataJointFieldSource``).

    In the pipeline, open it with ``CellViz.from_schema(schema, field_key)``.

    The state can also be driven from code (``select_presentation``, ``set_channel``, ``select_rois``,
    ``set_threshold``, ``set_view``, ``set_overlay``, ``set_time``, ``export``), which is what the tests do.

    Parameters
    ----------
    width : figure width in CSS pixels (height follows at a fixed aspect ratio).
    export_dir : folder for MP4 exports; default ``<processed store>/cellviz/<schema>`` (``~/cellviz_exports``
        without a local ``processed`` store). Can be changed in the export row.
    overlay : initial field overlay, one of ``OVERLAYS`` ('outlines', 'mask', 'activity', 'none').
    """

    @classmethod
    def from_schema(cls, schema, field_key=None, raw_id=None, preprocess_id=None, **kwargs) -> "CellViz":
        """CellViz for one field of a populated, activated schema (read-only); see ``open_cellviz``."""
        return open_cellviz(schema, field_key=field_key, raw_id=raw_id, preprocess_id=preprocess_id, **kwargs)

    @staticmethod
    def list_fields(schema):
        """Fields of ``schema`` that CellViz can show (ROI mask + Traces), as a DataFrame of field keys."""
        import pandas as pd
        from djimaging.cellviz.dj_adapter import CellVizTables, list_fields
        return pd.DataFrame(list_fields(CellVizTables.from_schema(schema)))

    def __init__(self, source, presentation: Optional[str] = None, channel: Optional[str] = None,
                 backend: str = 'auto', width: int = 800, dpi: int = 100, export_dir=None, figsize=None,
                 overlay: str = 'outlines'):
        import ipywidgets as widgets
        self._w = widgets
        self.source = source
        import matplotlib
        if backend == 'auto':
            backend = 'ipympl' if any(s in matplotlib.get_backend().lower() for s in ('ipympl', 'widget')) else 'image'
        if backend not in ('ipympl', 'image'):
            raise ValueError(f"backend must be 'auto', 'ipympl' or 'image', got {backend!r}")
        self.backend = backend
        figsize = figsize or (width / dpi, round(0.69 * width) / dpi)
        self.width_px = int(round(figsize[0] * dpi))
        if backend == 'ipympl':
            import matplotlib.pyplot as plt
            with plt.ioff():
                fig = plt.figure(figsize=figsize, dpi=dpi)
            self.figure = CellVizFigure(fig, header=False)
        else:
            self.figure = CellVizFigure(figsize=figsize, dpi=dpi, header=False)
        self.figure.animated = True  # blitting in both backends (ipympl sends only the changed pixels)
        if export_dir is not None:
            self._export_dir_source = 'given'
        elif hasattr(source, 'default_export_dir'):
            export_dir, self._export_dir_source = source.default_export_dir()
        else:
            export_dir, self._export_dir_source = processed_export_dir()
        self._initial_export_dir = Path(export_dir).expanduser()
        if overlay not in OVERLAYS:
            raise ValueError(f"overlay must be one of {list(OVERLAYS)}, got {overlay!r}")

        self.infos = {i.name: i for i in source.list_presentations()}
        if not self.infos:
            raise ValueError("source has no presentations")
        self.pres: Optional[PresentationData] = None
        self.selected: list = []          # selection order defines colours; kept across filter / presentation changes
        self.threshold: Optional[float] = None
        self.view = 'native'
        self.display = 'all'
        self.overlay = overlay
        self.signal: Optional[str] = None
        self.frame = 0
        self._highres = {}                 # channel -> (HighResImage | None, note)
        self._registration = {}            # (presentation, channel) -> (Registration, hr_rois | None)
        self._registration_note = ''
        self._suspend = False
        self._render_time = 0.
        self._last_tick_end = 0.
        self._play_anchor = None           # (wall-clock time, frame) when playback (re)started
        self._last_tick_frame = 0
        self._last_raw_tick = 0             # last frame value the browser's play timer sent
        self.dropped_ticks = 0
        self.timings = deque(maxlen=500)   # per frame-only render: dict(prepare, draw, encode, total) in seconds
        self._qi_metric = None
        self.last_export = None

        self._build_widgets()
        first = presentation or next((n for n, i in self.infos.items() if i.has_mask), next(iter(self.infos)))
        self._load(first, channel)

    # =================================================================== public API
    def select_presentation(self, name: str, channel: Optional[str] = None) -> None:
        self._load(name, channel)

    def set_channel(self, channel: str) -> None:
        self._load(self.pres.info.name, channel)

    def set_signal(self, signal: str) -> None:
        self.w_signal.value = signal

    def set_view(self, view: str) -> None:
        self.w_view.value = view

    def set_display(self, display: str) -> None:
        self.w_display.value = display

    def set_overlay(self, overlay: str) -> None:
        """Field view overlay: 'outlines', 'mask' (ROI mask), 'activity' or 'none' (Recording and HighRes view)."""
        if overlay not in OVERLAYS:
            raise ValueError(f"overlay must be one of {list(OVERLAYS)}, got {overlay!r}")
        self.w_overlay.value = overlay

    @property
    def export_dir(self) -> Path:
        """Folder of the MP4 exports (the export row's folder field)."""
        return Path(self.w_export_dir.value.strip()).expanduser()

    @export_dir.setter
    def export_dir(self, path) -> None:
        self.w_export_dir.value = str(path)

    def set_threshold(self, threshold: Optional[float]) -> None:
        """``None`` turns the quality filter off; otherwise ROIs with QI >= threshold are eligible."""
        if threshold is None:
            self.w_qi_on.value = False
        else:
            self.w_qi.value = float(threshold)
            self.w_qi_on.value = True
        self._on_quality()

    def select_rois(self, roi_ids) -> None:
        self.selected = [int(r) for r in roi_ids]
        self._sync_roi_list()
        self.render()

    def toggle_roi(self, roi_id: int) -> None:
        roi_id = int(roi_id)
        if roi_id in self.selected:
            self.selected.remove(roi_id)
        elif roi_id in self.eligible:
            self.selected.append(roi_id)
        else:
            self._status(f"ROI {roi_id} is not selectable: {self._why_not_eligible(roi_id)}")
            return
        self._sync_roi_list()
        self.render()

    def set_frame(self, index: int) -> None:
        index = int(np.clip(index, 0, self.pres.n_frames - 1))
        if self.w_frame.value != index:
            self.w_frame.value = index  # observer renders
        else:
            self.frame = index
            self.render(frame_only=True)

    def set_time(self, t: float) -> None:
        """Show the latest frame starting at or before ``t`` (seconds, presentation clock)."""
        self.set_frame(frame_index_at(self.pres.frame_times, t))

    @property
    def current_time(self) -> float:
        return float(self.pres.frame_times[self.frame])

    @property
    def classes(self):
        return classify(self.pres.roi_ids, self.pres.quality, self.threshold)

    @property
    def eligible(self) -> tuple:
        return self.classes.eligible

    @property
    def state(self) -> ViewState:
        """The current view as a ``ViewState`` (also what ``export`` renders)."""
        hr, hr_rois, view_text = None, None, 'recording resolution'
        if self.view == 'highres':
            hr, _ = self._highres.get(self.pres.channel, (None, ''))
            reg, hr_rois = self._registration.get((self.pres.info.name, self.pres.channel), (None, None))
            view_text = f"HighRes, ROIs registered (shift {tuple(round(v, 2) for v in reg.shift_func_px)} px)" \
                if reg is not None and reg.ok else 'HighRes'
        return ViewState(
            pres=self.pres, signal=self.signal, frame=self.frame, view=self.view, highres=hr, hr_rois=hr_rois,
            overlay=self.overlay, display=self.display, eligible=self.eligible, selected=tuple(self.selected),
            field_label=getattr(self.source, 'label', ''), quality_text=self._quality_text(), view_text=view_text)

    def export(self, path=None, t_start: Optional[float] = None, t_stop: Optional[float] = None,
               speed: float = 1.0) -> dict:
        """Export the current view to MP4 (see ``export.export_mp4``); independent of the preview speed.

        ``path``: an ``.mp4`` file, or a folder (the file name is generated); default: ``export_dir``."""
        if path is None or Path(path).suffix.lower() != '.mp4':
            stem = f"{self.pres.info.name.split()[0]}_{self.pres.channel}_{time.strftime('%Y%m%d-%H%M%S')}"
            folder = self.export_dir if path is None else Path(path).expanduser()
            path = folder / f"{_safe(getattr(self.source, 'label', 'field'))}_{_safe(stem)}.mp4"
        meta = export_mp4(self.state, path, t_start=t_start, t_stop=t_stop, speed=speed,
                          forbidden_roots=self._forbidden_roots(),
                          progress=lambda k, n: setattr(self.w_export_status, 'value', f'rendering {k}/{n}'))
        self.last_export = meta
        self.w_export_status.value = (f"wrote {html.escape(meta['file'])} ({meta['n_frames']} frames, "
                                      f"{meta['t_start']:.2f}-{meta['t_stop']:.2f} s, {meta['fps']:.2f} fps)")
        return meta

    def _forbidden_roots(self) -> list:
        """Read-only data folders of the source (reference store / experiment folder): never export into them."""
        return list(getattr(self.source, 'data_roots', ())) + \
            [p for p in [getattr(self.source, 'experiment_dir', None)] if p is not None]

    def playback_stats(self) -> dict:
        """Median per-frame times (ms) of the frame-only renders so far: preparing the artists, Matplotlib drawing,
        image encoding (image backend; ipympl encodes its diff image inside ``draw``) and total."""
        if not self.timings:
            return {}
        out = {k: round(1000 * float(np.median([t[k] for t in self.timings])), 1)
               for k in ('prepare', 'draw', 'encode', 'total')}
        out.update(n_frames=len(self.timings), dropped_ticks=self.dropped_ticks, backend=self.backend,
                   play_step=self.w_play.step, play_interval_ms=self.w_play.interval)
        return out

    @property
    def widget(self):
        return self._root

    def _ipython_display_(self):
        from IPython.display import display
        display(self._root)

    # =================================================================== widgets
    def _build_widgets(self) -> None:
        w = self._w
        lay = w.Layout
        wrap = dict(flex_flow='row wrap', align_items='center', width='100%')
        self.w_title = w.HTML(layout=lay(margin='0 12px 0 0'))
        self.w_pres = w.Dropdown(options=list(self.infos), description='Stimulus',
                                 layout=lay(width='auto', flex='1 1 260px', max_width='380px'),
                                 style=dict(description_width='60px'))
        self.w_channel = w.Dropdown(description='Channel', layout=lay(width='170px'),
                                    style=dict(description_width='55px'))
        self.w_signal = w.Dropdown(description='Signal', layout=lay(width='180px'),
                                   style=dict(description_width='45px'))
        self.w_view = w.ToggleButtons(options=[('Recording', 'native'), ('HighRes', 'highres')], value='native',
                                      style=dict(button_width='88px'), tooltips=[
                                          'functional movie at recording resolution',
                                          'static HighRes image with the registered ROIs'])
        self.w_overlay = w.Dropdown(options=[(label, key) for key, label in OVERLAYS.items()], value=self.overlay,
                                    description='Overlay', layout=lay(width='190px'),
                                    style=dict(description_width='52px'),
                                    tooltip='drawn on the field view (Recording and HighRes); selected ROIs are '
                                            'always outlined')
        self.w_display = w.ToggleButtons(options=[('All eligible ROIs', 'all'), ('Selected ROIs', 'selected')],
                                         value='all', style=dict(button_width='118px'))

        # sidebar: quality filter directly above the ROI list
        self.w_qi_on = w.Checkbox(value=False, description='QI filter', indent=False, layout=lay(width='auto'))
        self.w_qi = w.FloatSlider(value=0.35, min=0, max=1, step=0.01, description='≥', readout_format='.2f',
                                  continuous_update=False, layout=lay(width='100%'),
                                  style=dict(description_width='16px'))
        self.w_qi_info = w.HTML(layout=lay(width='100%'))
        self.w_rois = w.SelectMultiple(options=[], rows=14, layout=lay(width='100%'))
        self.w_clear = w.Button(description='clear selection', layout=lay(width='auto'))
        self.w_selection_note = w.HTML(layout=lay(width='100%'))

        self.w_play = w.Play(min=0, max=0, step=1, interval=67, description='play')
        self.w_frame = w.IntSlider(min=0, max=0, step=1, readout=False, continuous_update=True,
                                   layout=lay(width='auto', flex='1 1 220px'))
        w.jslink((self.w_play, 'value'), (self.w_frame, 'value'))
        self.w_speed = w.Dropdown(options=[(f'{s:g}x', s) for s in SPEEDS], value=1.0, description='speed',
                                  layout=lay(width='120px'), style=dict(description_width='40px'))
        self.w_time = w.HTML(layout=lay(width='auto', margin='0 0 0 8px'))
        self.w_t0 = w.FloatText(description='from (s)', layout=lay(width='140px'), style=dict(description_width='55px'))
        self.w_t1 = w.FloatText(description='to (s)', layout=lay(width='125px'), style=dict(description_width='40px'))
        self.w_export_speed = w.Dropdown(options=[(f'{s:g}x', s) for s in SPEEDS], value=1.0,
                                         description='export speed', layout=lay(width='160px'),
                                         style=dict(description_width='80px'))
        self.w_export_dir = w.Text(value=str(self._initial_export_dir), description='folder',
                                   layout=lay(width='auto', flex='1 1 320px'), style=dict(description_width='42px'),
                                   tooltip=f'export folder (default from {self._export_dir_source})')
        self.w_export = w.Button(description='Export MP4', icon='film', layout=lay(width='120px'))
        ffmpeg, how = find_ffmpeg()
        self.w_export_status = w.HTML(value='' if ffmpeg else f'<span style="color:#a00">export unavailable: {how}</span>',
                                      layout=lay(width='auto'))
        self.w_export.disabled = ffmpeg is None
        self.w_status = w.HTML()
        self.w_notes = w.HTML()  # content of the collapsed Details section
        self.w_details = w.Accordion(children=[self.w_notes], selected_index=None)
        self.w_details.set_title(0, 'Details')

        if self.backend == 'image':
            image_layout = lay(width='100%', max_width=f'{self.width_px}px', height='auto')
            self.w_canvas = w.Image(format='png', layout=image_layout)
            canvas_box = w.Box([self.w_canvas], layout=lay(flex=f'1 1 {min(self.width_px, 480)}px',
                                                           max_width=f'{self.width_px}px', min_width='0'))
        else:
            canvas = self.figure.fig.canvas
            canvas.header_visible = False
            canvas.footer_visible = False
            canvas.toolbar_position = 'left'
            canvas.mpl_connect('button_press_event', self._on_click)
            _use_fast_png(canvas)
            # A canvas view nested in other widgets can be created after ipympl's initial 'resize' message was
            # handled; it then starts from the synced `_size` trait, which ipympl only sets in that message (0 x 0
            # until then: an invisible figure). Set it now so the view starts at the figure size.
            if hasattr(canvas, '_size'):
                w_px, h_px = self.figure.fig.get_size_inches() * self.figure.fig.dpi
                ratio = getattr(canvas, 'device_pixel_ratio', 1) or 1
                canvas._size = (float(w_px / ratio), float(h_px / ratio))
            self.w_canvas = canvas
            canvas_box = w.Box([canvas], layout=lay(flex='0 1 auto', max_width='100%', overflow='auto'))

        self.w_pres.observe(lambda ch: self._load(ch['new'], None) if not self._suspend else None, names='value')
        self.w_channel.observe(lambda ch: self._load(self.pres.info.name, ch['new'])
                               if not self._suspend and self.pres is not None else None, names='value')
        self.w_signal.observe(lambda ch: self._set('signal', ch['new']), names='value')
        self.w_view.observe(lambda ch: self._on_view(ch['new']), names='value')
        self.w_overlay.observe(lambda ch: self._set('overlay', ch['new']), names='value')
        self.w_display.observe(lambda ch: self._set('display', ch['new']), names='value')
        self.w_rois.observe(self._on_roi_list, names='value')
        self.w_clear.on_click(lambda _: self.select_rois([]))
        self.w_qi_on.observe(lambda ch: self._on_quality(), names='value')
        self.w_qi.observe(lambda ch: self._on_quality(), names='value')
        self.w_frame.observe(self._on_frame, names='value')
        self.w_play.observe(self._on_playing, names='playing')
        self.w_speed.observe(lambda ch: self._update_interval(), names='value')
        self.w_export.on_click(lambda _: self._on_export())

        row = lambda children, **kw: w.HBox(children, layout=lay(**{**wrap, **kw}))  # noqa: E731
        sidebar = w.VBox([self.w_qi_on, self.w_qi, self.w_qi_info, self.w_rois, self.w_clear, self.w_selection_note],
                         layout=lay(flex=f'1 1 {SIDEBAR_WIDTH}', max_width='320px', min_width='200px',
                                    margin='0 0 0 8px', overflow='hidden'))
        self._root = w.VBox([
            row([self.w_title, self.w_pres, self.w_channel, self.w_signal]),
            row([self.w_view, self.w_display, self.w_overlay]),
            row([canvas_box, sidebar], align_items='flex-start'),
            row([self.w_play, self.w_frame, self.w_speed, self.w_time]),
            row([self.w_t0, self.w_t1, self.w_export_speed, self.w_export_dir, self.w_export]),
            row([self.w_export_status]),
            self.w_status, self.w_details], layout=lay(width='100%', max_width='100%'))

    # =================================================================== loading
    def _load(self, name: str, channel: Optional[str]) -> None:
        info = self.infos[name]
        channel = channel or (self.pres.channel if self.pres is not None and self.pres.channel in info.channels
                              else info.channels[0])
        self._status(f"loading {name} ({channel}) ...")
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pres = self.source.load_presentation(name, channel)
        self.pres = pres
        self.frame = 0
        self._suspend = True
        try:
            self.w_pres.value = name
            self.w_channel.options = list(info.channels)
            self.w_channel.value = channel
            self.w_signal.options = list(pres.signals) or ['(no traces)']
            self.signal = ('preprocessed' if 'preprocessed' in pres.signals else next(iter(pres.signals), None))
            self.w_signal.value = self.signal or '(no traces)'
            for wid in (self.w_frame, self.w_play):
                wid.value = 0
                wid.max = pres.n_frames - 1
            self.w_t0.value = round(float(pres.frame_times[0]), 3)
            self.w_t1.value = round(pres.t_end, 3)
            self._setup_quality_widgets()
        finally:
            self._suspend = False
        self._update_interval()
        if self.view == 'highres' and not self._prepare_highres():
            self.view = 'native'
            self._suspend = True
            self.w_view.value = 'native'
            self._suspend = False
        self._sync_roi_list()
        self._status(f"loaded in {time.time() - t:.1f} s")
        self.render()

    def _setup_quality_widgets(self) -> None:
        q = self.pres.quality
        if q is None:
            self.w_qi_on.value = False
            self.w_qi_on.disabled = self.w_qi.disabled = True
            self.w_qi_on.description = 'QI filter'
            self.threshold = None
            self._qi_metric = None
            return
        self.w_qi_on.disabled = self.w_qi.disabled = False
        if q.metric != self._qi_metric:  # other metric (e.g. chirp -> moving bar): filter off, metric default
            vals = np.array([v for v in q.values.values() if np.isfinite(v)])
            hi = max(1.0, float(np.ceil(vals.max() * 10) / 10)) if vals.size else 1.0
            self.w_qi.max = hi
            self.w_qi.value = min(DEFAULT_THRESHOLDS.get(q.metric, 0.5), hi)
            self.w_qi_on.value = False
            self.w_qi_on.description = f'QI filter ({q.metric})'
            self._qi_metric = q.metric
        self.threshold = float(self.w_qi.value) if self.w_qi_on.value else None

    def _prepare_highres(self) -> bool:
        """Load + register the HighRes image for the current presentation/channel. False (with message) if unavailable."""
        ch = self.pres.channel
        if ch not in self._highres:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self._highres[ch] = self.source.load_highres(ch)
        hr, note = self._highres[ch]
        if hr is None:
            self._registration_note = f"HighRes unavailable: {note}"
            self._status(f"HighRes view unavailable: {note}. Playback stays at recording resolution.")
            return False
        key = (self.pres.info.name, ch)
        if key not in self._registration:
            if self.pres.rois is None:
                reg, hr_rois = register_highres(self.pres, hr), None
            else:
                reg = register_highres(self.pres, hr)
                hr_rois = map_labels_to_highres(self.pres.rois, reg, hr.image.shape) if reg.ok else None
            self._registration[key] = (reg, hr_rois)
        reg, hr_rois = self._registration[key]
        if not reg.ok:
            self._registration_note = f"HighRes registration failed: {reg.reason}"
            self._status("HighRes view unavailable: the ROI registration could not be verified (see Details). "
                         "Playback stays at recording resolution.")
            return False
        cut = rois_outside(self.pres.rois, hr_rois) if self.pres.rois is not None and hr_rois is not None else []
        self._registration_note = (f"HighRes registered: {reg.reason}" + (f"; ROIs cut at the border: {cut}" if cut
                                   else '') + (f"; {note}" if note else '') + f"; source {hr.source}")
        cut_text = f" ({len(cut)} cut at the border)" if cut else ''
        self._status(f"HighRes: ROIs registered to the HighRes image{cut_text}")
        return True

    # =================================================================== events
    def _set(self, attr: str, value) -> None:
        if self._suspend:
            return
        if attr == 'signal' and value not in self.pres.signals:
            value = None
        setattr(self, attr, value)
        self.render()

    def _on_view(self, view: str) -> None:
        if self._suspend:
            return
        if view == 'highres' and not self._prepare_highres():
            self._suspend = True
            self.w_view.value = 'native'
            self._suspend = False
            view = 'native'
        self.view = view
        self.render()

    def _on_quality(self) -> None:
        if self._suspend or self.pres is None:
            return
        self.threshold = float(self.w_qi.value) if (self.w_qi_on.value and self.pres.quality is not None) else None
        self._sync_roi_list()
        self.render()

    def _on_roi_list(self, change) -> None:
        if self._suspend:
            return
        chosen = [int(v) for v in change['new']]
        eligible = set(self.eligible)
        kept = [r for r in self.selected if r in chosen or r not in eligible]  # keep hidden (ineligible) selections
        self.selected = kept + [r for r in chosen if r not in kept]
        self.render()

    def _on_playing(self, change) -> None:
        frame = int(self.w_frame.value)
        self._play_anchor = (time.perf_counter(), frame) if change['new'] else None
        self._last_raw_tick = self._last_tick_frame = frame

    def _clock_frame(self, tick_frame: int, now: float) -> int:
        """Frame due at wall-clock time ``now`` since playback started (recording time = elapsed * speed).

        The play widget's browser timer fires late under load, and each tick only advances ``step`` frames, so
        counting ticks lets playback fall behind the recording. The frame is therefore taken from the clock; the
        widget is moved ahead when it lags. A tick far from the last one (user scrubbing while playing) re-anchors.
        """
        step = max(1, int(self.w_play.step))
        anchor = self._play_anchor
        # ticks the browser sent before the widget was moved ahead lie between its own last tick and the frame shown
        stale = self._last_raw_tick <= tick_frame <= self._last_tick_frame
        self._last_raw_tick = tick_frame
        if anchor is None or (not stale and abs(tick_frame - self._last_tick_frame) > 4 * step):
            self._play_anchor = anchor = (now, tick_frame)
        due = anchor[1] + int((now - anchor[0]) * self.w_speed.value / self.pres.frame_dt)
        if stale:
            due = max(due, self._last_tick_frame)  # never step back for a stale tick
        due = min(due, self.pres.n_frames - 1)
        if due > tick_frame:
            self._suspend = True  # the echo of this value from the browser is the frame rendered here
            try:
                self.w_play.value = due
            finally:
                self._suspend = False
            return due
        return tick_frame

    def _on_frame(self, change) -> None:
        if self._suspend:
            return
        start = time.perf_counter()
        frame = int(change['new'])
        playing = bool(getattr(self.w_play, 'playing', False))
        if playing:
            frame = self._clock_frame(frame, start)
        self._last_tick_frame = frame
        already_shown = self.figure.state is not None and self.figure.state.frame == frame and playing
        self.frame = frame
        # A play tick handled right after the previous one finished was queued while that frame rendered: it is
        # already late, and the next tick (at the right recording time) follows within one interval. Drop it.
        queued = (playing and frame < self.pres.n_frames - 1
                  and start - self._last_tick_end < 0.2e-3 * self.w_play.interval)
        if queued or already_shown:
            self.dropped_ticks += queued
        else:
            self.render(frame_only=True)
        self._last_tick_end = time.perf_counter()

    def _on_click(self, event) -> None:
        toolbar = getattr(self.figure.fig.canvas, 'toolbar', None)
        if toolbar is not None and getattr(toolbar, 'mode', ''):
            return
        roi = self.figure.roi_at(event.inaxes, event.xdata, event.ydata)
        if roi is not None:
            self.toggle_roi(roi)
            return
        t = self.figure.time_at(event.inaxes, event.xdata)
        if t is not None:
            self.set_time(t)

    def _on_export(self) -> None:
        try:
            self.export(t_start=self.w_t0.value, t_stop=self.w_t1.value, speed=self.w_export_speed.value)
        except Exception as e:  # noqa: BLE001 (shown in the widget)
            self.w_export_status.value = f'<span style="color:#a00">export failed: {type(e).__name__}: {e}</span>'

    def _update_interval(self) -> None:
        if self.pres is None:
            return
        interval = 1000 * self.pres.frame_dt / self.w_speed.value
        # keep wall-clock speed if rendering is slower than the interval: skip frames instead of lagging
        step = max(1, int(np.ceil(1000 * self._render_time / interval))) if self._render_time else 1
        self.w_play.interval = max(10, int(round(interval * step)))
        self.w_play.step = step
        if self._play_anchor is not None and self.w_speed.value != getattr(self, '_anchor_speed', None):
            self._play_anchor = (time.perf_counter(), self.frame)  # speed changed while playing: restart the clock
        self._anchor_speed = self.w_speed.value

    # =================================================================== rendering
    def render(self, frame_only: bool = False) -> None:
        if self.pres is None:
            return
        t = time.perf_counter()
        self.figure.render(self.state)
        t_prep = time.perf_counter()
        canvas = self.figure.fig.canvas
        if self.backend == 'ipympl' and getattr(canvas, 'syncing_data_url', False):
            # the browser side of the canvas is not connected yet: let it request the first image itself
            # (ipympl's draw handshake); pushing images before that leaves the figure blank in the notebook
            canvas.draw_idle()
        else:
            self.figure.draw()
        t_draw = time.perf_counter()
        if self.backend == 'image':
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(self.figure.rgb()).save(buf, format='PNG', compress_level=1)
            self.w_canvas.value = buf.getvalue()
        t_end = time.perf_counter()
        t0, t1 = self.pres.frame_interval(self.frame)
        self.w_time.value = (f"<span style='font-variant-numeric:tabular-nums'>t = {t0:7.2f} s &nbsp;·&nbsp; "
                             f"frame {self.frame + 1}/{self.pres.n_frames}</span>")
        if frame_only:
            dt = t_end - t
            self.timings.append(dict(prepare=t_prep - t, draw=t_draw - t_prep, encode=t_end - t_draw, total=dt))
            self._render_time = 0.8 * self._render_time + 0.2 * dt if self._render_time else dt
            step = self.w_play.step
            self._update_interval()
            if self.w_play.step != step:
                self._status(f"rendering takes {1000 * self._render_time:.0f} ms per frame: playing every "
                             f"{self.w_play.step}. frame to keep the recording speed")
        else:
            self._update_title_and_notes()

    def _update_title_and_notes(self) -> None:
        p = self.pres
        self.w_title.value = f"<b>{html.escape(getattr(self.source, 'label', ''))}</b>"
        c = self.classes
        q = p.quality
        if q is None:
            self.w_qi_info.value = "<small>no QI for this stimulus (see Details)</small>"
        else:
            n_all = len(p.roi_ids)
            if c.active:
                extra = f"{len(c.below)} below, {len(c.missing)} without value"
                self.w_qi_info.value = f"<b>{len(c.eligible)}</b> of {n_all} ROIs eligible<br><small>{extra}</small>"
            else:
                self.w_qi_info.value = (f"{n_all} ROIs (filter off)" +
                                        (f"<br><small>{len(c.missing)} without QI value</small>" if c.missing else ''))
        in_mask = set(p.roi_ids.tolist())
        hidden = [r for r in self.selected if r in in_mask and r not in set(c.eligible)]
        absent = [r for r in self.selected if r not in in_mask]
        sel_notes = []
        if hidden:
            sel_notes.append(f"selected but hidden by the quality filter or without value: {hidden}")
        if absent:
            sel_notes.append(f"selected ROI IDs not in this presentation's mask: {absent}")
        self.w_selection_note.value = '<small>' + '<br>'.join(sel_notes) + '</small>' if sel_notes else ''

        def section(title, lines):
            lines = [ln for ln in lines if ln]
            return f"<b>{title}</b><br>" + '<br>'.join(html.escape(str(ln)) for ln in lines) if lines else ''

        signal = p.signals.get(self.signal) if self.signal else None
        meta = ', '.join(f"{k}={v}" for k, v in p.meta.items())
        sections = [
            section('Presentation', [f"key: {', '.join(f'{k}={v}' for k, v in p.info.key.items())}",
                                     f"source: {p.source}", f"scan: {meta}" if meta else '',
                                     f"pixel size {p.pixel_size_um:.3f} um, frame_dt {p.frame_dt * 1000:.2f} ms, "
                                     f"{p.n_frames} frames, {p.triggers.size} triggers"]),
            section('Signal', [f"{signal.name} [{signal.units}]: {signal.description}" if signal else 'no traces']),
            section('Trial average', [self._trial_note()]),
            section('Quality', [self._quality_source(), self._quality_text()]),
            section('Mask', [', '.join(f"{k}={v}" for k, v in p.mask_info.items())] +
                    [f"ROI {r}: {msg}" for r, msg in sorted(p.roi_notes.items())]),
            section('HighRes', [self._registration_note]),
            section('Export', [f"default folder from {self._export_dir_source}: {self._initial_export_dir}",
                               "data folders are never written to: " +
                               (', '.join(map(str, self._forbidden_roots())) or '(none known)')]),
            section('Notes', list(getattr(self.source, 'notes', [])) + list(p.notes) + sel_notes +
                    (["image mode: select ROIs in the list (clicking needs %matplotlib widget / ipympl)"]
                     if self.backend == 'image' else [])),
        ]
        self.w_notes.value = '<small>' + '<br><br>'.join(s for s in sections if s) + '</small>'

    def _quality_source(self) -> str:
        q = self.pres.quality
        if q is None:
            return self.pres.quality_unavailable
        ref = f", reference {q.reference:.2f}" if q.reference is not None else ''
        extra = f"; also: {', '.join(q.extra)}" if q.extra else ''
        return f"{q.table}.{q.metric}: {q.source}{ref}{extra}"

    def _trial_note(self) -> str:
        stim = self.pres.stimulus
        if stim is None:
            return 'no stimulus repeat structure: no trial average'
        cond = stim.conditions
        how = (f"one trial per trigger, conditions {cond} (repeats of one condition only)" if cond else
               f"{stim.ntrigger_rep} triggers per repeat")
        return (f"split_trace_by_reps as Snippets: {how}, trigger_delay={stim.trigger_delay}, "
                f"pad_trace={stim.pad_trace}; mean ± SD (ddof=1) across valid repeats; all eligible ROIs: ROIs "
                f"averaged within each repeat first")

    def _sync_roi_list(self) -> None:
        if self.pres is None:
            return
        q = self.pres.quality
        signal = self.pres.signals.get(self.signal) if self.signal else None
        opts = []
        for r in self.eligible:
            label = f"ROI {r}"
            if q is not None:
                v = q.value(r)
                label += f"  {q.metric}={v:.2f}" if np.isfinite(v) else f"  {q.metric}=n/a"
            if signal is not None and r not in signal.traces:
                label += '  (no trace)'
            opts.append((label, r))
        self._suspend = True
        try:
            self.w_rois.options = opts
            self.w_rois.value = tuple(r for r in self.selected if r in set(self.eligible))
        finally:
            self._suspend = False

    def _why_not_eligible(self, roi_id: int) -> str:
        c = self.classes
        if roi_id in c.below:
            return f"below the threshold ({self.pres.quality.metric}={self.pres.quality.value(roi_id):.2f} " \
                   f"< {self.threshold:.2f})"
        if roi_id in c.missing:
            return "no quality value while the filter is active"
        return "not in the mask"

    def _quality_text(self) -> str:
        q = self.pres.quality
        if q is None:
            return "no quality filter"
        if self.threshold is None:
            return f"{q.metric}: filter off"
        return f"{q.metric} >= {self.threshold:.2f}: {self.classes.summary()}"

    def _status(self, text: str) -> None:
        self.w_status.value = f'<small>{html.escape(text)}</small>'


def _use_fast_png(canvas) -> None:
    """Encode this ipympl canvas' frame updates with zlib level 1 instead of PIL's default 6.

    Same protocol as ``FigureCanvasWebAggCore.get_diff_image`` (full image or diff against the last sent buffer), only
    the PNG compression level differs; the browser decodes either. Only this canvas instance is changed.
    """
    import types
    from PIL import Image

    def get_diff_image(self):
        if not self._png_is_old:
            return None
        pixels = np.asarray(self.get_renderer().buffer_rgba())
        buff = pixels.view(np.uint32).squeeze(2)
        if self._force_full or buff.shape != self._last_buff.shape or (pixels[:, :, 3] != 255).any():
            self.set_image_mode('full')
            output = buff
        else:
            self.set_image_mode('diff')
            output = np.where(buff != self._last_buff, buff, 0)
        self._last_buff = buff.copy()
        self._force_full = False
        self._png_is_old = False
        with io.BytesIO() as png:
            Image.fromarray(output.view(dtype=np.uint8).reshape((*output.shape, 4))).save(
                png, format='png', compress_level=1)
            return png.getvalue()

    if hasattr(canvas, '_last_buff') and hasattr(canvas, 'set_image_mode'):
        canvas.get_diff_image = types.MethodType(get_diff_image, canvas)


def _safe(text: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(text)).strip('_')


def open_cellviz(schema, field_key=None, raw_id=None, preprocess_id=None, **kwargs) -> CellViz:
    """CellViz for one field of a populated, activated schema (read-only); display the result in a notebook cell.

    Parameters
    ----------
    schema : your schema module, ``globals()`` after ``from my_schema import *``, the module's ``schema`` object, or a
        ``CellVizTables``. Tables are found by their example-schema names (``Presentation``, ``RoiMask``, ``Traces``,
        and if present ``PreprocessTraces``, ``ChirpQI``, ``OsDsIndexes``, ``HighRes``, ...).
    field_key : the field to show; may be partial (e.g. ``dict(date='2026-09-16', field='GCL0')``) as long as it
        matches one field with traces (``CellViz.list_fields(schema)``), or a longer key, e.g. of a presentation.
        ``None``: the first field with traces.
    raw_id, preprocess_id : pin these if the schema has several parameter sets.
    kwargs : passed to ``CellViz`` (``presentation``, ``channel``, ``overlay``, ``export_dir``, ``width``, ...).
    """
    from djimaging.cellviz.dj_adapter import CellVizTables, DataJointFieldSource, resolve_field_key
    tables = CellVizTables.from_schema(schema)
    if raw_id is None and field_key is not None:
        raw_id = dict(field_key).get('raw_id')
    if raw_id is not None:
        field_key = dict(field_key or {}, raw_id=raw_id)
    field_key, notes = resolve_field_key(tables, field_key)
    source = DataJointFieldSource(tables, field_key, raw_id=raw_id, preprocess_id=preprocess_id)
    source.notes = notes + source.notes
    return CellViz(source, **kwargs)
