"""Matplotlib renderer of CellViz, shared by the notebook viewer and the video export.

Layout (positions computed in pixels, so the field stays square at any figure size)::

    Recording (square field view)  | ROI activity (ROI x time heatmap)
    Response trace (~60 %)          | Trial mean ± SD (~40 %)

Everything that depends on the playback time is drawn from one frame index ``i`` whose interval
``[frame_times[i], +frame_dt)`` is the current time; traces and the heatmap show all samples with timestamps before the
end of that interval. The trial panel shows the mean ± SD of the current trial's condition (``trials``), revealed up to
the same time relative to the current trial's onset.

Per-frame cost: static parts (images, outlines, ticks, trigger marks, titles) are drawn once into a cached background;
each frame only restores it and redraws the per-frame artists (movie image, cursors, line/band data, labels). This
works for Agg (image backend, MP4 export) and ipympl (``canvas.blit`` sends only the changed pixels). A full redraw
(e.g. after a resize) re-captures the background through ``draw_event``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import matplotlib
import matplotlib.ticker
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import blended_transform_factory

from djimaging.cellviz.data import HighResImage, PresentationData, RoiLabels
from djimaging.cellviz.timing import n_samples_before
from djimaging.cellviz.trials import TrialCache

SELECTION_COLORS = [matplotlib.colormaps['tab10'](i) for i in range(10)]
OUTLINE_COLOR = '#00d0ff'
POPULATION_COLOR = '0.15'
HEATMAP_CMAP = 'RdBu_r'
OVERLAY_CMAP = 'magma'
MAX_TRIGGER_LABELS = 60

# What is drawn on the field view (Recording and HighRes), besides the outlines of the selected ROIs, which are always
# drawn: outlines of the eligible ROIs, the ROI mask (filled labels), the ROI activity (fill coloured by the trace at
# the current time) or nothing.
OVERLAYS = {'outlines': 'ROI outlines', 'mask': 'ROI mask', 'activity': 'ROI activity', 'none': 'none'}
MASK_COLORS = [matplotlib.colormaps['tab20'](i) for i in range(20) if i not in (14, 15)]  # no greys (= excluded)
MASK_ALPHA = 0.45                 # fill of eligible ROIs in the 'mask' overlay
MASK_EXCLUDED = (0.6, 0.6, 0.6, 0.3)  # fill of ROIs excluded by the quality filter in the 'mask' overlay


def mask_color(roi_id: int) -> tuple:
    """Fill colour of a ROI in the 'mask' overlay: fixed per ROI ID, so it is the same in every view and export."""
    return MASK_COLORS[(int(roi_id) * 7) % len(MASK_COLORS)]  # stride 7: consecutive IDs get distinct hues


def _segments_patch(segments: np.ndarray, color, lw: float, zorder: float, alpha: float = 1.) -> PathPatch:
    """All ``segments`` (n, 2, 2) as one compound path: one Agg draw call instead of one per segment (a
    ``LineCollection`` of ~1600 pixel edges costs several ms per frame; this a fraction of that)."""
    codes = np.tile(np.array([MplPath.MOVETO, MplPath.LINETO], dtype=MplPath.code_type), len(segments))
    return PathPatch(MplPath(segments.reshape(-1, 2), codes), fill=False, edgecolor=color, lw=lw, alpha=alpha,
                     zorder=zorder)


def mask_rgba(labels: RoiLabels, eligible) -> np.ndarray:
    """RGBA image ``[y, x, 4]`` of the ROI mask: eligible ROIs filled with ``mask_color``, the other ROIs grey."""
    eligible = set(int(r) for r in eligible)
    rgba = np.zeros(labels.shape + (4,))
    for label, roi_id in labels.label_to_roi_id.items():
        if roi_id in eligible:
            rgba[labels.labels == label] = (*mask_color(roi_id)[:3], MASK_ALPHA)
        else:
            rgba[labels.labels == label] = MASK_EXCLUDED
    return rgba


FS_TITLE, FS_LABEL, FS_TICK, FS_SMALL = 10, 9, 8, 7.5  # points


def outline_segments(pixels: np.ndarray) -> np.ndarray:
    """Line segments along the pixel edges of a boolean ``[row, col]`` mask, in ``imshow`` data coordinates
    (pixel ``(r, c)`` centred at ``(x=c, y=r)``). Valid for ``origin='lower'`` and ``'upper'``. From CellViz."""
    pixels = np.asarray(pixels, dtype=bool)
    padded = np.pad(pixels, 1, constant_values=False)
    rows, cols = np.nonzero(padded[1:, 1:-1] != padded[:-1, 1:-1])
    horizontal = [((c - 0.5, r - 0.5), (c + 0.5, r - 0.5)) for r, c in zip(rows, cols)]
    rows, cols = np.nonzero(padded[1:-1, 1:] != padded[1:-1, :-1])
    vertical = [((c - 0.5, r - 0.5), (c - 0.5, r + 0.5)) for r, c in zip(rows, cols)]
    return np.array(horizontal + vertical, dtype=float).reshape(-1, 2, 2)


@dataclass(frozen=True, eq=False)
class ViewState:
    """Everything the figure shows. Produced by the viewer (or directly, e.g. for the export)."""
    pres: PresentationData
    signal: Optional[str]           # key of pres.signals, None if there are no traces
    frame: int = 0
    view: str = 'native'            # 'native' or 'highres'
    highres: Optional[HighResImage] = None
    hr_rois: Optional[RoiLabels] = None  # ROI labels registered to the HighRes grid
    overlay: str = 'outlines'       # key of OVERLAYS: 'outlines', 'mask', 'activity' or 'none' (both views)
    display: str = 'all'            # 'all' (eligible population) or 'selected'
    eligible: tuple = ()            # ROI IDs passing the quality filter
    selected: tuple = ()            # selected ROI IDs (in selection order; only eligible ones are drawn)
    field_label: str = ''
    quality_text: str = ''
    view_text: str = ''
    clim: Optional[tuple] = None    # fixed movie contrast; default: 1st / 99.5th percentile of the movie
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.overlay not in OVERLAYS:
            raise ValueError(f"overlay must be one of {list(OVERLAYS)}, got {self.overlay!r}")

    @property
    def shown_selected(self) -> tuple:
        eligible = set(self.eligible)
        return tuple(r for r in self.selected if r in eligible)

    def color_of(self, roi_id: int):
        """Colour by position in the selection (not among the shown ROIs), so it is stable when the filter changes."""
        if roi_id not in self.shown_selected:
            return OUTLINE_COLOR
        return SELECTION_COLORS[self.selected.index(roi_id) % len(SELECTION_COLORS)]


_IDENTITY_FIELDS = ('pres', 'highres', 'hr_rois')


def same_except_frame(a: ViewState, b: ViewState) -> bool:
    """True if two states differ at most in the frame (data objects are compared by identity)."""
    for f in fields(ViewState):
        if f.name == 'frame':
            continue
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if not (va is vb if f.name in _IDENTITY_FIELDS else va == vb):
            return False
    return True


class _SignalCache:
    """Per (presentation, signal): traces resampled on the frame grid for heatmap / overlay / mean (display only)."""

    def __init__(self, pres: PresentationData, signal: Optional[str]):
        self.key = (id(pres), signal)
        self.rows, self.ranges, self._z = {}, {}, {}
        if signal is None or signal not in pres.signals:
            return
        mid = pres.frame_times + pres.frame_dt / 2
        for roi_id, trace in pres.signals[signal].traces.items():
            row = np.interp(mid, trace.times, trace.values, left=np.nan, right=np.nan)
            self.rows[int(roi_id)] = row
            lo, hi = np.nanpercentile(trace.values, [1, 99]) if np.isfinite(trace.values).any() else (0., 1.)
            self.ranges[int(roi_id)] = (lo, hi if hi > lo else lo + 1.)

    def zscored(self, roi_ids) -> np.ndarray:
        out = []
        for r in roi_ids:
            if r not in self._z:
                row = self.rows[r]
                sd = np.nanstd(row)
                self._z[r] = (row - np.nanmean(row)) / (sd if sd > 0 else 1.)
            out.append(self._z[r])
        return np.array(out)

    def mean(self, roi_ids) -> np.ndarray:
        rows = np.array([self.rows[r] for r in roi_ids])
        n = np.isfinite(rows).sum(axis=0)
        return np.where(n > 0, np.nansum(rows, axis=0) / np.maximum(n, 1), np.nan)  # NaN where no ROI has data


def default_clim(pres: PresentationData) -> tuple:
    """Fixed movie contrast: 1st / 99.5th percentile over the movie, without the light-artifact columns."""
    frames = pres.frames[::max(1, pres.n_frames // 200), :, pres.npixartifact:]
    lo, hi = np.percentile(frames, [1, 99.5])
    return float(lo), float(hi if hi > lo else lo + 1)


_STAMP_CACHE: dict = {}


def text_bitmap(text: str, fontsize: float, color, dpi: float, weight: str = 'normal') -> np.ndarray:
    """RGBA bitmap of ``text`` (transparent background), rendered once per (text, size, colour, dpi, weight) and
    cached."""
    key = (text, fontsize, str(color), round(float(dpi), 3), weight)
    if key not in _STAMP_CACHE:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        f = Figure(figsize=(1, 1), dpi=dpi)
        FigureCanvasAgg(f)
        f.patch.set_alpha(0)
        t = f.text(0, 0, text, fontsize=fontsize, color=color, ha='left', va='baseline', weight=weight)
        bb = t.get_window_extent(f.canvas.get_renderer())
        w, h = int(np.ceil(bb.width)) + 2, int(np.ceil(bb.height)) + 2
        f.set_size_inches(w / dpi, h / dpi)
        t.set_position((1 / w, (1 - bb.y0) / h))
        f.canvas.draw()
        _STAMP_CACHE[key] = np.asarray(f.canvas.buffer_rgba()).copy()
        if len(_STAMP_CACHE) > 256:
            _STAMP_CACHE.pop(next(iter(_STAMP_CACHE)))
    return _STAMP_CACHE[key]


class TextStamp:
    """A per-frame label drawn as a cached bitmap: Agg text rendering costs several ms per string and frame, a
    bitmap a fraction of that. Positioned in axes fraction (or ``xycoords``; ``box_alignment`` like ``ha`` / ``va``)."""

    def __init__(self, ax, xy, box_alignment=(1, 1), fontsize=FS_SMALL, color='0.2', zorder=6,
                 xycoords='axes fraction', weight='normal', text=None):
        self.ax, self.fontsize, self.color, self.weight = ax, fontsize, color, weight
        self.image = OffsetImage(np.zeros((1, 1, 4)), zoom=1, dpi_cor=True)
        self.artist = AnnotationBbox(self.image, xy, xycoords=xycoords, frameon=False, annotation_clip=False,
                                     box_alignment=box_alignment, pad=0, zorder=zorder)
        ax.add_artist(self.artist)
        self._text, self._dpi = None, None
        if text is not None:
            self.set_text(text)

    def set_text(self, text: str) -> None:
        dpi = self.ax.figure.dpi
        if (text, dpi) == (self._text, self._dpi):
            return
        self._text, self._dpi = text, dpi
        self.artist.set_visible(bool(text))
        if text:
            self.image.set_data(text_bitmap(text, self.fontsize, self.color, dpi, self.weight))
            self.image.set_zoom(72. / dpi)  # dpi_cor scales by dpi/72: the bitmap keeps its pixel size

    def get_text(self) -> str:
        return self._text or ''


@dataclass
class _TrialSeries:
    """One curve of the trial panel: per condition (x, mean, lo, hi) with the stacking offset already applied."""
    line: object
    band: object
    per_condition: list            # [(x, y, lo, hi, n) | None]


class CellVizFigure:
    """Draws a ``ViewState``. ``render(state)`` decides whether a full rebuild is needed or only the frame changed.

    ``header=True`` (default, used by the MP4 export) writes field, stimulus, channel, signal and the playback time
    into the figure. The notebook viewer shows those in its widgets instead (``header=False``), so each is shown once.
    """

    def __init__(self, figure: Optional[Figure] = None, figsize=(9.0, 6.2), dpi: int = 100, header: bool = True):
        if figure is None:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            figure = Figure(figsize=figsize, dpi=dpi)
            FigureCanvasAgg(figure)
        self.fig = figure
        self.header = header
        self.state: Optional[ViewState] = None
        self._cache: Optional[_SignalCache] = None
        self._trials: Optional[TrialCache] = None
        self._outlines = {}
        self._trigger_marks = {}
        self._frame_artists = []
        self.builds = 0  # number of full rebuilds (for tests / profiling)
        self.animated = False  # blitting: per-frame artists are drawn over a cached background
        self._background = None
        self.fig.canvas.mpl_connect('draw_event', self._on_draw_event)
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)

    # ------------------------------------------------------------------ public
    def render(self, state: ViewState) -> None:
        """Update the figure to ``state``. Only per-frame artists are touched if only ``frame`` changed."""
        prev = self.state
        self.state = state
        if prev is None or not same_except_frame(prev, state):
            self._build()
        self._update_frame()

    def draw(self) -> None:
        """Render to the canvas (Agg / ipympl). With ``animated``, only the per-frame artists are redrawn."""
        canvas = self.fig.canvas
        if not self.animated or self._background is None:
            canvas.draw()  # draw_event -> background captured and per-frame artists drawn on top
            return
        canvas.restore_region(self._background)
        self._draw_frame_artists()
        canvas.blit(self.fig.bbox)

    def rgb(self) -> np.ndarray:
        """Current image as (height, width, 3) uint8 (after ``draw``)."""
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def roi_at(self, ax, xdata: float, ydata: float) -> Optional[int]:
        """ROI ID under a click in the field view or the heatmap (any ROI of the mask; the caller filters)."""
        s = self.state
        if s is None or xdata is None or ydata is None:
            return None
        if ax is self.ax_field:
            labels = s.hr_rois if (s.view == 'highres' and s.hr_rois is not None) else s.pres.rois
            if labels is None:
                return None
            return labels.roi_id_at(int(round(ydata)), int(round(xdata)))
        if ax is self.ax_heat and self._heat_ids:
            k = int(round(ydata))
            return self._heat_ids[k] if 0 <= k < len(self._heat_ids) else None
        return None

    def time_at(self, ax, xdata: float) -> Optional[float]:
        """Recording time of a click: on the trace / heatmap directly, on the trial panel relative to the current
        trial's onset."""
        if xdata is None:
            return None
        if ax in (self.ax_trace, self.ax_heat):
            return float(xdata)
        if ax is self.ax_trial and self._current_trial >= 0:
            return float(self._trials.layout.onsets[self._current_trial] + xdata)
        return None

    # --------------------------------------------------------------- blitting
    def _draw_frame_artists(self) -> None:
        for artist in self._frame_artists:
            self.fig.draw_artist(artist)

    def _on_draw_event(self, event) -> None:
        if not self.animated or self.state is None:
            return
        self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_frame_artists()

    def _on_resize(self, event) -> None:
        self._background = None
        if self.state is not None:
            self._place_axes()

    # --------------------------------------------------------------- building
    def _place_axes(self) -> None:
        """Axes rectangles from pixel margins: square field, heatmap beside it, 60/40 bottom row."""
        w, h = self.fig.get_size_inches() * self.fig.dpi
        px = self.fig.dpi / 72.  # pixels per point
        left, right = 6 * FS_LABEL * px, 10.  # y tick labels + y label of the trace panel
        top = (3.2 * FS_TITLE * px if self.header else 0.) + 2.2 * FS_TITLE * px
        gap_rows = 1.8 * FS_TICK * px + 2.2 * FS_TITLE * px + 1.6 * FS_SMALL * px  # heat ticks, titles, triggers
        bottom = 2.0 * FS_TICK * px + 1.8 * FS_LABEL * px
        usable_h = h - top - gap_rows - bottom
        side = max(40., min(0.55 * usable_h, 0.40 * (w - left - right)))
        y_top = h - top - side
        heat_left = left + side + 5.5 * FS_LABEL * px
        bottom_h = max(30., y_top - gap_rows - bottom)
        mid_gap = 2.5 * FS_LABEL * px
        trace_w = 0.6 * (w - left - right - mid_gap)

        def rect(x0, y0, ww, hh):
            return [x0 / w, y0 / h, max(ww, 1.) / w, max(hh, 1.) / h]

        self.ax_field.set_position(rect(left, y_top, side, side))
        self.ax_heat.set_position(rect(heat_left, y_top, w - right - heat_left, side))
        self.ax_trace.set_position(rect(left, bottom, trace_w, bottom_h))
        self.ax_trial.set_position(rect(left + trace_w + mid_gap, bottom, w - right - left - trace_w - mid_gap,
                                        bottom_h))

    def _build(self) -> None:
        s = self.state
        pres = s.pres
        self._background = None
        if self._cache is None or self._cache.key != (id(pres), s.signal):
            self._cache = _SignalCache(pres, s.signal)
        if self._trials is None or self._trials.key != (id(pres), s.signal):
            self._trials = TrialCache(pres, s.signal)
        self.builds += 1
        self._reset_axes()
        self._place_axes()
        self._frame_artists = []
        self._current_trial = -1

        self.time_text = None
        if self.header:
            units = pres.signals[s.signal].units if s.signal in pres.signals else ''
            stim = pres.info.stim_name or pres.info.name
            self.fig.text(0.01, 0.99, f"{s.field_label}  ·  {stim}  ·  {pres.channel}", ha='left', va='top',
                          fontsize=FS_TITLE + 1, weight='bold')
            self.fig.text(0.01, 0.99 - 1.6 * (FS_TITLE + 1) / 72 / self.fig.get_size_inches()[1],
                          f"{s.signal or 'no traces'} [{units}]  ·  {s.view_text or s.view}  ·  "
                          f"overlay: {OVERLAYS[s.overlay]}  ·  {s.quality_text}",
                          ha='left', va='top', fontsize=FS_SMALL)
            self.time_text = self.fig.text(0.99, 0.99, '', ha='right', va='top', fontsize=FS_TITLE + 1)

        self._build_field()
        self._build_heatmap()
        self._build_traces()
        self._build_trials()
        if self.time_text is not None:
            self._frame_artists.append(self.time_text)
        if self.animated:
            for artist in self._frame_artists:
                artist.set_animated(True)

    def _reset_axes(self) -> None:
        """Empty the four axes for a rebuild. Creating axes (ticks, spines) is the slowest part of a rebuild, so they
        are created once and afterwards only the artists drawn into them are removed."""
        axes = [getattr(self, name, None) for name in ('ax_field', 'ax_heat', 'ax_trace', 'ax_trial')]
        if any(ax is None or ax.figure is not self.fig or ax not in self.fig.axes for ax in axes):
            self.fig.clear()
            self.ax_field = self.fig.add_axes([0, 0, 1, 1])
            self.ax_heat = self.fig.add_axes([0, 0, 1, 1])
            self.ax_trace = self.fig.add_axes([0, 0, 1, 1])
            self.ax_trial = self.fig.add_axes([0, 0, 1, 1], sharey=self.ax_trace)
            return
        for text in list(self.fig.texts):
            text.remove()
        for ax in axes:
            for artist in [*ax.lines, *ax.collections, *ax.images, *ax.patches, *ax.texts, *ax.artists]:
                artist.remove()
            for loc in ('left', 'center', 'right'):
                ax.set_title('', loc=loc)
            ax.set_xlabel('')
            ax.set_ylabel('')
            for label in ax.get_yticklabels():
                label.set_color('k')
                label.set_fontweight('normal')
            ax.relim()  # forget the data limits of the removed artists
            ax.set_autoscale_on(True)
        for ax in (self.ax_trace, self.ax_trial):
            ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        self.ax_trial.set_xlim(0, 1)

    def _shown_ids(self) -> tuple:
        s = self.state
        return s.shown_selected if s.display == 'selected' else tuple(s.eligible)

    def _build_field(self) -> None:
        s, ax = self.state, self.ax_field
        pres = s.pres
        hr_mode = s.view == 'highres' and s.highres is not None
        if hr_mode:
            img = s.highres.image
            ax.imshow(img, cmap='gray', origin='lower', interpolation='nearest',
                      vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.8))
            labels = s.hr_rois
            ax.set_title('HighRes (static)', fontsize=FS_TITLE, loc='left')
        else:
            clim = s.clim or default_clim(pres)
            self.movie_image = ax.imshow(pres.frames[0], cmap='gray', origin='lower', interpolation='nearest',
                                         vmin=clim[0], vmax=clim[1])
            self._frame_artists.append(self.movie_image)
            labels = pres.rois
            ax.set_title('Recording', fontsize=FS_TITLE, loc='left')
        n_rows, n_cols = (s.highres.image if hr_mode else pres.frames[0]).shape
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.set_anchor('C')
        ax.set_xticks([])
        ax.set_yticks([])

        self.overlay_image = None
        layers = []  # everything drawn above the field image: (zorder, artist)
        if labels is None:
            stamp = TextStamp(ax, (0.5, 0.04), box_alignment=(0.5, 0), color='w', zorder=5, text='no ROI mask')
            layers.append(stamp.artist)
        else:
            layers += self._build_roi_layers(ax, labels)

        if not hr_mode or self.overlay_image is not None:
            # The field image (movie) or the activity fill changes every frame. With blitting, per-frame artists are
            # drawn over the cached background, which would hide everything static above them; so the ROI layers
            # are redrawn every frame as well, in zorder (= the order of a full draw).
            self._frame_artists += sorted(layers, key=lambda a: a.get_zorder())

    def _build_roi_layers(self, ax, labels: RoiLabels) -> list:
        """ROI overlay of the field view (``state.overlay``) and the always shown selection; returns the artists."""
        s = self.state
        key = id(labels)
        if self._outlines.get('key') != key:
            self._outlines = {'key': key, **{int(r): outline_segments(labels.pixels(r)) for r in labels.roi_ids}}
        sel = s.shown_selected
        layers = []
        if s.overlay == 'activity' and s.signal is not None:
            ids = [r for r in s.eligible if r in self._cache.rows and r in labels.label_to_roi_id.values()]
            index_img = np.full(labels.shape, -1, dtype=int)
            for k, r in enumerate(ids):
                index_img[labels.pixels(r)] = k
            self._overlay_ids, self._overlay_index = ids, index_img
            self.overlay_image = ax.imshow(np.zeros(labels.shape + (4,)), origin='lower', interpolation='nearest',
                                           zorder=2)
            self._frame_artists.append(self.overlay_image)
        if s.overlay == 'mask':
            layers.append(ax.imshow(mask_rgba(labels, s.eligible), origin='lower', interpolation='nearest', zorder=2))
        others = [r for r in s.eligible if r in self._outlines and r not in sel and len(self._outlines[r])]
        if others and s.overlay == 'mask':  # thin light edges keep touching ROIs apart
            layers.append(ax.add_patch(_segments_patch(
                np.concatenate([self._outlines[r] for r in others]), 'w', lw=0.4, alpha=0.6, zorder=3)))
        elif others and s.overlay in ('outlines', 'activity'):
            faint = s.display == 'selected'
            layers.append(ax.add_patch(_segments_patch(
                np.concatenate([self._outlines[r] for r in others]), OUTLINE_COLOR, lw=0.5 if faint else 0.8,
                alpha=0.3 if faint else 0.85, zorder=3)))
        for r in sel:
            if r in self._outlines:
                layers.append(ax.add_collection(LineCollection(self._outlines[r], colors=[s.color_of(r)],
                                                               linewidths=2.2, zorder=4)))
                yy, xx = np.nonzero(labels.pixels(r))
                if xx.size:
                    stamp = TextStamp(ax, (xx.max() + 0.8 * (labels.shape[1] / 64), yy.mean()), xycoords='data',
                                      box_alignment=(0, 0.5), fontsize=FS_TICK, color=s.color_of(r), weight='bold',
                                      zorder=5, text=str(r))
                    layers.append(stamp.artist)
        return layers

    def _build_heatmap(self) -> None:
        s, ax = self.state, self.ax_heat
        pres = s.pres
        ids = [r for r in self._shown_ids() if r in self._cache.rows]
        self._heat_ids = ids
        t0, t1 = float(pres.frame_times[0]), pres.t_end
        ax.set_xlim(t0, t1)
        ax.tick_params(labelsize=FS_TICK)
        self.heat_cover = self.heat_cursor = None
        ax.set_title('ROI activity' + (' (z-score)' if ids else ''), fontsize=FS_TITLE, loc='left')
        if not ids:
            reason = 'no traces' if not self._cache.rows else 'no ROI to show (selection empty or filtered out)'
            ax.text(0.5, 0.5, reason, ha='center', va='center', transform=ax.transAxes, fontsize=FS_LABEL)
            ax.set_yticks([])
            return
        mat = self._cache.zscored(ids)
        ax.imshow(mat, aspect='auto', cmap=HEATMAP_CMAP, vmin=-3, vmax=3, interpolation='nearest',
                  extent=(t0, t1, len(ids) - 0.5, -0.5))
        ax.set_xlim(t0, t1)
        ax.set_ylim(len(ids) - 0.5, -0.5)
        self.heat_cover = Rectangle((t1, -0.5), 0, len(ids), facecolor=ax.get_facecolor(), edgecolor='none', zorder=3)
        ax.add_patch(self.heat_cover)
        self._draw_trigger_lines(ax, zorder=4)
        self.heat_cursor = ax.axvline(t0, color='k', lw=1, zorder=5)
        self._frame_artists += [self.heat_cover, self.heat_cursor]
        step = max(1, int(np.ceil(len(ids) / 16)))
        ax.set_yticks(range(0, len(ids), step))
        ax.set_yticklabels([str(ids[k]) for k in range(0, len(ids), step)])
        for tick, k in zip(ax.get_yticklabels(), range(0, len(ids), step)):
            if ids[k] in s.shown_selected:
                tick.set_color(s.color_of(ids[k]))
                tick.set_fontweight('bold')
        ax.set_ylabel('ROI', fontsize=FS_LABEL, labelpad=2)
        what = 'all eligible' if s.display == 'all' else 'selected'
        ax.set_title(f"{what}, n = {len(ids)}", fontsize=FS_SMALL, loc='right', color='0.35')

    def _trace_offsets(self, signal) -> tuple:
        """Selected ROIs shown in the trace panels and their stacking offsets (same in both panels)."""
        s = self.state
        sel = [r for r in s.shown_selected if r in signal.traces]
        if len(sel) < 2:
            return sel, [0.] * len(sel)
        spans = [np.subtract(*self._cache.ranges[r][::-1]) for r in sel]
        spacing = 1.15 * max(spans)
        return sel, [-k * spacing for k in range(len(sel))]

    def _build_traces(self) -> None:
        s, ax = self.state, self.ax_trace
        pres = s.pres
        t0, t1 = float(pres.frame_times[0]), pres.t_end
        ax.set_xlim(t0, t1)
        ax.set_xlabel('Recording time (s)', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_title('Response trace', fontsize=FS_TITLE, loc='left', pad=1.9 * FS_SMALL)
        self._trace_lines, self._trace_markers = [], []
        self._y_range = []
        xform = blended_transform_factory(ax.transData, ax.transAxes)
        self.trial_window = Rectangle((t0, 0), 0, 1, transform=xform, facecolor='0.5', alpha=0.1, edgecolor='none',
                                      zorder=0.5, visible=False)
        ax.add_patch(self.trial_window)
        self.trace_cursor = Rectangle((t0, 0), pres.frame_dt, 1, transform=xform, facecolor='tab:red', alpha=0.25,
                                      edgecolor='tab:red', lw=1, zorder=2)
        ax.add_patch(self.trace_cursor)
        self._frame_artists += [self.trial_window, self.trace_cursor]
        self._draw_trigger_lines(ax, zorder=1)
        self._draw_trigger_labels(ax)

        signal = pres.signals.get(s.signal) if s.signal else None
        if signal is None:
            ax.text(0.5, 0.5, 'no traces for this presentation', ha='center', va='center', transform=ax.transAxes,
                    fontsize=FS_LABEL)
            ax.set_yticks([])
            return
        ax.set_ylabel(signal.units, fontsize=FS_LABEL)
        sel, offsets = self._trace_offsets(signal)
        if sel:
            label_xform = blended_transform_factory(ax.transAxes, ax.transData)
            for r, off in zip(sel, offsets):
                tr = signal.traces[r]
                line, = ax.plot([], [], color=s.color_of(r), lw=1.0, zorder=3)
                marker, = ax.plot([], [], 'o', color=s.color_of(r), ms=4, zorder=4)
                self._trace_lines.append((line, tr.times, tr.values + off))
                self._trace_markers.append((marker, tr, off))
                ax.text(0.004, off + np.nanmedian(tr.values), str(r), color=s.color_of(r), fontsize=FS_TICK,
                        weight='bold', va='center', transform=label_xform, zorder=5)
                self._y_range += [np.nanmin(tr.values) + off, np.nanmax(tr.values) + off]
            status = f"ROI {', '.join(map(str, sel))}" + (' (stacked)' if len(sel) > 1 else '')
        else:
            ids = [r for r in s.eligible if r in self._cache.rows]
            if not ids:
                ax.text(0.5, 0.5, 'no eligible ROI with a trace', ha='center', va='center', transform=ax.transAxes,
                        fontsize=FS_LABEL)
                return
            mean = self._cache.mean(ids)
            line, = ax.plot([], [], color=POPULATION_COLOR, lw=1.0, zorder=3)
            self._trace_lines.append((line, pres.frame_times + pres.frame_dt / 2, mean))
            self._y_range += [np.nanmin(mean), np.nanmax(mean)]
            status = f"mean of {len(ids)} eligible ROIs"
        ax.set_title(status, fontsize=FS_SMALL, loc='right', color='0.35', pad=1.9 * FS_SMALL)
        for line, _, _ in self._trace_lines:
            self._frame_artists.append(line)
        for marker, _, _ in self._trace_markers:
            self._frame_artists.append(marker)

    def _build_trials(self) -> None:
        s, ax = self.state, self.ax_trial
        ax.set_title('Trial mean ± SD', fontsize=FS_TITLE, loc='left', pad=1.9 * FS_SMALL)
        ax.set_xlabel('Time from onset (s)', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK, labelleft=False)
        self._trial_series, self.trial_cursor, self.trial_label = [], None, None
        self._trial_label_key = None
        trials = self._trials
        signal = s.pres.signals.get(s.signal) if s.signal else None
        if not trials.available or signal is None:
            ax.text(0.5, 0.5, trials.unavailable_reason(), ha='center', va='center', transform=ax.transAxes,
                    fontsize=FS_LABEL, color='0.3', wrap=True)
            self._finish_ylim()
            return
        sel, offsets = self._trace_offsets(signal)
        problems = []
        if sel:
            items = [(trials.roi_summary(r), off, s.color_of(r)) for r, off in zip(sel, offsets)]
            problems = [f"ROI {r}: {trials.error(r) or 'no valid repeat'}" for r, (summ, _, _) in zip(sel, items)
                        if summ is None]
        else:
            ids = [r for r in s.eligible if r in signal.traces]
            items = [(trials.population_summary(ids), 0., POPULATION_COLOR)] if ids else []
        t_lo, t_hi = [], []
        for summary, off, color in items:
            if summary is None:
                continue
            per_condition = []
            for c in summary.conditions:
                if c is None:
                    per_condition.append(None)
                    continue
                lo = c.mean - c.sd + off if c.sd is not None else None
                hi = c.mean + c.sd + off if c.sd is not None else None
                per_condition.append((c.rel_times, c.mean + off, lo, hi, c.n))
                t_lo.append(c.rel_times[0])
                t_hi.append(c.rel_times[-1])
            band = PolyCollection([], facecolors=[color], alpha=0.25, edgecolors='none', zorder=2)
            ax.add_collection(band)
            line, = ax.plot([], [], color=color, lw=1.2, zorder=3)
            self._trial_series.append(_TrialSeries(line, band, per_condition))
            vlo, vhi = summary.value_range(off)
            self._y_range += [vlo, vhi]
        if not self._trial_series:
            msg = '; '.join(problems) if problems else 'no valid repeats'
            ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes, fontsize=FS_SMALL, wrap=True)
            self._finish_ylim()
            return
        dt = s.pres.frame_dt
        ax.set_xlim(min(t_lo), max(t_hi) + dt)
        layout = trials.layout
        ntr = int(s.pres.stimulus.ntrigger_rep)
        if ntr > 1:  # triggers inside a repetition (e.g. chirp phases), at their mean time from onset
            trig = s.pres.triggers
            n_full = trig.size // ntr
            rel = (trig[:n_full * ntr].reshape(n_full, ntr) - trig[:n_full * ntr:ntr, None]).mean(axis=0)
            ax.vlines(rel, 0, 1, transform=blended_transform_factory(ax.transData, ax.transAxes), colors='0.45',
                      linewidths=0.8, linestyles='--', alpha=0.8, zorder=1)
        else:
            ax.axvline(0, color='0.45', lw=0.8, ls='--', zorder=1)
        xform = blended_transform_factory(ax.transData, ax.transAxes)
        self.trial_cursor = Rectangle((0, 0), dt, 1, transform=xform, facecolor='tab:red', alpha=0.25,
                                      edgecolor='tab:red', lw=1, zorder=4, visible=False)
        ax.add_patch(self.trial_cursor)
        self.trial_label = TextStamp(ax, (0.99, 0.97))
        self._frame_artists += [srs.band for srs in self._trial_series] + [srs.line for srs in self._trial_series]
        self._frame_artists += [self.trial_cursor, self.trial_label.artist]
        self._condition_names = layout.condition_names
        if problems:
            ax.text(0.01, 0.02, '; '.join(problems), transform=ax.transAxes, fontsize=FS_SMALL, color='#a00')
        self._finish_ylim()

    def _finish_ylim(self) -> None:
        vals = np.asarray([v for v in self._y_range if np.isfinite(v)])
        if vals.size:
            lo, hi = vals.min(), vals.max()
            pad = 0.05 * (hi - lo) if hi > lo else 1.
            self.ax_trace.set_ylim(lo - pad, hi + pad)

    # ----------------------------------------------------------------- triggers
    def _trigger_marks_for(self, pres: PresentationData) -> dict:
        """Trigger times, repeat-start flags and labels of a presentation; computed once per presentation."""
        if self._trigger_marks.get('key') != id(pres):
            trig = pres.triggers
            stim = pres.stimulus
            strong = np.zeros(trig.size, dtype=bool)
            if stim is not None and stim.isrepeated and trig.size:
                conds = stim.conditions
                period = len(conds) if conds else max(1, int(stim.ntrigger_rep))
                strong[::period] = True
            labels = list(pres.trigger_labels) if pres.trigger_labels is not None else [''] * trig.size
            self._trigger_marks = dict(key=id(pres), times=trig, strong=strong, labels=labels)
        return self._trigger_marks

    def _draw_trigger_lines(self, ax, zorder: float) -> None:
        marks = self._trigger_marks_for(self.state.pres)
        trig, strong = marks['times'], marks['strong']
        if trig.size == 0:
            return
        xform = blended_transform_factory(ax.transData, ax.transAxes)
        many = trig.size > MAX_TRIGGER_LABELS
        weak = trig[~strong]
        if weak.size:
            ax.vlines(weak, 0, 1, transform=xform, colors='0.45', linewidths=0.4 if many else 0.6, linestyles='--',
                      alpha=0.4 if many else 0.6, zorder=zorder)
        if strong.any():
            ax.vlines(trig[strong], 0, 1, transform=xform, colors='0.3', linewidths=1.0, linestyles='--',
                      alpha=0.9, zorder=zorder)

    def _draw_trigger_labels(self, ax) -> None:
        """Horizontal labels just above the trace panel, thinned so that neighbours never overlap."""
        marks = self._trigger_marks_for(self.state.pres)
        trig, labels = marks['times'], marks['labels']
        idx = [k for k, lab in enumerate(labels) if lab]
        if not idx or trig.size > MAX_TRIGGER_LABELS:
            return
        x0, x1 = ax.get_xlim()
        width_px = ax.get_position().width * self.fig.get_size_inches()[0] * self.fig.dpi
        px_per_s = width_px / (x1 - x0)
        char_px = 0.62 * FS_SMALL * self.fig.dpi / 72
        need = max(len(labels[k]) for k in idx) * char_px + 6
        gaps = np.diff(trig[idx]) * px_per_s if len(idx) > 1 else np.array([np.inf])
        stride = max(1, int(np.ceil(need / max(gaps.min(), 1e-9))))
        xform = blended_transform_factory(ax.transData, ax.transAxes)
        for k in idx[::stride]:
            ax.text(trig[k], 1.0, labels[k], transform=xform, fontsize=FS_SMALL, ha='left', va='bottom', color='0.3',
                    clip_on=False)

    # ---------------------------------------------------------------- per frame
    def _update_frame(self) -> None:
        s = self.state
        pres = s.pres
        i = int(np.clip(s.frame, 0, pres.n_frames - 1))
        t0, t1 = pres.frame_interval(i)
        if s.view != 'highres' or s.highres is None:
            self.movie_image.set_data(pres.frames[i])
        if self.time_text is not None:
            trig = pres.triggers
            rel = f"   (trigger 1 {t0 - trig[0]:+.2f} s)" if trig.size else ''
            self.time_text.set_text(f"t = {t0:.2f} s   frame {i + 1}/{pres.n_frames}{rel}")
        if self.heat_cover is not None:
            self.heat_cover.set_x(t1)
            self.heat_cover.set_width(max(0., pres.t_end - t1))
            self.heat_cursor.set_xdata([t0, t0])
        self.trace_cursor.set_x(t0)
        self.trace_cursor.set_width(t1 - t0)
        for line, times, values in self._trace_lines:
            n = n_samples_before(times, t1)
            line.set_data(times[:n], values[:n])
        for marker, tr, off in self._trace_markers:
            idx = tr.samples_in(t0, t1)
            marker.set_data(tr.times[idx], tr.values[idx] + off)
        self._update_trials(t0, t1)
        if self.overlay_image is not None:
            self._update_overlay(i)

    def _update_trials(self, t0: float, t1: float) -> None:
        layout = self._trials.layout if self._trials is not None else None
        if layout is None or not self._trial_series:
            self.trial_window.set_visible(False)
            return
        k = layout.trial_before(t1)
        self._current_trial = k
        cond = int(layout.condition[k]) if k >= 0 else int(layout.condition[0])
        elapsed = t1 - layout.onsets[k] if k >= 0 else -np.inf
        ns = set()
        for srs in self._trial_series:
            entry = srs.per_condition[cond] if cond < len(srs.per_condition) else None
            if entry is None:
                srs.line.set_data([], [])
                srs.band.set_verts([])
                continue
            x, y, lo, hi, n_rep = entry
            ns.add(n_rep)
            n = n_samples_before(x, elapsed) if k >= 0 else 0
            srs.line.set_data(x[:n], y[:n])
            if lo is not None and n > 1:
                srs.band.set_verts([np.concatenate([np.column_stack([x[:n], lo[:n]]),
                                                    np.column_stack([x[n - 1::-1], hi[n - 1::-1]])])])
            else:
                srs.band.set_verts([])
        if k >= 0:
            onset = layout.onsets[k]
            self.trial_cursor.set_visible(True)
            self.trial_cursor.set_x(t0 - onset)
            self.trial_cursor.set_width(t1 - t0)
            end = layout.onsets[k + 1] if k + 1 < layout.onsets.size else onset + self.ax_trial.get_xlim()[1]
            self.trial_window.set_visible(True)
            self.trial_window.set_x(onset)
            self.trial_window.set_width(end - onset)
        else:
            self.trial_cursor.set_visible(False)
            self.trial_window.set_visible(False)
        label_key = (cond, k >= 0, tuple(sorted(ns)), self.fig.dpi)
        if label_key != self._trial_label_key:
            self._trial_label_key = label_key
            name = self._condition_names[cond]
            n_text = 'no valid repeat' if not ns else \
                'n = ' + ('–'.join(str(v) for v in (min(ns), max(ns))) if len(ns) > 1 else str(min(ns)))
            rep = ' repeat' if ns == {1} else ' repeats'
            parts = ([name] if name else []) + [n_text + (rep if ns else '')]
            if ns == {1}:
                parts.append('no SD')
            if k < 0:
                parts.append('before 1st trial')
            self.trial_label.set_text('  ·  '.join(parts))

    def _update_overlay(self, i: int) -> None:
        ids = self._overlay_ids
        if not ids:
            return
        vals = np.array([self._cache.rows[r][i] for r in ids])
        lo = np.array([self._cache.ranges[r][0] for r in ids])
        hi = np.array([self._cache.ranges[r][1] for r in ids])
        colors = matplotlib.colormaps[OVERLAY_CMAP](np.clip((vals - lo) / (hi - lo), 0, 1))
        colors[:, 3] = np.where(np.isfinite(vals), 0.7, 0.)
        rgba = np.zeros(self._overlay_index.shape + (4,))
        inside = self._overlay_index >= 0
        rgba[inside] = colors[self._overlay_index[inside]]
        self.overlay_image.set_data(rgba)
