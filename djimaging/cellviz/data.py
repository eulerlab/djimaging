"""Source-independent data model of CellViz.

Loaders (``offline.LocalFieldSource`` for local ScanM files, ``dj_adapter.DataJointFieldSource`` for pipeline results)
convert what they read into these objects; the renderer and the viewer only see these objects.

Conventions (adapted from the CellViz prototype)
------------------------------------------------
- djimaging stores images, stacks and ROI masks indexed ``[x, y(, t)]`` and plots them with
  ``imshow(arr.T, origin='lower')``. Here everything is stored in that display orientation: images ``[y, x]``,
  movies ``(n_frames, y, x)``, drawn with ``origin='lower'``. ``to_display`` does this conversion once, at loading.
- ROI masks keep the labels of the source (djimaging "Igor" format: ROI pixels ``-roi_id``, background ``1``/``0``).
  ROI IDs only come from the explicit ``label_to_roi_id`` mapping. IDs are never renumbered, so non-consecutive IDs
  and the IDs stored in ``Roi`` / ``Traces`` stay identical.
- Times are seconds on the clock of the presentation; movie frames, every trace and the triggers carry their own
  timestamps (see ``timing``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from djimaging.cellviz.timing import check_increasing, samples_in_interval
from djimaging.utils.mask_format_utils import is_igor_format, to_igor_format


def to_display(arr_xy: np.ndarray) -> np.ndarray:
    """djimaging ``[x, y]`` image or ``[x, y, t]`` stack -> display ``[y, x]`` image or ``(t, y, x)`` movie."""
    arr_xy = np.asarray(arr_xy)
    if arr_xy.ndim == 2:
        return arr_xy.T
    if arr_xy.ndim == 3:
        return np.transpose(arr_xy, (2, 1, 0))
    raise ValueError(f"expected a 2-D image or 3-D stack, got shape {arr_xy.shape}")


@dataclass(frozen=True)
class RoiLabels:
    """ROI label image in display orientation ``[y, x]`` plus the explicit label -> ROI ID mapping."""
    labels: np.ndarray
    label_to_roi_id: Mapping[int, int]
    background_labels: Sequence[int]

    def __post_init__(self):
        labels = np.asarray(self.labels)
        if labels.ndim != 2:
            raise ValueError(f"labels must be 2-D, got shape {labels.shape}")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"labels must be an integer array, got dtype {labels.dtype}")
        mapping = {int(k): int(v) for k, v in self.label_to_roi_id.items()}
        background = tuple(int(b) for b in self.background_labels)
        if set(mapping) & set(background):
            raise ValueError(f"labels {sorted(set(mapping) & set(background))} are both ROI and background")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("label_to_roi_id must be one-to-one (duplicate ROI IDs)")
        present = set(np.unique(labels).tolist())
        unexplained = present - set(mapping) - set(background)
        if unexplained:
            raise ValueError(f"mask contains labels {sorted(unexplained)} that are neither mapped nor background")
        missing = set(mapping) - present
        if missing:
            raise ValueError(f"mapped labels {sorted(missing)} do not occur in the mask")
        object.__setattr__(self, 'labels', labels)
        object.__setattr__(self, 'label_to_roi_id', mapping)
        object.__setattr__(self, 'background_labels', background)
        object.__setattr__(self, '_roi_id_to_label', {v: k for k, v in mapping.items()})

    @classmethod
    def from_igor_xy(cls, roi_mask_xy: np.ndarray) -> "RoiLabels":
        """From a djimaging Igor-format mask ``[x, y]`` (as stored in ``RoiMask``): ROI ID = ``-label``."""
        mask = np.asarray(roi_mask_xy)
        if not is_igor_format(mask):
            raise ValueError(f"ROI mask is not in Igor format: values {np.unique(mask)[:10]}")
        labels = mask.astype(np.int64).T
        values = np.unique(labels)
        return cls(labels=labels, label_to_roi_id={int(v): int(-v) for v in values if v < 0},
                   background_labels=tuple(int(v) for v in values if v >= 0))

    @classmethod
    def from_python_xy(cls, roi_mask_xy: np.ndarray) -> "RoiLabels":
        """From a python-format mask file ``[x, y]`` (background 0, ROI k > 0), converted exactly like
        ``RoiMask._load_presentation_roi_mask`` does (``to_igor_format``), so that ROI ID ``k`` stays ``k``."""
        return cls.from_igor_xy(to_igor_format(np.asarray(roi_mask_xy).astype(np.int32)))

    @property
    def shape(self) -> tuple:
        return self.labels.shape

    @property
    def roi_ids(self) -> np.ndarray:
        return np.array(sorted(self.label_to_roi_id.values()), dtype=int)

    def label_of(self, roi_id: int) -> int:
        return self._roi_id_to_label[int(roi_id)]

    def pixels(self, roi_id: int) -> np.ndarray:
        """Boolean ``[y, x]`` mask of ``roi_id``."""
        return self.labels == self.label_of(roi_id)

    def roi_id_at(self, row: int, col: int) -> Optional[int]:
        """ROI ID at display pixel ``(row=y, col=x)``, or None for background / outside."""
        n_rows, n_cols = self.shape
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            return None
        return self.label_to_roi_id.get(int(self.labels[row, col]))


@dataclass(frozen=True)
class Trace:
    """One trace of one ROI with its own sample timestamps (seconds, presentation clock)."""
    times: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        times = check_increasing(self.times, 'trace times')
        values = np.asarray(self.values, dtype=float)
        if values.shape != times.shape:
            raise ValueError(f"trace values shape {values.shape} != times shape {times.shape}")
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'values', values)

    def samples_in(self, t_start: float, t_stop: float) -> np.ndarray:
        return samples_in_interval(self.times, t_start, t_stop)


@dataclass(frozen=True)
class Signal:
    """A set of traces of one kind (e.g. raw ``Traces.trace`` or ``PreprocessTraces.pp_trace``), keyed by ROI ID."""
    name: str
    units: str
    traces: Mapping[int, Trace]
    description: str = ''


@dataclass(frozen=True)
class QualityValues:
    """Response quality of the ROIs of one presentation (one value per ROI ID).

    ``values`` holds the metric for ROIs that have an entry; ROIs without an entry are *missing* (no value), which is
    different from a value below the threshold. NaN values are also treated as missing.
    """
    table: str                    # e.g. 'ChirpQI', 'OsDsIndexes'
    metric: str                   # e.g. 'qidx', 'd_qi'
    values: Mapping[int, float]
    source: str                   # 'database' or a description of the offline recomputation
    reference: Optional[float] = None  # e.g. min_qidx = 1 / n_repetitions, shown for orientation only
    extra: Mapping[str, Mapping[int, float]] = field(default_factory=dict)  # e.g. ds_index, os_index

    def value(self, roi_id: int) -> float:
        v = self.values.get(int(roi_id), np.nan)
        return float(v) if v is not None else np.nan


@dataclass(frozen=True)
class StimulusInfo:
    """What the pipeline needs to cut a presentation into repeats (``Stimulus`` + ``Snippets`` settings).

    Repeats are cut exactly as ``SnippetsTemplate.make`` does: ``split_trace_by_reps(trace, times, triggers,
    ntrigger_rep, delay=trigger_delay, allow_drop_last=True, pad_trace=pad_trace)``. With ``ntrigger_rep == 1`` and a
    ``trial_info`` (moving bar), every trigger starts one trial whose condition is ``trial_info[k % len(trial_info)]``
    (the order ``preprocess_mb_snippets`` / ``get_dir_idx`` assume).
    """
    isrepeated: bool
    ntrigger_rep: int
    trial_info: Optional[Sequence[Mapping]] = None
    trigger_delay: float = 0.          # Stimulus.stim_dict['trigger_delay'], as used by Snippets
    pad_trace: bool = False            # Snippets._pad_trace (SnippetsTemplate default False, example schema True)

    @property
    def conditions(self) -> Optional[list]:
        """Names of the per-trigger conditions (e.g. moving-bar directions), or None for whole repetitions."""
        if self.trial_info and int(self.ntrigger_rep) == 1:
            return [t['name'] for t in self.trial_info]
        return None


@dataclass(frozen=True)
class PresentationInfo:
    """Entry of the presentation selector: everything that is known without loading the movie."""
    name: str                     # unique, shown in the selector
    key: Mapping                  # full key of the presentation (DataJoint primary key, or file based offline)
    stim_name: Optional[str]      # None if the file does not match any known stimulus
    stim_family: str = ''
    channels: Sequence[str] = ('wDataCh0',)
    has_mask: bool = True
    notes: Sequence[str] = ()


@dataclass(frozen=True)
class PresentationData:
    """Everything CellViz shows for one presentation and one channel."""
    info: PresentationInfo
    channel: str
    frames: np.ndarray            # (n_frames, y, x)
    frame_times: np.ndarray       # (n_frames,), frame start times
    frame_dt: float
    rois: Optional[RoiLabels]
    signals: Mapping[str, Signal]
    triggers: np.ndarray
    trigger_labels: Optional[Sequence[str]] = None  # one label per trigger, e.g. 'rep 1' or '45°', or None
    stimulus: Optional[StimulusInfo] = None  # repeat structure for the trial average; None: no trial panel
    quality: Optional[QualityValues] = None
    quality_unavailable: str = ''  # why there is no quality metric for this presentation
    roi_notes: Mapping[int, str] = field(default_factory=dict)  # e.g. ROIs without a trace and why
    mask_info: Mapping = field(default_factory=dict)  # as_field_mask, shift_dx, shift_dy, roi_mask_file
    pixel_size_um: float = np.nan
    npixartifact: int = 0
    meta: Mapping = field(default_factory=dict)  # absx, absy, nxpix, nypix, zoom, angle_deg, scan_type
    notes: Sequence[str] = ()
    source: str = ''

    def __post_init__(self):
        frames = np.asarray(self.frames)
        if frames.ndim != 3:
            raise ValueError(f"frames must be (n_frames, y, x), got shape {frames.shape}")
        frame_times = check_increasing(self.frame_times, 'frame_times')
        if frame_times.size != frames.shape[0]:
            raise ValueError(f"{frame_times.size} frame_times for {frames.shape[0]} frames")
        if not self.frame_dt > 0:
            raise ValueError(f"frame_dt must be positive, got {self.frame_dt}")
        if self.rois is not None and self.rois.shape != frames.shape[1:]:
            raise ValueError(f"ROI mask shape {self.rois.shape} != frame shape {frames.shape[1:]}; "
                             f"mask and movie must be in the same (presentation) coordinate system")
        triggers = np.asarray(self.triggers, dtype=float).ravel()
        if self.trigger_labels is not None and len(self.trigger_labels) != triggers.size:
            raise ValueError(f"{len(self.trigger_labels)} trigger labels for {triggers.size} triggers")

        roi_ids = set() if self.rois is None else set(self.rois.roi_ids.tolist())
        t_lo, t_hi = frame_times[0], frame_times[-1] + self.frame_dt
        for signal in self.signals.values():
            unknown = set(int(r) for r in signal.traces) - roi_ids
            if unknown:
                raise ValueError(f"signal {signal.name!r} has traces for ROI IDs {sorted(unknown)} not in the mask")
            for roi_id, trace in signal.traces.items():
                if trace.times[-1] < t_lo or trace.times[0] >= t_hi:
                    raise ValueError(f"{signal.name} trace of ROI {roi_id} ({trace.times[0]:.3f}-{trace.times[-1]:.3f} s) "
                                     f"does not overlap the movie ({t_lo:.3f}-{t_hi:.3f} s): clock mismatch?")
        if self.quality is not None:
            unknown = set(int(r) for r in self.quality.values) - roi_ids
            if unknown:
                raise ValueError(f"quality values for ROI IDs {sorted(unknown)} not in the mask")
        object.__setattr__(self, 'frames', frames)
        object.__setattr__(self, 'frame_times', frame_times)
        object.__setattr__(self, 'frame_dt', float(self.frame_dt))
        object.__setattr__(self, 'triggers', triggers)

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def roi_ids(self) -> np.ndarray:
        return np.zeros(0, dtype=int) if self.rois is None else self.rois.roi_ids

    def frame_interval(self, index: int) -> tuple:
        t0 = float(self.frame_times[index])
        return t0, t0 + self.frame_dt

    @property
    def t_end(self) -> float:
        return float(self.frame_times[-1] + self.frame_dt)


@dataclass(frozen=True)
class HighResImage:
    """Static high-resolution image of a field in display orientation ``[y, x]`` with its scan geometry."""
    image: np.ndarray
    channel: str
    pixel_size_um: float
    meta: Mapping = field(default_factory=dict)  # absx, absy, nxpix, nypix, zoom, angle_deg, scan_type
    source: str = ''

    def __post_init__(self):
        image = np.asarray(self.image, dtype=float)
        if image.ndim != 2:
            raise ValueError(f"high-res image must be 2-D, got shape {image.shape}")
        object.__setattr__(self, 'image', image)
