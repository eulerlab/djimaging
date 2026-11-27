"""SYNTHETIC test fixture for CellViz (not real data).

Covers what the local recordings do not: non-consecutive ROI IDs, a mask that is shifted between two presentations of
the same stimulus (two conditions), traces whose time origin differs from the movie, known triggers and known QI
values (including missing ones), a HighRes image with a known shift, and a non-square frame (to catch x/y
transposes).

Encoding that makes every value traceable: in presentation ``p`` (offset ``P = 10000 * p``) all pixels of ROI ``k`` in
frame ``i`` have the value ``P + 100 * k + i``; background pixels are ``P``. The trace of ROI ``k`` is the ROI mean,
i.e. sample ``j`` of ROI ``k`` equals ``P + 100 * k + (frame of that sample)``.
"""
from __future__ import annotations

import numpy as np

from djimaging.cellviz.data import (
    HighResImage, PresentationData, PresentationInfo, QualityValues, RoiLabels, Signal, StimulusInfo, Trace)

NY, NX = 16, 20          # display shape (rows = y, cols = x), deliberately not square
N_FRAMES = 120
FRAME_DT = 0.05
ROI_IDS = (2, 5, 11)     # non-consecutive
TRIGGERS = np.array([1.0, 1.5, 3.0, 3.5])
HR_SCALE = 4
HR_SHIFT = (0, -8)       # (dy, dx) in HighRes pixels (= -2 functional pixels in x)
LINE_OFFSETS = {2: 0.004, 5: 0.021, 11: 0.037}  # ROI trace time offsets within a frame (line scan)
LATE_START = {11: 10}    # ROI 11 in 'drug' starts 10 frames later (clipped trace, different time origin)
# chirp: 2 triggers per repeat -> repeats start at 1.0 and 3.0 s; moving bar: one trial per trigger, directions
# 0, 180, 0, 180 (two repeats of each direction)
CHIRP_STIM = StimulusInfo(isrepeated=True, ntrigger_rep=2, pad_trace=True)
MB_STIM = StimulusInfo(isrepeated=True, ntrigger_rep=1, pad_trace=True,
                       trial_info=[dict(name=0, ntrigger=1), dict(name=180, ntrigger=1)])


def roi_labels_xy(shift_x: int = 0) -> np.ndarray:
    """Igor-format mask in djimaging ``[x, y]`` orientation."""
    mask = np.ones((NX, NY), dtype=np.int32)
    mask[2 + shift_x:5 + shift_x, 3:6] = -2
    mask[9 + shift_x:12 + shift_x, 8:12] = -5
    mask[14 + shift_x:17 + shift_x, 2:4] = -11
    return mask


def _movie(rois: RoiLabels, offset: float) -> np.ndarray:
    frames = np.full((N_FRAMES, NY, NX), offset, dtype=np.float32)
    for k in rois.roi_ids:
        frames[:, rois.pixels(k)] = (offset + 100 * k + np.arange(N_FRAMES))[:, None]
    return frames


def make_presentation(name: str, cond1: str, offset: float, shift_x: int, quality: dict, stim: str = 'chirp',
                      missing_trace: tuple = ()) -> PresentationData:
    rois = RoiLabels.from_igor_xy(roi_labels_xy(shift_x))
    frames = _movie(rois, offset)
    frame_times = np.arange(N_FRAMES) * FRAME_DT
    traces = {}
    for k in ROI_IDS:
        if k in missing_trace:
            continue
        start = LATE_START.get(k, 0) if cond1 == 'drug' else 0
        idx = np.arange(start, N_FRAMES)
        traces[k] = Trace(frame_times[idx] + LINE_OFFSETS[k], offset + 100 * k + idx)
    family = 'movingbar' if stim == 'movingbar' else 'chirp'
    metric = 'd_qi' if stim == 'movingbar' else 'qidx'
    table = 'OsDsIndexes' if stim == 'movingbar' else 'ChirpQI'
    info = PresentationInfo(name=name, key=dict(field='SYN', stim_name=stim, cond1=cond1), stim_name=stim,
                            stim_family=family, channels=('wDataCh0', 'wDataCh1'))
    return PresentationData(
        info=info, channel='wDataCh0', frames=frames, frame_times=frame_times, frame_dt=FRAME_DT, rois=rois,
        signals={'raw': Signal('raw', 'a.u.', traces, 'synthetic')}, triggers=TRIGGERS,
        trigger_labels=['0°', '180°', '0°', '180°'] if stim == 'movingbar' else ['rep 1', '', 'rep 2', ''],
        stimulus=MB_STIM if stim == 'movingbar' else CHIRP_STIM,
        quality=QualityValues(table, metric, quality, 'synthetic', reference=0.5),
        pixel_size_um=2.0, npixartifact=0,
        meta=dict(absx=10.0, absy=20.0, scan_type='xy', angle_deg=0.0), source='SYNTHETIC')


def make_highres(pres: PresentationData) -> HighResImage:
    """HighRes = 4x upsampled functional median, shifted by ``HR_SHIFT``: functional pixel (y, x) is found at HighRes
    pixels ``[y*4 + dy, ...) x [x*4 + dx, ...)``."""
    func = np.median(pres.frames, axis=0)
    up = np.kron(func, np.ones((HR_SCALE, HR_SCALE)))
    hr = np.full(up.shape, float(np.median(func)))
    dy, dx = HR_SHIFT
    hr[max(0, dy):up.shape[0] + min(0, dy), max(0, dx):up.shape[1] + min(0, dx)] = \
        up[max(0, -dy):up.shape[0] - max(0, dy), max(0, -dx):up.shape[1] - max(0, dx)]
    return HighResImage(hr, 'wDataCh0', pres.pixel_size_um / HR_SCALE, dict(pres.meta), source='SYNTHETIC')


class SyntheticFieldSource:
    """Field source protocol (``list_presentations``, ``load_presentation``, ``load_highres``) on synthetic data."""

    label = 'SYNTHETIC field'
    notes = ['SYNTHETIC fixture']

    def __init__(self, with_highres: bool = True, data_roots: tuple = ()):
        self.with_highres = with_highres
        self.data_roots = data_roots
        self._pres = {
            'chirp [control]': make_presentation('chirp [control]', 'control', 0., 0, {2: 0.5, 5: 0.2}),
            'chirp [drug]': make_presentation('chirp [drug]', 'drug', 10000., 1, {2: 0.1, 5: 0.6, 11: 0.4}),
            'movingbar': make_presentation('movingbar', 'control', 20000., 0, {2: 0.7, 5: 0.65, 11: 0.2},
                                           stim='movingbar', missing_trace=(5,)),
        }
        self.loads = []

    def list_presentations(self):
        return [p.info for p in self._pres.values()]

    def load_presentation(self, name, channel='wDataCh0'):
        self.loads.append((name, channel))
        return self._pres[name]

    def load_highres(self, channel='wDataCh0'):
        if not self.with_highres:
            return None, 'no HighRes recording for this field (SYNTHETIC)'
        return make_highres(self._pres['chirp [control]']), ''
