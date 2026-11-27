"""Data model, timing, quality filter, registration and renderer of CellViz (SYNTHETIC data)."""
from dataclasses import replace

import numpy as np
import pytest

from djimaging.cellviz.data import PresentationData, RoiLabels, Signal, Trace, to_display
from djimaging.cellviz.quality import classify, quality_family, unsupported_reason
from djimaging.cellviz.registration import (
    Registration, check_metadata, estimate_shift, map_labels_to_highres, register_highres, rois_outside)
from djimaging.cellviz.render import CellVizFigure, ViewState, outline_segments
from djimaging.cellviz.timing import frame_index_at, frame_range, n_samples_before
from djimaging.utils.mask_format_utils import to_python_format
from tests.cellviz.synthetic import (
    FRAME_DT, HR_SCALE, HR_SHIFT, LINE_OFFSETS, NX, NY, ROI_IDS, SyntheticFieldSource, make_highres, roi_labels_xy)


@pytest.fixture
def source():
    return SyntheticFieldSource()


# ------------------------------------------------------------------ data model
def test_roi_labels_keep_ids_and_orientation():
    mask_xy = roi_labels_xy()
    rois = RoiLabels.from_igor_xy(mask_xy)
    assert rois.roi_ids.tolist() == list(ROI_IDS)  # non-consecutive IDs are not renumbered
    assert rois.shape == (NY, NX)
    for x in range(NX):
        for y in range(NY):
            expected = -mask_xy[x, y] if mask_xy[x, y] < 0 else None
            assert rois.roi_id_at(row=y, col=x) == expected


def test_python_format_mask_is_converted_like_roimask():
    mask_python_xy = to_python_format(roi_labels_xy())
    rois = RoiLabels.from_python_xy(mask_python_xy)
    assert rois.roi_ids.tolist() == list(ROI_IDS)
    assert set(rois.background_labels) == {1}
    np.testing.assert_array_equal(rois.labels, to_display(roi_labels_xy()))


def test_unknown_label_rejected():
    labels = np.zeros((4, 4), dtype=int)
    labels[0, 0] = 3
    with pytest.raises(ValueError, match='neither mapped nor background'):
        RoiLabels(labels, {}, (0,))


def test_presentation_rejects_mixups(source):
    pres = source.load_presentation('chirp [control]')
    bad_traces = dict(pres.signals['raw'].traces)
    bad_traces[99] = bad_traces[2]
    with pytest.raises(ValueError, match='not in the mask'):
        replace(pres, signals={'raw': Signal('raw', 'a.u.', bad_traces)})
    shifted = {2: Trace(pres.signals['raw'].traces[2].times + 1000., pres.signals['raw'].traces[2].values)}
    with pytest.raises(ValueError, match='clock mismatch'):
        replace(pres, signals={'raw': Signal('raw', 'a.u.', shifted)})
    with pytest.raises(ValueError, match='same \\(presentation\\) coordinate system'):
        replace(pres, rois=RoiLabels.from_igor_xy(np.ones((NY, NX), dtype=int) - 2 * np.eye(NY, NX, dtype=int)))


# ------------------------------------------------------------------ timing
def test_timing_lookups():
    frame_times = np.arange(10) * 0.1
    assert frame_index_at(frame_times, 0.25) == 2
    assert frame_index_at(frame_times, 0.2) == 2
    assert frame_index_at(frame_times, -5) == 0 and frame_index_at(frame_times, 50) == 9
    assert list(frame_range(frame_times, 0.1, 0.25, 0.55)) == [2, 3, 4, 5]
    assert list(frame_range(frame_times, 0.1, None, None)) == list(range(10))
    assert n_samples_before(np.array([0.0, 0.1, 0.2]), 0.1) == 1


# ------------------------------------------------------------------ quality
def test_quality_classes_distinguish_missing_from_below(source):
    q = source.load_presentation('chirp [control]').quality  # {2: 0.5, 5: 0.2}, 11 missing
    off = classify(ROI_IDS, q, None)
    assert off.eligible == ROI_IDS and off.missing == (11,) and not off.active
    on = classify(ROI_IDS, q, 0.3)
    assert (on.eligible, on.below, on.missing) == ((2,), (5,), (11,))
    assert classify(ROI_IDS, q, 0.5).eligible == (2,)  # QI >= threshold
    assert '1 eligible, 1 below threshold, 1 missing value' == on.summary()


def test_quality_family():
    assert quality_family('gChirp', 'chirp') == 'chirp' and quality_family('movingbar') == 'movingbar'
    assert quality_family('noise', 'noise') is None
    assert 'not identified' in unsupported_reason(None)


# ------------------------------------------------------------------ registration
def test_registration_recovers_known_shift_and_preserves_labels(source):
    pres = source.load_presentation('chirp [control]')
    hr = make_highres(pres)
    reg = register_highres(pres, hr)
    assert reg.ok, reg.reason
    assert reg.scale == HR_SCALE and reg.shift_hr_px == HR_SHIFT
    hr_rois = map_labels_to_highres(pres.rois, reg, hr.image.shape)
    assert hr_rois.labels.dtype == pres.rois.labels.dtype
    assert hr_rois.roi_ids.tolist() == list(ROI_IDS)
    for k in ROI_IDS:  # every functional pixel of ROI k maps onto the 4x4 HighRes block moved by the shift
        ys, xs = np.nonzero(pres.rois.pixels(k))
        for y, x in zip(ys, xs):
            block = hr_rois.labels[y * 4 + HR_SHIFT[0]:(y + 1) * 4 + HR_SHIFT[0],
                                   x * 4 + HR_SHIFT[1]:(x + 1) * 4 + HR_SHIFT[1]]
            assert np.all(block == pres.rois.label_of(k))
    assert rois_outside(pres.rois, hr_rois) == []


def test_registration_refuses_wrong_metadata_and_uncorrelated_image(source):
    pres = source.load_presentation('chirp [control]')
    hr = make_highres(pres)
    moved = replace(hr, meta=dict(hr.meta, absx=hr.meta['absx'] + 50))
    reg = register_highres(pres, moved)
    assert not reg.ok and 'scan centres differ' in reg.reason
    noise = replace(hr, image=np.random.default_rng(0).normal(size=hr.image.shape))
    assert not register_highres(pres, noise).ok
    ok, reason, _ = check_metadata(pres.meta, (NY, NX), 2.0, hr.meta, (NY * 4, NX * 3), 0.5)
    assert not ok and 'fields of view differ' in reason


def test_shift_estimate_ignores_artifact_columns(source):
    pres = source.load_presentation('chirp [control]')
    func = np.median(pres.frames, axis=0)
    hr = make_highres(pres).image
    func_art = func.copy()
    func_art[:, :2] = 1e6  # light artifact must not drive the estimate
    assert estimate_shift(func_art, hr, HR_SCALE, npixartifact=2)[0] == HR_SHIFT


def test_map_labels_requires_valid_registration(source):
    pres = source.load_presentation('chirp [control]')
    with pytest.raises(ValueError):
        map_labels_to_highres(pres.rois, Registration(False, 'nope'), (64, 80))


# ------------------------------------------------------------------ renderer synchronisation
def _state(pres, **kw):
    return ViewState(pres=pres, signal='raw', eligible=tuple(pres.roi_ids), **kw)


def test_frame_trace_and_heatmap_share_the_clock(source):
    pres = source.load_presentation('chirp [drug]')  # offset 10000, ROI 11 starts at frame 10
    fig = CellVizFigure()
    for i in (0, 5, 37, len(pres.frame_times) - 1):
        fig.render(_state(pres, frame=i, selected=(2, 11)))
        t0, t1 = pres.frame_interval(i)
        np.testing.assert_array_equal(fig.movie_image.get_array(), pres.frames[i])
        assert fig.heat_cover.get_x() == pytest.approx(t1)
        assert fig.trace_cursor.get_x() == pytest.approx(t0)
        lines = {k: line for (line, _, _), k in zip(fig._trace_lines, (2, 11))}
        markers = {k: (m, off) for (m, _, off), k in zip(fig._trace_markers, (2, 11))}
        for k in (2, 11):
            x, y = lines[k].get_data()
            first = 10 if k == 11 else 0
            assert len(x) == max(0, i - first + 1)  # built up to the current frame, nothing later
            if len(x):
                assert x[-1] == pytest.approx(i * FRAME_DT + LINE_OFFSETS[k]) and x[-1] < t1
                m, off = markers[k]
                assert m.get_data()[1][0] - off == 10000 + 100 * k + i  # the sample of this frame and ROI


def test_selected_outline_matches_mask_and_click(source):
    pres = source.load_presentation('chirp [drug]')  # shifted mask
    fig = CellVizFigure()
    fig.render(_state(pres, selected=(5,)))
    seg = [c for c in fig.ax_field.collections if len(c.get_segments()) == len(outline_segments(pres.rois.pixels(5)))]
    assert any(np.allclose(np.array(c.get_segments()), outline_segments(pres.rois.pixels(5))) for c in seg)
    ys, xs = np.nonzero(pres.rois.pixels(5))
    assert fig.roi_at(fig.ax_field, float(xs[0]), float(ys[0])) == 5
    assert fig.roi_at(fig.ax_heat, 3.0, float(fig._heat_ids.index(5))) == 5
    assert fig.time_at(fig.ax_trace, 2.5) == 2.5


def test_highres_view_click_maps_to_same_roi(source):
    pres = source.load_presentation('chirp [control]')
    hr = make_highres(pres)
    reg = register_highres(pres, hr)
    hr_rois = map_labels_to_highres(pres.rois, reg, hr.image.shape)
    fig = CellVizFigure()
    fig.render(_state(pres, view='highres', highres=hr, hr_rois=hr_rois, selected=(11,), frame=30, overlay='activity'))
    ys, xs = np.nonzero(hr_rois.pixels(11))
    assert fig.roi_at(fig.ax_field, float(xs[0]), float(ys[0])) == 11
    assert fig.overlay_image is not None and fig.overlay_image.get_array().shape == hr.image.shape + (4,)


def test_frame_only_update_does_not_rebuild(source):
    pres = source.load_presentation('chirp [control]')
    fig = CellVizFigure()
    fig.render(_state(pres, frame=1))
    ax, image = fig.ax_field, fig.movie_image
    fig.render(_state(pres, frame=2))
    assert fig.builds == 1 and fig.movie_image is image  # only per-frame artists were updated
    fig.render(_state(pres, frame=2, display='selected', selected=(2,)))
    assert fig.builds == 2 and fig.movie_image is not image
    assert fig.ax_field is ax  # rebuilds reuse the axes (creating axes is the slow part of a rebuild)
    assert len(fig.ax_field.images) == 1 and len(fig.ax_trace.lines) == 2  # old artists removed: 1 line + marker


def test_presentation_data_from_source_is_complete(source):
    for info in source.list_presentations():
        pres = source.load_presentation(info.name)
        assert isinstance(pres, PresentationData) and pres.frames.shape == (len(pres.frame_times), NY, NX)
