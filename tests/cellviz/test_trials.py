"""Trial mean ± SD panel, blitting and layout of CellViz (SYNTHETIC data).

Synthetic encoding (see ``synthetic.py``): sample ``j`` of ROI ``k`` in presentation offset ``P`` is ``P + 100 k + j``
at ``t = j * 0.05 + line offset``. Triggers 1.0, 1.5, 3.0, 3.5 s. With ``pad_trace=True`` the pipeline starts each
snippet at the last sample before the onset trigger, i.e. frame 19 for 1.0 s and frame 59 for 3.0 s, so every expected
value below follows from the frame indices.
"""
from dataclasses import replace

import numpy as np
import pytest

from djimaging.cellviz.data import StimulusInfo
from djimaging.cellviz.render import CellVizFigure, ViewState
from djimaging.cellviz.trials import TrialCache, summarize, trial_layout
from tests.cellviz.synthetic import FRAME_DT, TRIGGERS, SyntheticFieldSource


@pytest.fixture
def source():
    return SyntheticFieldSource()


def _state(pres, **kw):
    return ViewState(pres=pres, signal='raw', eligible=tuple(pres.roi_ids), **kw)


# ------------------------------------------------------------------ summaries
def test_chirp_summary_uses_pipeline_alignment(source):
    pres = source.load_presentation('chirp [control]')
    cache = TrialCache(pres, 'raw')
    assert cache.layout.onsets.tolist() == [1.0, 3.0] and cache.layout.condition_names == ('',)
    snip = cache.snippets(2)
    assert snip.values.shape == (41, 2)  # median repeat length 40 frames + 1 (pad_trace)
    np.testing.assert_array_equal(snip.values[0], [200 + 19, 200 + 59])
    (c,) = cache.roi_summary(2).conditions
    assert c.n == 2
    np.testing.assert_allclose(c.mean, 200 + 39 + np.arange(41))
    np.testing.assert_allclose(c.sd, 40 / np.sqrt(2))  # SD across the two repeats, ddof=1
    np.testing.assert_allclose(c.rel_times[0], 19 * FRAME_DT + 0.004 - 1.0)  # relative to the onset trigger


def test_population_averages_rois_within_each_repeat_first(source):
    pres = source.load_presentation('chirp [control]')
    (c,) = TrialCache(pres, 'raw').population_summary((2, 5, 11)).conditions
    np.testing.assert_allclose(c.mean, 100 * np.mean([2, 5, 11]) + 39 + np.arange(41))
    # variability across repeats of the population mean; pooling ROIs and repeats would give SD of several hundred
    np.testing.assert_allclose(c.sd, 40 / np.sqrt(2))
    assert c.n == 2


def test_moving_bar_never_pools_directions(source):
    pres = source.load_presentation('movingbar')  # triggers -> directions 0, 180, 0, 180
    cache = TrialCache(pres, 'raw')
    assert cache.layout.condition_names == ('0°', '180°') and cache.layout.condition.tolist() == [0, 1, 0, 1]
    c0, c180 = cache.roi_summary(2).conditions
    assert c0.n == c180.n == 2
    np.testing.assert_allclose(c0.mean, 20000 + 200 + (19 + 59) / 2 + np.arange(11))   # trials starting 1.0, 3.0 s
    np.testing.assert_allclose(c180.mean, 20000 + 200 + (29 + 69) / 2 + np.arange(11))  # trials starting 1.5, 3.5 s
    assert cache.roi_summary(5) is None  # ROI 5 has no trace in this presentation


def test_single_repeat_has_no_sd_and_invalid_repeats_are_skipped(source):
    pres = source.load_presentation('movingbar')
    four = StimulusInfo(True, 1, [dict(name=d, ntrigger=1) for d in (0, 90, 180, 270)], pad_trace=True)
    summary = TrialCache(replace(pres, stimulus=four), 'raw').roi_summary(2)
    assert [c.n for c in summary.conditions] == [1, 1, 1, 1] and all(c.sd is None for c in summary.conditions)
    layout = trial_layout(TRIGGERS, StimulusInfo(True, 2))
    snip = TrialCache(source.load_presentation('chirp [control]'), 'raw').snippets(2)
    bad = replace(snip, valid=np.array([True, False]))
    (c,) = summarize(bad, layout).conditions
    assert c.n == 1 and c.sd is None


def test_no_repeat_structure_disables_panel(source):
    pres = replace(source.load_presentation('chirp [control]'), stimulus=None)
    fig = CellVizFigure()
    fig.render(_state(pres, frame=50))
    assert not fig._trial_series and fig._trials.unavailable_reason() == 'stimulus unknown: no repeat structure'


# ------------------------------------------------------------------ panel: reveal, scrubbing, colours
def _trial_xy(fig, k=0):
    return fig._trial_series[k].line.get_data()


def test_trial_panel_reveals_progressively_forward_and_backward(source):
    pres = source.load_presentation('chirp [control]')
    fig = CellVizFigure()
    for i in (5, 30, 45, 70, 30, 10, 70):  # includes backward jumps and a jump into the second repeat
        fig.render(_state(pres, frame=i, display='selected', selected=(2,)))
        t0, t1 = pres.frame_interval(i)
        x, y = _trial_xy(fig)
        if t1 <= 1.0:  # before the first onset: nothing revealed, cursor hidden
            assert len(x) == 0 and not fig.trial_cursor.get_visible()
            continue
        onset = 3.0 if t1 > 3.0 else 1.0
        assert fig._current_trial == (1 if onset == 3.0 else 0)
        assert len(x) > 0 and x[-1] < t1 - onset and (len(x) == 41 or x[len(x) - 1] < t1 - onset)
        full = fig._trial_series[0].per_condition[0][0]
        assert len(x) == np.searchsorted(full, t1 - onset)  # exactly the samples before the current time
        verts = fig._trial_series[0].band.get_paths()[0].vertices if len(x) > 1 else None
        if verts is not None:
            assert verts[:, 0].max() == pytest.approx(x[-1])  # band revealed exactly as far as the mean
        assert fig.trial_cursor.get_x() == pytest.approx(t0 - onset)
        assert fig.trial_window.get_x() == pytest.approx(onset)
    assert 'n = 2 repeats' in fig.trial_label.get_text()


def test_trial_panel_follows_current_direction(source):
    pres = source.load_presentation('movingbar')
    fig = CellVizFigure()
    for i, cond in ((35, '180°'), (65, '0°'), (78, '180°')):  # t1 = 1.8 s (trial 1), 3.3 s (trial 2), 3.95 s (trial 3)
        fig.render(_state(pres, frame=i, display='selected', selected=(2,)))
        x, y = _trial_xy(fig)
        expected = fig._trial_series[0].per_condition[0 if cond == '0°' else 1][1]
        np.testing.assert_array_equal(y, expected[:len(y)])
        assert fig.trial_label.get_text().startswith(cond)


def test_selected_colours_match_between_panels_and_population_mode(source):
    pres = source.load_presentation('chirp [control]')
    fig = CellVizFigure()
    fig.render(_state(pres, frame=60, display='selected', selected=(5, 2)))
    trace_colours = [line.get_color() for line, _, _ in fig._trace_lines]
    trial_colours = [s.line.get_color() for s in fig._trial_series]
    assert trace_colours == trial_colours and len(trial_colours) == 2
    assert fig.ax_trial.get_shared_y_axes().joined(fig.ax_trial, fig.ax_trace)  # same signal, units and offsets
    fig.render(_state(pres, frame=60, display='all'))
    assert len(fig._trial_series) == 1 and fig.ax_trial.get_xlabel() == 'Time from onset (s)'
    assert fig.ax_trace.get_xlabel() == 'Recording time (s)'


def test_click_on_trial_panel_seeks_within_current_trial(source):
    pres = source.load_presentation('chirp [control]')
    fig = CellVizFigure()
    fig.render(_state(pres, frame=70))
    assert fig.time_at(fig.ax_trial, 0.5) == pytest.approx(3.5)


# ------------------------------------------------------------------ blitting and layout
@pytest.mark.parametrize('display', ['all', 'selected'])
def test_blitted_frames_leave_no_stale_pixels(source, display):
    pres = source.load_presentation('movingbar')
    blit = CellVizFigure()
    blit.animated = True
    for i in (10, 40, 75, 25):  # several frames, incl. a direction change and a backward jump
        blit.render(_state(pres, frame=i, display=display, selected=(2, 11)))
        blit.draw()
    ref = CellVizFigure()  # fresh figure, first draw of frame 25 (full draw + per-frame artists on top)
    ref.animated = True
    ref.render(_state(pres, frame=25, display=display, selected=(2, 11)))
    ref.draw()
    assert np.array_equal(blit.rgb(), ref.rgb())
    blit.fig.canvas.draw()  # a full redraw (e.g. after a resize) re-captures the background and redraws the frame
    assert np.array_equal(blit.rgb(), ref.rgb())


def test_field_axes_square_and_bottom_row_split(source):
    pres = source.load_presentation('chirp [control]')
    for size in ((9.0, 6.2), (12.8, 7.36), (6.0, 5.0)):
        fig = CellVizFigure(figsize=size)
        fig.render(_state(pres))
        w, h = fig.fig.get_size_inches() * fig.fig.dpi
        box = fig.ax_field.get_position(original=True)  # allotted box; a non-square frame letterboxes inside
        assert box.width * w == pytest.approx(box.height * h)
        trace, trial = fig.ax_trace.get_position(), fig.ax_trial.get_position()
        assert trace.width / (trace.width + trial.width) == pytest.approx(0.6)
        assert trace.y0 == trial.y0 and trial.x0 > trace.x1


def test_trigger_labels_stay_in_the_label_band(source):
    pres = source.load_presentation('movingbar')
    fig = CellVizFigure()
    fig.render(_state(pres))
    fig.fig.canvas.draw()
    renderer = fig.fig.canvas.get_renderer()
    labels = [t for t in fig.ax_trace.texts if t.get_text() in ('0°', '180°')]
    assert labels
    boxes = [t.get_window_extent(renderer) for t in labels]
    trace_top = fig.ax_trace.get_window_extent(renderer).y1
    heat_bottom = fig.ax_heat.get_window_extent(renderer).y0
    for a, b in zip(boxes, boxes[1:]):
        assert a.x1 <= b.x0  # no overlap between neighbours
    assert all(b.y0 >= trace_top - 1 and b.y1 < heat_bottom for b in boxes)  # above the trace, below the heatmap


def test_rebuilds_on_reused_axes_leave_no_residue(source):
    pres = source.load_presentation('chirp [control]')
    mb = source.load_presentation('movingbar')
    fig = CellVizFigure()
    fig.animated = True
    for st in (_state(pres, frame=30, display='selected', selected=(2, 5)),
               replace(_state(mb, frame=50, display='all'), eligible=(2,)),
               replace(_state(pres, frame=70, display='selected', selected=(11,)), signal=None),
               _state(pres, frame=70, display='selected', selected=(5,))):
        fig.render(st)
        fig.draw()
    ref = CellVizFigure()
    ref.animated = True
    ref.render(_state(pres, frame=70, display='selected', selected=(5,)))
    ref.draw()
    assert fig.builds == 4 and np.array_equal(fig.rgb(), ref.rgb())
