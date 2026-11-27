"""Notebook viewer and MP4 export (SYNTHETIC data). Needs ipywidgets (and imageio-ffmpeg for the export tests)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('ipywidgets')

from djimaging.cellviz.export import read_video  # noqa: E402
from djimaging.cellviz.render import CellVizFigure  # noqa: E402
from djimaging.cellviz.viewer import CellViz  # noqa: E402
from tests.cellviz.synthetic import FRAME_DT, HR_SHIFT, SyntheticFieldSource  # noqa: E402


@pytest.fixture
def viewer(tmp_path):
    return CellViz(SyntheticFieldSource(), presentation='chirp [control]', export_dir=tmp_path, backend='image')


def click(v, ax, x, y):
    v._on_click(SimpleNamespace(inaxes=ax, xdata=x, ydata=y))


def test_presentations_and_switching_select_matching_data(viewer):
    v = viewer
    assert list(v.w_pres.options) == ['chirp [control]', 'chirp [drug]', 'movingbar']
    assert v.pres.info.key['cond1'] == 'control' and v.pres.frames[0, 0, 0] == 0
    v.select_presentation('chirp [drug]')
    assert v.pres.info.key['cond1'] == 'drug' and v.pres.frames[0, 0, 0] == 10000
    assert v.state.pres is v.pres and 'cond1=drug' in v.w_notes.value  # keys are in the Details section
    assert v.w_title.value == '<b>SYNTHETIC field</b>'  # field shown once; stimulus/channel in their dropdowns
    v.select_presentation('chirp [control]', channel='wDataCh1')
    assert v.source.loads[-1] == ('chirp [control]', 'wDataCh1')


def test_click_selects_roi_with_matching_contour_and_trace(viewer):
    v = viewer
    rois = v.pres.rois
    ys, xs = np.nonzero(rois.pixels(5))
    click(v, v.figure.ax_field, float(xs[0]), float(ys[0]))
    assert v.selected == [5] and v.w_rois.value == (5,)
    line = v.figure._trace_lines[0][0]
    v.set_frame(40)
    x, y = line.get_data()
    assert x[-1] == pytest.approx(40 * FRAME_DT + 0.021) and y[-1] == 100 * 5 + 40  # ROI 5's own trace, frame 40
    click(v, v.figure.ax_field, float(xs[0]), float(ys[0]))  # second click deselects
    assert v.selected == []
    click(v, v.figure.ax_trace, 2.26, 0.)  # click on the trace axis seeks
    assert v.frame == int(2.26 // FRAME_DT)


def test_selection_follows_roi_ids_across_shifted_masks(viewer):
    v = viewer
    v.select_rois([2, 11])
    v.select_presentation('chirp [drug]')  # mask shifted by one pixel in x
    assert v.selected == [2, 11]
    outline_x = [c for c in v.figure.ax_field.collections if c.get_linewidths()[0] > 2]
    xs = np.nonzero(v.pres.rois.pixels(2))[1]
    assert min(np.array(outline_x[0].get_segments())[:, :, 0].min(axis=1)) == pytest.approx(xs.min() - 0.5)


def test_scrubbing_and_speed_keep_alignment(viewer):
    v = viewer
    v.select_rois([2])
    for t in (0.0, 0.049, 0.05, 1.234, 5.97):
        v.set_time(t)
        i = v.frame
        assert v.pres.frame_times[i] <= t < v.pres.frame_times[i] + FRAME_DT
        x, y = v.figure._trace_lines[0][0].get_data()
        assert y[-1] == 200 + i and x[-1] < v.pres.frame_times[i] + FRAME_DT
        assert v.figure.heat_cover.get_x() == pytest.approx(v.pres.frame_times[i] + FRAME_DT)
    state = v.state
    for speed in (0.25, 0.5, 1.0, 2.0):
        v._render_time = 0.
        v.w_speed.value = speed
        assert v.w_play.interval == max(10, round(1000 * FRAME_DT / speed))
        assert v.state.frame == state.frame and v.current_time == state.pres.frame_times[state.frame]


def test_quality_filter_drives_visibility_and_selection(viewer):
    v = viewer  # control: qidx {2: 0.5, 5: 0.2}, ROI 11 missing
    assert v.threshold is None and v.eligible == (2, 5, 11)  # filter starts off
    v.select_rois([2, 5])
    v.set_threshold(0.3)
    assert v.eligible == (2,)
    assert [val for _, val in v.w_rois.options] == [2] and v.w_rois.value == (2,)
    assert v.selected == [2, 5] and v.state.shown_selected == (2,)  # ROI 5 kept, hidden
    assert 'selected but hidden' in v.w_selection_note.value and '[5]' in v.w_selection_note.value
    assert '<b>1</b> of 3 ROIs eligible' in v.w_qi_info.value and '1 below, 1 without value' in v.w_qi_info.value
    assert v.figure._heat_ids == [2]
    v.toggle_roi(11)
    assert 11 not in v.selected and 'no quality value' in v.w_status.value
    v.set_threshold(None)
    assert v.state.shown_selected == (2, 5) and v.figure._heat_ids == [2, 5, 11]


def test_metric_change_resets_filter(viewer):
    v = viewer
    v.set_threshold(0.3)
    v.select_presentation('chirp [drug]')  # same metric: filter stays on
    assert v.threshold == pytest.approx(0.3) and v.eligible == (5, 11)  # drug: {2: 0.1, 5: 0.6, 11: 0.4}
    v.select_presentation('movingbar')  # other metric: off, d_qi default
    assert v.threshold is None and v.w_qi_on.description == 'QI filter (d_qi)'


def test_highres_registered_and_fallback(tmp_path):
    v = CellViz(SyntheticFieldSource(), presentation='chirp [control]', backend='image')
    v.set_view('highres')
    assert v.view == 'highres' and v.state.hr_rois is not None
    assert v.state.hr_rois.roi_ids.tolist() == v.pres.roi_ids.tolist()
    assert f"{HR_SHIFT[1] / 4:+.2f}" in v.w_notes.value and 'registered' in v.w_status.value  # diagnostics: Details
    v2 = CellViz(SyntheticFieldSource(with_highres=False), presentation='chirp [control]', backend='image')
    v2.set_view('highres')
    assert v2.view == 'native' and 'HighRes view unavailable' in v2.w_status.value
    v2.set_frame(10)
    np.testing.assert_array_equal(v2.figure.movie_image.get_array(), v2.pres.frames[10])


def test_export_replays_same_range_and_selection(viewer, tmp_path):
    pytest.importorskip('imageio_ffmpeg')
    v = viewer
    v.select_rois([5, 2])
    v.set_display('selected')
    meta = v.export(tmp_path / 'out.mp4', t_start=1.0, t_stop=2.0, speed=0.5)
    assert (meta['frame_start'], meta['frame_stop']) == (20, 40) and meta['fps'] == pytest.approx(0.5 / FRAME_DT)
    assert meta['selected'] == [5, 2] and meta['display'] == 'selected'
    assert json.loads((tmp_path / 'out.mp4.json').read_text())['n_frames'] == 20
    vmeta, frames = read_video(tmp_path / 'out.mp4')
    assert len(frames) == 20 and vmeta['fps'] == pytest.approx(10, rel=1e-3)
    fig = CellVizFigure(figsize=(12.8, 7.36), dpi=100)
    fig.animated = True
    for k in (0, 19):
        fig.render(v.state.__class__(**{**v.state.__dict__, 'frame': 20 + k}))
        fig.draw()
        ref = fig.rgb().astype(float)
        assert np.abs(frames[k].astype(float) - ref).mean() < 4  # same frame up to compression
    other = fig.rgb().astype(float)
    assert np.abs(frames[0].astype(float) - other).mean() > np.abs(frames[19].astype(float) - other).mean()


def test_export_refuses_data_folder(tmp_path):
    pytest.importorskip('imageio_ffmpeg')
    data = tmp_path / 'data'
    v = CellViz(SyntheticFieldSource(data_roots=(data,)), backend='image')
    with pytest.raises(ValueError, match='read-only'):
        v.export(data / 'sub' / 'x.mp4', t_start=0, t_stop=0.2)


def test_colours_follow_selection_order_and_absent_ids_are_reported(viewer):
    v = viewer
    v.select_rois([5, 99, 2])  # 99 is not in the mask
    colours = {r: v.state.color_of(r) for r in (5, 2)}
    v.set_threshold(0.3)  # ROI 5 drops out; ROI 2 keeps its colour
    assert v.state.color_of(2) == colours[2]
    assert "not in this presentation's mask: [99]" in v.w_selection_note.value
    assert 'hidden by the quality filter or without value: [5]' in v.w_selection_note.value


def test_queued_play_ticks_are_dropped_but_playback_stays_on_time(viewer):
    import time
    v = viewer
    v.w_play.playing = True
    n = v.dropped_ticks
    v._last_tick_end = time.perf_counter()  # tick handled right after the previous one finished: it was queued
    v.w_frame.value = 10
    assert v.dropped_ticks == n + 1 and v.frame == 10 and v.figure.state.frame != 10
    v._last_tick_end = 0.  # next tick arrives after a normal wait: rendered
    v.w_frame.value = 11
    assert v.figure.state.frame == 11 and v.dropped_ticks == n + 1
    v._last_tick_end = time.perf_counter()
    v.w_frame.value = v.pres.n_frames - 1  # the last frame is always shown
    assert v.figure.state.frame == v.pres.n_frames - 1
    v.w_play.playing = False
    v._last_tick_end = time.perf_counter()
    v.w_frame.value = 3  # scrubbing (not playing) always renders
    assert v.figure.state.frame == 3
    stats = v.playback_stats()
    assert set(stats) >= {'prepare', 'draw', 'encode', 'total', 'dropped_ticks'} and stats['total'] > 0


def test_trial_panel_in_viewer_and_export(viewer, tmp_path):
    v = viewer
    v.select_rois([2])
    v.set_display('selected')
    v.set_time(3.4)  # second chirp repeat
    x, _ = v.figure._trial_series[0].line.get_data()
    assert v.figure._current_trial == 1 and 0 < x[-1] < 0.45
    v.set_time(1.2)  # backward into the first repeat
    x, _ = v.figure._trial_series[0].line.get_data()
    assert v.figure._current_trial == 0 and x[-1] < 0.25
    pytest.importorskip('imageio_ffmpeg')
    meta = v.export(tmp_path / 'trial.mp4', t_start=2.9, t_stop=3.3)
    _, frames = read_video(tmp_path / 'trial.mp4')
    assert len(frames) == meta['n_frames'] == len(range(meta['frame_start'], meta['frame_stop']))
    fig = CellVizFigure(figsize=(12.8, 7.36), dpi=100)
    fig.animated = True
    fig.render(v.state.__class__(**{**v.state.__dict__, 'frame': meta['frame_stop'] - 1}))
    fig.draw()
    assert np.abs(frames[-1].astype(float) - fig.rgb().astype(float)).mean() < 4  # incl. the revealed trial panel
    assert len(fig._trial_series[0].line.get_data()[0]) > 0


def test_playback_follows_the_wall_clock_when_ticks_are_late(viewer):
    import time
    v = viewer
    v.w_play.playing = True  # anchors the playback clock at the current frame
    v._play_anchor = (time.perf_counter() - 1.0, 20)  # pretend playback started 1 s ago at frame 20
    v._last_tick_frame = v._last_raw_tick = 20
    v._last_tick_end = 0.
    v.w_frame.value = 21  # a late tick: the browser timer only reached frame 21
    due = 20 + int(1.0 / FRAME_DT)
    assert v.frame >= due and v.figure.state.frame == v.frame  # rendered at the recording time, not the tick
    assert v.w_play.value == v.frame  # the play widget continues from there
    v._last_tick_end = 0.
    v.w_frame.value = v.frame  # echo of that value from the browser: not rendered twice
    assert v.figure.state.frame == v.frame
    v.w_play.playing = False
    assert v._play_anchor is None


def test_stale_ticks_after_catch_up_do_not_reset_the_clock(viewer):
    import time
    v = viewer
    v.w_play.playing = True
    v._play_anchor = (time.perf_counter() - 1.0, 20)
    v._last_tick_frame, v._last_raw_tick, v._last_tick_end = 20, 20, 0.
    v.w_frame.value = 21  # late tick -> jumps ahead to the clock
    caught_up, anchor = v.frame, v._play_anchor
    v._last_tick_end = 0.
    v.w_frame.value = 22  # a tick sent before the jump: smaller than the frame shown
    assert v._play_anchor == anchor and v.frame >= caught_up  # no step back, clock kept
    v._last_tick_end = 0.
    v.w_frame.value = 3  # a real jump backwards (user scrubbed while playing): re-anchored there
    assert v.frame == 3 and v._play_anchor[1] == 3
