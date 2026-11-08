"""ROI canvas in motion mode: drawn on the presentation of the mask, no shifts between presentations."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from djimaging.autorois.roi_canvas import InteractiveRoiCanvas, RoiCanvas


def make_canvas(**kwargs):
    rng = np.random.default_rng(0)
    stacks = {'ch0': [rng.random((8, 6, 4)) + i for i in range(3)]}
    kwargs.setdefault('output_files', ['mask0.npy', 'mask1.npy', 'mask2.npy'])
    return RoiCanvas(stacks=stacks, **kwargs)


def test_starts_on_main_presentation():
    canvas = make_canvas(main_stim_idx=2)
    assert canvas._selected_stim_idx == 2 and canvas.output_file == 'mask2.npy'
    np.testing.assert_allclose(canvas.backgrounds['stack: ch0[mean]'], canvas.stacks['ch0'][2].mean(axis=2),
                               rtol=1e-6)
    assert make_canvas()._selected_stim_idx == 0  # default unchanged


def test_main_presentation_out_of_range():
    with pytest.raises(ValueError):
        make_canvas(main_stim_idx=3)


def test_no_shifts_without_allow_shifts():
    canvas = make_canvas(allow_shifts=False, main_stim_idx=1)
    assert not canvas.allow_shifts and np.all(canvas.shifts == 0)
    with pytest.raises(ValueError):
        make_canvas(allow_shifts=False, shifts=[[0, 0], [1, 0], [0, 0]])


def test_same_mask_for_all_presentations_without_shifts():
    """prep_roi_mask_for_file shifts the mask by the shift of the selected presentation, which is always 0."""
    canvas = make_canvas(allow_shifts=False, main_stim_idx=1)
    canvas.roi_masks[2:4, 2:4] = 1
    canvas.exec_save_current_roi = lambda: None
    masks = []
    for idx in range(3):
        canvas._selected_stim_idx = idx
        masks.append(InteractiveRoiCanvas.prep_roi_mask_for_file(canvas))
    assert all(np.array_equal(m, canvas.roi_masks) for m in masks)


def _widget_stub(allow_shifts, selected, main=0, read_only=False):
    widget = lambda: SimpleNamespace(disabled=None, value=None)  # noqa: E731
    return SimpleNamespace(
        allow_shifts=allow_shifts, _selected_stim_idx=selected, main_stim_idx=main, log=lambda msg: None,
        shifts=np.zeros((3, 2), dtype=int), widget_read_only=SimpleNamespace(value=read_only),
        widget_shift_dx=widget(), widget_shift_dy=widget(), widget_auto_shift=widget())


@pytest.mark.parametrize('allow_shifts, selected, disabled', [
    (True, 0, True),  # main presentation: never shifted
    (True, 1, False),
    (False, 1, True),  # motion mode: shifts disabled for all presentations
])
def test_shift_widgets(allow_shifts, selected, disabled):
    stub = _widget_stub(allow_shifts, selected)
    InteractiveRoiCanvas.update_shift_widget(stub)
    assert stub.widget_shift_dx.disabled is disabled and stub.widget_shift_dy.disabled is disabled
    assert stub.widget_auto_shift.disabled is disabled


def test_set_shift_rejected_without_allow_shifts():
    stub = _widget_stub(False, 1)
    with pytest.raises(ValueError):
        InteractiveRoiCanvas.set_shift(stub, 1, axis='x')
    assert np.all(stub.shifts == 0)


# Interactive canvas in motion mode (the widgets work without a running notebook)

def _initial_mask():
    mask = np.zeros((8, 6), dtype=int)
    mask[2:4, 1:3] = 7  # labels as in the database; the canvas relabels them
    mask[5:7, 3:5] = 9
    return mask


def make_interactive(tmp_path, view_masks=None, **kwargs):
    pytest.importorskip('ipycanvas')
    rng = np.random.default_rng(0)
    stacks = {'ch0': [rng.random((8, 6, 4)) for _ in range(3)]}
    field_file = str(tmp_path / 'ROIs' / 'pres1_FieldROIs.npy')
    canvas = InteractiveRoiCanvas(
        stacks=stacks, output_files=[field_file] * 3, initial_roi_mask=_initial_mask(), main_stim_idx=1,
        allow_shifts=False, view_masks=view_masks, **kwargs)
    canvas.set_read_only(False)
    return canvas, field_file


def test_other_presentations_are_view_only(tmp_path):
    canvas, _ = make_interactive(tmp_path)
    assert not canvas.is_view_only and not canvas.widget_save.disabled and not canvas.widget_save_to_file.disabled

    canvas.set_selected_stim(canvas.pres_names[0])
    assert canvas.is_view_only
    assert canvas.widget_save.disabled and canvas.widget_tool.disabled and canvas.widget_save_to_file.disabled
    assert canvas.widget_shift_dx.disabled and canvas.widget_auto_shift.disabled
    canvas.set_dangerzone(True)  # does not enable editing on a view-only presentation
    assert canvas.widget_exec_autorois.disabled and canvas.widget_kill_all.disabled
    with pytest.raises(RuntimeError):
        canvas.exec_save_to_file()

    before, current_before = canvas.roi_masks.copy(), canvas.current_mask.copy()
    canvas.set_selected_tool('draw')  # e.g. via the widget; drawing is still ignored
    canvas._apply_tool(1 * canvas.rescale, 1 * canvas.rescale, down=True)
    assert np.array_equal(canvas.current_mask, current_before) and np.array_equal(canvas.roi_masks, before)

    canvas.set_selected_stim(canvas.pres_names[1])
    assert not canvas.is_view_only and not canvas.widget_save.disabled and not canvas.widget_save_to_file.disabled


def test_save_all_writes_only_the_field_mask_file(tmp_path):
    from djimaging.utils.mask_utils import load_roi_mask_file
    canvas, field_file = make_interactive(tmp_path)
    canvas.output_files = [str(tmp_path / 'ROIs' / f'pres{i}_FieldROIs.npy') for i in range(3)]  # to detect copies
    canvas.set_selected_stim(canvas.pres_names[2])
    canvas.exec_save_all_to_file(raise_error=True)
    assert [os.path.isfile(f) for f in canvas.output_files] == [False, True, False]
    assert np.array_equal(load_roi_mask_file(field_file) > 0, _initial_mask() > 0)


def test_view_masks_shown_until_the_mask_is_edited(tmp_path, monkeypatch):
    view = np.zeros((8, 6), dtype=int)
    view[3:5, 1:3] = 7  # ROI 7 moved by one pixel in this presentation
    view[5:7, 3:5] = 9
    canvas, _ = make_interactive(tmp_path, view_masks=[view, None, None])
    monkeypatch.setattr(canvas, 'compute_autoshift', lambda shift_max=5, fun_progress=None: (1, 0))

    canvas.set_selected_stim(canvas.pres_names[0])
    shown = canvas.get_displayed_roi_masks()
    assert canvas._view_source == 'MotionRoiMask'
    # Same ROI, same (canvas) label as in the editable mask, so the same color
    label_7 = canvas.roi_masks[2, 1]
    assert np.all(shown[3:5, 1:3] == label_7) and shown[2, 1] == 0

    canvas.set_selected_stim(canvas.pres_names[1])  # edit the mask: the view mask is outdated -> preview
    canvas.exec_kill_roi()
    canvas.set_selected_stim(canvas.pres_names[0])
    canvas.get_displayed_roi_masks()
    assert canvas._view_source.startswith('preview')


def test_preview_moves_mask_by_auto_shift(tmp_path, monkeypatch):
    from djimaging.utils.mask_utils import shift_array
    canvas, _ = make_interactive(tmp_path)
    monkeypatch.setattr(canvas, 'compute_autoshift', lambda shift_max=5, fun_progress=None: (1, -1))
    canvas.set_selected_stim(canvas.pres_names[2])
    shown = canvas.get_displayed_roi_masks()
    assert canvas._view_source == 'preview (global shift 1, -1)'
    # As saved for a presentation with this shift (prep_roi_mask_for_file)
    assert np.array_equal(shown, shift_array(canvas.roi_masks.copy(), shift=(-1, 1), cval=0))
