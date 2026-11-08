"""ROI canvas in motion mode: drawn on the presentation of the mask, no shifts between presentations."""
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
