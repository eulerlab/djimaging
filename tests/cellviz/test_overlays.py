"""Field view overlays of CellViz (ROI outlines / ROI mask / ROI activity / none) in the Recording and HighRes view,
with and without blitting (SYNTHETIC data)."""
import numpy as np
import pytest

from djimaging.cellviz.registration import map_labels_to_highres, register_highres
from djimaging.cellviz.render import OVERLAYS, CellVizFigure, ViewState, mask_color, mask_rgba
from tests.cellviz.synthetic import SyntheticFieldSource, make_highres


@pytest.fixture(scope='module')
def pres():
    return SyntheticFieldSource().load_presentation('chirp [control]')


@pytest.fixture(scope='module')
def highres(pres):
    hr = make_highres(pres)
    return hr, map_labels_to_highres(pres.rois, register_highres(pres, hr), hr.image.shape)


def _state(pres, view='native', highres=None, **kw):
    extra = dict(view='highres', highres=highres[0], hr_rois=highres[1]) if view == 'highres' else {}
    return ViewState(pres=pres, signal='raw', eligible=(2, 5), selected=(5,), **extra, **kw)


def _field_pixels(fig):
    """RGB pixels of the field axes (plus the ROI labels right of it)."""
    bb = fig.ax_field.get_window_extent()
    h = fig.fig.bbox.height
    return fig.rgb()[int(h - bb.y1) - 2:int(h - bb.y0) + 2, int(bb.x0) - 2:int(bb.x1) + 25].astype(int)


def _count_color(img, rgb, tol=30):
    return int((np.abs(img - np.asarray(rgb)).sum(-1) < tol).sum())


def _count_cyan(img):
    """Pixels of the (alpha-blended) cyan outline colour of the eligible ROIs."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return int(((r < 60) & (g > 140) & (b > 180)).sum())


@pytest.mark.parametrize('view', ['native', 'highres'])
@pytest.mark.parametrize('overlay', list(OVERLAYS))
def test_overlay_is_drawn_identically_with_blitting(pres, highres, view, overlay):
    """With blitting, per-frame artists are drawn over a cached background. In the Recording view the movie changes
    every frame, so the ROI layers above it must be redrawn too; otherwise they disappear under the movie."""
    plain = CellVizFigure()
    plain.render(_state(pres, view, highres, overlay=overlay, frame=30))
    plain.draw()
    blit = CellVizFigure()
    blit.animated = True
    for frame in (10, 50, 30):
        blit.render(_state(pres, view, highres, overlay=overlay, frame=frame))
        blit.draw()
    a, b = _field_pixels(plain), _field_pixels(blit)
    tolerance = 60 if (view, overlay) == ('highres', 'activity') else 0  # Agg composites two images in a full draw
    assert np.abs(a - b).sum(-1).max() <= tolerance
    selected = tuple(int(round(255 * c)) for c in blit.state.color_of(5)[:3])
    assert _count_color(b, selected) > 20  # the selected ROI is always outlined, also above the movie


@pytest.mark.parametrize('view', ['native', 'highres'])
def test_overlay_modes_show_what_they_say(pres, highres, view):
    counts = {}
    for overlay in OVERLAYS:
        fig = CellVizFigure()
        fig.animated = True
        fig.render(_state(pres, view, highres, overlay=overlay, frame=30))
        fig.draw()
        img = _field_pixels(fig)
        counts[overlay] = dict(cyan=_count_cyan(img), images=len(fig.ax_field.images),
                               activity=fig.overlay_image is not None)
    assert counts['outlines']['cyan'] > 0 and counts['none']['cyan'] == 0 and counts['mask']['cyan'] == 0
    assert counts['mask']['images'] == counts['none']['images'] + 1  # the filled mask
    assert counts['activity']['activity'] and not counts['mask']['activity'] and not counts['outlines']['activity']


def test_mask_overlay_colours_eligible_rois_and_greys_out_the_rest(pres):
    rgba = mask_rgba(pres.rois, eligible=(2, 5))
    for roi_id in (2, 5):
        px = rgba[pres.rois.pixels(roi_id)]
        assert np.allclose(px[:, :3], mask_color(roi_id)[:3]) and np.all(px[:, 3] > 0.4)
    assert np.allclose(rgba[pres.rois.pixels(11)][:, :3], 0.6)  # ROI 11 excluded (e.g. by the QI filter): grey
    background = ~(pres.rois.pixels(2) | pres.rois.pixels(5) | pres.rois.pixels(11))
    assert np.all(rgba[background][:, 3] == 0)
    assert mask_color(2) != mask_color(5) != mask_color(11)


def test_invalid_overlay_is_rejected(pres):
    with pytest.raises(ValueError, match='overlay must be one of'):
        ViewState(pres=pres, signal='raw', overlay=True)


def test_viewer_overlay_selector_in_recording_view(tmp_path):
    pytest.importorskip('ipywidgets')
    from djimaging.cellviz.viewer import CellViz
    v = CellViz(SyntheticFieldSource(), presentation='chirp [control]', backend='image', export_dir=tmp_path)
    assert v.view == 'native' and v.overlay == 'outlines'
    assert [key for _, key in v.w_overlay.options] == list(OVERLAYS)
    n_images = len(v.figure.ax_field.images)
    v.set_overlay('mask')
    assert v.state.overlay == 'mask' and len(v.figure.ax_field.images) == n_images + 1
    v.set_frame(40)  # frame-only update keeps the mask above the movie
    assert v.figure.ax_field.images[-1] in v.figure._frame_artists
    v.set_overlay('activity')
    assert v.figure.overlay_image is not None
    with pytest.raises(ValueError):
        v.set_overlay('bogus')
    v2 = CellViz(SyntheticFieldSource(), presentation='chirp [control]', backend='image', overlay='mask')
    assert v2.state.overlay == 'mask'


def test_export_records_the_overlay(tmp_path):
    pytest.importorskip('ipywidgets')
    pytest.importorskip('imageio_ffmpeg')
    from djimaging.cellviz.viewer import CellViz
    v = CellViz(SyntheticFieldSource(), presentation='chirp [control]', backend='image', export_dir=tmp_path)
    v.set_overlay('mask')
    meta = v.export(t_start=1.0, t_stop=1.2)  # no path: a generated file name in the export folder
    assert meta['overlay'] == 'mask' and meta['file'].startswith(str(tmp_path)) and meta['file'].endswith('.mp4')
    v.export_dir = tmp_path / 'other'  # the folder field of the export row
    assert v.export(t_start=1.0, t_stop=1.2)['file'].startswith(str(tmp_path / 'other'))
