import numpy as np
import pytest

from djimaging.utils import image_utils as iu


def _img():
    """Non-square (3, 4) image with distinct values."""
    return np.arange(12, dtype=float).reshape(3, 4)


def test_resize_image_nearest_non_square():
    out = iu.resize_image(_img(), output_shape=(6, 8), order=0)
    assert out.shape == (6, 8)
    np.testing.assert_array_equal(out, np.repeat(np.repeat(_img(), 2, axis=0), 2, axis=1))


def test_resize_image_keeps_nan_regions():
    img = np.array([[1., np.nan], [3., 4.], [5., 6.]])
    out = iu.resize_image(img, output_shape=(6, 4), order=0)
    expected = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    np.testing.assert_array_equal(out[np.isfinite(out)], expected[np.isfinite(expected)])
    assert np.isnan(img[0, 1])  # input unchanged


def test_resize_image_int_input():
    out = iu.resize_image(np.arange(6).reshape(2, 3), output_shape=(4, 6), order=0)
    assert out.dtype == float
    np.testing.assert_array_equal(out[::2, ::2], np.arange(6).reshape(2, 3))


def test_rotate_image_90_non_square():
    out = iu.rotate_image(_img(), angle=90, order=0)
    # counter-clockwise rotation, resized to the rotated shape
    np.testing.assert_array_equal(out, np.rot90(_img()))


def test_rotate_image_nan_and_cval():
    img = _img()
    img[0, 0] = np.nan
    out = iu.rotate_image(img, angle=90, order=0)
    expected = np.rot90(img)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    np.testing.assert_array_equal(out[np.isfinite(out)], expected[np.isfinite(expected)])

    out = iu.rotate_image(_img(), angle=45, order=0, cval=-1.)
    assert not np.any(np.isnan(out))
    assert out[0, 0] == -1.  # corner outside the rotated image


def test_rotate_image_float_does_not_modify_input():
    img = _img()
    img[1, 1] = np.nan
    iu.rotate_image(img, angle=30)
    assert np.isnan(img[1, 1]) and img[0, 0] == 0.


@pytest.mark.xfail(strict=True, reason="BUG: rotate_image crashes on integer images (NaN assigned to int array)")
def test_rotate_image_int_input():
    out = iu.rotate_image(np.arange(12).reshape(3, 4), angle=90, order=0)
    np.testing.assert_array_equal(out, np.rot90(np.arange(12).reshape(3, 4)))


def test_rescale_image_3d_scalar_scales_spatial_axes_only():
    img = np.stack([_img(), -_img()], axis=2)
    out = iu.rescale_image(img, 2)
    assert out.shape == (6, 8, 2)
    np.testing.assert_array_equal(out[..., 1], -np.repeat(np.repeat(_img(), 2, axis=0), 2, axis=1))


def test_rescale_image_3d_tuple():
    img = np.stack([_img()] * 4, axis=2)
    out = iu.rescale_image(img, (2, 1))
    assert out.shape == (6, 4, 4)
    np.testing.assert_array_equal(out[..., 0], np.repeat(_img(), 2, axis=0))


@pytest.mark.xfail(strict=True, reason="BUG: rescale_image crashes on 2-D images (scale padded to 3 axes)")
@pytest.mark.parametrize('scale, shape', [(2, (6, 8)), ((2, 1), (6, 4))])
def test_rescale_image_2d(scale, shape):
    assert iu.rescale_image(_img(), scale).shape == shape


def test_color_image():
    out = iu.color_image(_img(), cmap='gray', alpha=100)
    assert out.shape == (3, 4, 4)
    assert np.all(out[..., 3] == 100)
    # gray colormap: min -> black, max -> white, channels equal
    np.testing.assert_array_equal(out[0, 0, :3], [0, 0, 0])
    np.testing.assert_array_equal(out[-1, -1, :3], [255, 255, 255])
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    assert np.all(np.diff(out[..., 0].ravel()) >= 0)


def test_color_image_int_and_gamma():
    lin = iu.color_image(np.arange(12).reshape(3, 4), cmap='gray')
    gam = iu.color_image(np.arange(12).reshape(3, 4), cmap='gray', gamma=2.)
    assert np.all(gam[..., 0] <= lin[..., 0])
    with pytest.raises(AssertionError):
        iu.color_image(_img(), alpha=0.5)
    with pytest.raises(AssertionError):
        iu.color_image(np.zeros((2, 2, 2)))


def test_int_upscale_image():
    out = iu.int_upscale_image(_img(), 3)
    assert out.shape == (9, 12)
    np.testing.assert_array_equal(out[::3, ::3], _img())
    np.testing.assert_array_equal(out[2::3, 2::3], _img())


def test_int_downscale_image_3d_block_mean():
    img = np.arange(48, dtype=float).reshape(4, 6, 2)
    out = iu.int_downscale_image(img, 2)
    assert out.shape == (2, 3, 2)
    for c in range(2):
        expected = img[..., c].reshape(2, 2, 3, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(out[..., c], expected)


@pytest.mark.xfail(strict=True, reason="BUG: int_downscale_image adds a trailing axis to 2-D images")
def test_int_downscale_image_2d_keeps_ndim():
    img = np.arange(24, dtype=float).reshape(4, 6)
    out = iu.int_downscale_image(img, 2)
    np.testing.assert_allclose(out, img.reshape(2, 2, 3, 2).mean(axis=(1, 3)))


def test_int_rescale_image_dispatch():
    img = np.arange(48, dtype=float).reshape(4, 6, 2)
    assert iu.int_rescale_image(img, 1) is img
    assert iu.int_rescale_image(img, -1) is img
    assert iu.int_rescale_image(img, 0) is img
    assert iu.int_rescale_image(img, 2).shape == (8, 12, 2)
    assert iu.int_rescale_image(img, -2).shape == (2, 3, 2)
    # round trip: upscale, then downscale by block mean
    np.testing.assert_allclose(iu.int_rescale_image(iu.int_rescale_image(img, 3), -3), img)
