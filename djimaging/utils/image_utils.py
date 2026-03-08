import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.math_utils import normalize_zero_one


def resize_image(image: np.ndarray, output_shape: tuple, order: int = 0) -> np.ndarray:
    """Resize an image to `output_shape`, preserving NaN regions.

    Non-finite pixels are temporarily replaced with the image minimum before
    resizing and are restored as NaN in the output wherever the resized
    finite-pixel mask is zero.

    Parameters
    ----------
    image : np.ndarray
        Input image array of any shape.
    output_shape : tuple
        Desired output shape (same number of dimensions as `image`).
    order : int, optional
        Interpolation order passed to ``skimage.transform.resize``.
        Default is 0 (nearest-neighbour).

    Returns
    -------
    np.ndarray
        Resized image as a float array, with NaN where the original image
        had non-finite values.
    """
    from skimage.transform import resize

    image = image.copy().astype(float)
    finite_mask = np.isfinite(image)
    image[~finite_mask] = np.nanmin(image)
    image_resized = resize(
        image, output_shape=output_shape, order=order, mode='constant', anti_aliasing=False)
    mask_resized = resize(
        finite_mask.astype(int), output_shape=output_shape, order=0, mode='constant', anti_aliasing=False)
    assert image_resized.shape == mask_resized.shape
    image_resized[mask_resized == 0] = np.nan
    return image_resized


def rotate_image(image: np.ndarray, angle: float, order: int = 1,
                 cval: float | None = None) -> np.ndarray:
    """Rotate a 2-D image by `angle` degrees, preserving NaN regions.

    Non-finite pixels are temporarily replaced with the image minimum before
    rotation. After rotation, pixels outside the original finite region are set
    to NaN (or `cval` if provided).

    Parameters
    ----------
    image : np.ndarray
        2-D input image array.
    angle : float
        Rotation angle in degrees (counter-clockwise).
    order : int, optional
        Interpolation order passed to ``skimage.transform.rotate``.
        Default is 1 (bilinear).
    cval : float | None, optional
        Value used to fill pixels outside the original image boundary after
        rotation. If None (default), NaN is used.

    Returns
    -------
    np.ndarray
        Rotated image (resized to fit the rotated content) as a float array.
    """
    from skimage.transform import rotate

    finite_mask = np.isfinite(image)
    image = image.copy()

    minv = np.nanmin(image)
    image[~finite_mask] = minv
    image_rotated = rotate(image, angle=angle, resize=True, order=order, cval=minv if cval is None else cval)
    mask_rotated = rotate(finite_mask.astype(int), angle=angle, resize=True, order=0, cval=0)
    assert image_rotated.shape == mask_rotated.shape
    image_rotated[mask_rotated == 0] = np.nan if cval is None else cval
    return image_rotated


def rescale_image(image: np.ndarray, scale: float | tuple, order: int = 0) -> np.ndarray:
    """Rescale an image by a given scale factor.

    Parameters
    ----------
    image : np.ndarray
        Input image array of any shape.
    scale : float | tuple
        Scale factor(s). A scalar is applied uniformly to the first two
        spatial axes. A 2-tuple ``(sy, sx)`` scales the axes independently.
    order : int, optional
        Interpolation order passed to :func:`resize_image`. Default is 0.

    Returns
    -------
    np.ndarray
        Rescaled image array.
    """
    if not hasattr(scale, '__iter__'):
        scale = (scale, scale, 1)
    elif len(scale) == 2:
        scale = (scale[0], scale[1], 1)

    output_shape = np.ceil(np.asarray(image.shape) * np.asarray(scale)).astype('int')
    resized_image = resize_image(image, output_shape=output_shape, order=order)
    return resized_image


def color_image(data_img: np.ndarray, cmap: str = 'viridis',
                gamma: float = 1.0, alpha: int = 255) -> np.ndarray:
    """Map a 2-D data image to an RGBA colour image using a colormap.

    Parameters
    ----------
    data_img : np.ndarray
        2-D array of scalar values to colourise.
    cmap : str, optional
        Matplotlib colormap name. Default is ``'viridis'``.
    gamma : float, optional
        Gamma correction exponent applied after normalising to [0, 1].
        Default is 1.0 (no correction).
    alpha : int, optional
        Alpha (opacity) value in the range [0, 255]. Default is 255 (opaque).

    Returns
    -------
    np.ndarray
        Integer RGBA image array of shape ``(*data_img.shape, 4)`` with
        values in [0, 255].
    """
    assert data_img.ndim == 2
    assert isinstance(alpha, int)

    # convert to float if data_img is not a float
    if not np.issubdtype(data_img.dtype, np.floating):
        data_img = data_img.astype(np.float32)

    color_img = (plt.get_cmap(cmap)(normalize_zero_one(data_img) ** gamma) * 255).astype(int)
    color_img[:, :, -1] = alpha
    return color_img


def int_rescale_image(data_img: np.ndarray, scale: int) -> np.ndarray:
    """Rescale an image by an integer factor using pixel repetition or block averaging.

    Parameters
    ----------
    data_img : np.ndarray
        Input image array.
    scale : int
        Positive integer for upscaling, negative integer (absolute value > 1)
        for downscaling. Values of 1 or -1 return the original image unchanged.

    Returns
    -------
    np.ndarray
        Rescaled image array.
    """
    if scale > 1:
        return int_upscale_image(data_img, scale)
    elif scale < -1:
        return int_downscale_image(data_img, abs(scale))
    else:
        return data_img


def int_upscale_image(data_img: np.ndarray, upscale: int) -> np.ndarray:
    """Upscale an image by repeating each pixel `upscale` times along each spatial axis.

    Parameters
    ----------
    data_img : np.ndarray
        Input image array (at least 2-D).
    upscale : int
        Integer upscaling factor applied to both the first and second axes.

    Returns
    -------
    np.ndarray
        Upscaled image array.
    """
    return np.repeat(np.repeat(data_img, upscale, axis=0), upscale, axis=1)


def int_downscale_image(data_img: np.ndarray, downscale: int) -> np.ndarray:
    """Downscale an image by taking the mean of non-overlapping blocks.

    Parameters
    ----------
    data_img : np.ndarray
        Input image array (at least 2-D). The first two spatial dimensions
        must be divisible by `downscale`.
    downscale : int
        Integer downscaling factor applied to both the first and second axes.

    Returns
    -------
    np.ndarray
        Downscaled image array whose first two dimensions are
        ``image.shape[0] // downscale`` and ``image.shape[1] // downscale``.
    """
    data_img = data_img.copy()
    # data_img = data_img[::downscale][:, ::downscale]
    # Take mean of blocks of size downscale
    data_img = np.mean(
        np.mean(
            data_img.reshape(
                data_img.shape[0] // downscale, downscale, data_img.shape[1] // downscale, downscale, -1,
            ),
            axis=1
        ),
        axis=2
    )

    return data_img
