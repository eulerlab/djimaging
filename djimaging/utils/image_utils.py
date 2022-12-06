from skimage.transform import rotate, resize
import numpy as np


def resize_image(image, output_shape, order=0):
    finite_mask = np.isfinite(image)
    image = image.copy()
    image[~finite_mask] = np.nanmin(image)
    image_resized = resize(
        image, output_shape=output_shape, order=order, mode='constant', anti_aliasing=False)
    mask_resized = resize(
        finite_mask.astype(int), output_shape=output_shape, order=0, mode='constant', anti_aliasing=False)
    assert image_resized.shape == mask_resized.shape
    image_resized[mask_resized == 0] = np.nan
    return image_resized


def rotate_image(image, angle, order=1, cval=None):
    finite_mask = np.isfinite(image)
    image = image.copy()

    minv = np.nanmin(image)
    image[~finite_mask] = minv
    image_rotated = rotate(image, angle=angle, resize=True, order=order, cval=minv if cval is None else cval)
    mask_rotated = rotate(finite_mask.astype(int), angle=angle, resize=True, order=0, cval=0)
    assert image_rotated.shape == mask_rotated.shape
    image_rotated[mask_rotated == 0] = np.nan if cval is None else cval
    return image_rotated


def rescale_image(image, scale, order=0):
    output_shape = np.ceil(np.asarray(image.shape) * scale).astype('int')
    resized_image = resize_image(image, output_shape=output_shape, order=order)
    return resized_image
