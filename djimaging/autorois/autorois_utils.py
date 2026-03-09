import warnings
from functools import lru_cache

import numpy as np

try:
    import torch
except ImportError:
    warnings.warn("Could not import torch, AutoROIs will not work.")


@lru_cache(2)
def _get_neighbor_kernel(device: "torch.device") -> "torch.Tensor":
    """Return a 4-connected neighbor kernel (cross-shaped) as a cached tensor.

    Parameters
    ----------
    device : torch.device
        The device on which the kernel tensor will be placed.

    Returns
    -------
    torch.Tensor
        A 4D tensor of shape (1, 1, 3, 3) with the 4-connected neighbor pattern.
    """
    return torch.tensor([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]],
                        dtype=torch.float32,
                        device=device)


@lru_cache(2)
def _get_diagonal_neighbor_kernel(device: "torch.device") -> "torch.Tensor":
    """Return an 8-connected neighbor kernel (full 3x3 except center) as a cached tensor.

    Parameters
    ----------
    device : torch.device
        The device on which the kernel tensor will be placed.

    Returns
    -------
    torch.Tensor
        A 4D tensor of shape (1, 1, 3, 3) with the 8-connected neighbor pattern.
    """
    return torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]],
                        dtype=torch.float32,
                        device=device)


def _run_convolution(mask: "torch.Tensor", kernel: "torch.Tensor") -> "torch.Tensor":
    """Convolve a 2D boolean mask with the given kernel and return a boolean output.

    Parameters
    ----------
    mask : torch.Tensor
        2D boolean tensor to convolve.
    kernel : torch.Tensor
        4D convolution kernel of shape (1, 1, kH, kW).

    Returns
    -------
    torch.Tensor
        2D boolean tensor of the same shape as ``mask``.
    """
    mask_float = mask.to(torch.float32)
    mask4d = mask_float.unsqueeze(dim=0).unsqueeze(dim=0)
    conv_out = torch.nn.functional.conv2d(mask4d, kernel, padding=1)[0, 0]
    assert conv_out.shape == mask.shape
    conv_out_bool = conv_out.to(torch.bool)
    return conv_out_bool


def make_mask_ids_consecutive(mask: np.ndarray) -> np.ndarray:
    """Remap ROI mask values to a consecutive integer range starting at 0.

    Parameters
    ----------
    mask : np.ndarray
        Integer array where 0 is background and positive integers are ROI IDs.
        Must contain 0 as the minimum value (cellpose format).

    Returns
    -------
    np.ndarray
        Mask with the same shape as ``mask`` but with IDs remapped to
        ``{0, 1, 2, ..., n_rois}``.

    Raises
    ------
    AssertionError
        If the minimum value of ``mask`` is not 0.
    """
    values = np.unique(mask)
    assert values.min() == 0, "Mask has to be in cellpose format"
    if (values.max() + 1) == values.shape[0]:
        # consecutive case, nothing to do
        return mask
    else:
        # map all values to a consecutive range
        map_to_consecutive = {old_v: new_v for new_v, old_v in enumerate(sorted(values))}
        clean_mask = np.vectorize(map_to_consecutive.__getitem__)(mask)
        return clean_mask


def create_mask(
        cell_probs: "torch.Tensor",
        offsets: "torch.Tensor",
        center_mask: "torch.Tensor",
        cell_prob_threshold: float = 0.5,
        kernel_size: int = 5,
        center_prob_threshold: float = 0.1,
        max_number_of_cells: int = 99999,
) -> "torch.Tensor":
    """Create a ROI mask from the outputs of the instance UNet model.

    See https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
    for another implementation.

    Parameters
    ----------
    cell_probs : torch.Tensor
        Probability that a pixel contains a cell, shape: (dim_x, dim_y).
    offsets : torch.Tensor
        Offset to the cell center in x and y direction for each pixel,
        shape: (2, dim_x, dim_y).
    center_mask : torch.Tensor
        Probability that a pixel is the center of a cell, shape: (dim_x, dim_y).
    cell_prob_threshold : float, optional
        Threshold above which a pixel is considered a cell. Default is 0.5.
    kernel_size : int, optional
        Max-pooling kernel size used to find center peaks (Section 3.2 of
        https://arxiv.org/pdf/1911.10194.pdf). Default is 5.
    center_prob_threshold : float, optional
        Probability threshold for a pixel to be considered a center.
        Default is 0.1.
    max_number_of_cells : int, optional
        Maximum number of cells per mask. Default is 99999.

    Returns
    -------
    torch.Tensor
        Integer tensor of the same spatial shape as ``cell_probs`` where each
        pixel is labelled with its instance ID (0 = background).
    """
    # determine pixels that are cells
    pixel_is_cell = cell_probs > cell_prob_threshold

    # Determine instance centers
    center_mask_3d = center_mask.unsqueeze(0)
    padding = (kernel_size - 1) // 2
    center_max_pool = torch.nn.functional.max_pool2d(center_mask_3d, kernel_size=kernel_size, stride=1, padding=padding)
    assert center_mask_3d.shape == center_max_pool.shape
    # previous approach relied on boolean logic, but the previous approach didn't support max number of cells
    center_max_pool[center_mask_3d < center_prob_threshold] = -1.0
    center_max_pool[center_mask_3d != center_max_pool] = -1.0
    center_max_pool = center_max_pool.squeeze(0)
    center_max_pool[torch.logical_not(pixel_is_cell)] = -1.0

    pixel_is_center = center_max_pool > 0.0
    center_coords = torch.nonzero(pixel_is_center.squeeze(0))
    if center_coords.size(0) >= max_number_of_cells:
        top_k_scores, _ = torch.topk(torch.flatten(center_max_pool), max_number_of_cells)
        lowest_top_k_score = top_k_scores[-1]
        center_coords = torch.nonzero(center_max_pool >= lowest_top_k_score)
    elif center_coords.size(0) == 0:
        # this can happen in early training iterations, return an empty roi mask
        return torch.zeros_like(cell_probs)

    idx = torch.randperm(center_coords.size()[0])
    center_coords = center_coords[idx]

    # Map pixels to instance centers
    height, width = cell_probs.size()
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)

    center_loc = coord + offsets
    center_loc = center_loc.reshape((2, height * width)).transpose(1, 0)
    # center_coords: [K, 2] -> [K, 1, 2]
    # center_loc = [H*W, 2] -> [1, H*W, 2]
    center_coords = center_coords.unsqueeze(1)
    center_loc = center_loc.unsqueeze(0)
    offset_to_center = center_coords - center_loc
    distance_to_center = torch.norm(offset_to_center, dim=-1)
    instance_id_3d = torch.argmin(distance_to_center, dim=0).reshape((1, height, width)) + 1
    instance_id = instance_id_3d.squeeze(0)
    roi_mask = instance_id * pixel_is_cell

    # clean_roi_mask uses torch.histogram which is only implemented for cpu
    roi_mask = roi_mask.cpu()
    clean_roi_mask(roi_mask)

    roi_mask_consecutive_np = make_mask_ids_consecutive(roi_mask.numpy())
    roi_mask_consecutive = torch.tensor(roi_mask_consecutive_np, dtype=roi_mask.dtype)

    return roi_mask_consecutive


def clean_roi_mask(roi_mask: "torch.Tensor") -> None:
    """Remove isolated pixels and small ROIs from a ROI mask in-place.

    Steps applied:
    (1) Remove pixels of a cell that have no horizontal or vertical neighbour
        belonging to the same cell.
    (2) Remove cells that have fewer than 3 pixels.

    Parameters
    ----------
    roi_mask : torch.Tensor
        Integer 2D tensor where 0 is background and positive integers are ROI
        IDs. Modified in-place.
    """
    # Filter pixels that don't have any horizontal or vertical neighboring pixel
    kernel = _get_neighbor_kernel(roi_mask.device)
    max_cell_id = int(roi_mask.max())
    for cell_id in range(1, max_cell_id + 1):
        cell_mask = (roi_mask == cell_id)
        conv_out_bool = _run_convolution(cell_mask, kernel)
        no_neighbor_pixels = cell_mask != conv_out_bool
        invalid_cell_pixels = torch.logical_and(no_neighbor_pixels, cell_mask)
        roi_mask[invalid_cell_pixels] = 0

    # Filter cells that correspond to less than 0 pixels
    hist = torch.histogram(roi_mask.to(torch.float), bins=int(roi_mask.max() + 1)).hist
    for cell_id, pixel_count in enumerate(hist):
        if cell_id > 0 and pixel_count < 3:
            roi_mask[roi_mask == cell_id] = 0
