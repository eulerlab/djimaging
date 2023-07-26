import warnings
from functools import lru_cache

import numpy as np

try:
    import torch
except ImportError:
    warnings.warn("Could not import torch, AutoROIs will not work.")


@lru_cache(2)
def _get_neighbor_kernel(device: torch.device):
    return torch.tensor([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]],
                        dtype=torch.float32,
                        device=device)


@lru_cache(2)
def _get_diagonal_neighbor_kernel(device: torch.device):
    return torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]],
                        dtype=torch.float32,
                        device=device)


def _run_convolution(mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(torch.float32)
    mask4d = mask_float.unsqueeze(dim=0).unsqueeze(dim=0)
    conv_out = torch.nn.functional.conv2d(mask4d, kernel, padding=1)[0, 0]
    assert conv_out.shape == mask.shape
    conv_out_bool = conv_out.to(torch.bool)
    return conv_out_bool


def make_mask_ids_consecutive(mask: np.array) -> np.array:
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
        cell_probs: torch.Tensor,
        offsets: torch.Tensor,
        center_mask: torch.Tensor,
        cell_prob_threshold: float = 0.5,
        kernel_size: int = 5,
        center_prob_threshold: float = 0.1,
        max_number_of_cells: int = 99999,
) -> torch.Tensor:
    """
    Create a roi mask from the outputs of the instance unet model.
    See https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
    for another implementation

    Args:
        cell_probs: probability that a pixel contains a cell, shape: [dim_x, dim_y]
        offsets: offset to the cell center in x and y direction for each pixel, shape [2, dim_x, dim_y]
        center_mask: 'probability' that a pixel is the center of the cell, shape: [dim_x, dim_y]
        cell_prob_threshold: threshold that we consider a pixel to be a cell
        kernel_size: mentioned parameter in section 3.2 of https://arxiv.org/pdf/1911.10194.pdf, changed to 5
        center_prob_threshold: probability threshold that to determine whether a pixel is a center
        max_number_of_cells: maximum number of cells per mask
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


def clean_roi_mask(roi_mask: torch.Tensor) -> torch:
    """"
    (1) filter pixel of a cell that don't have a horizontal and vertical neighboring pixel
    (2) throw out cells that have less than 3 pixels
    Possible Todo: throw out pixels of a cell that are horizontal or vertical neighbors of another cell
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
