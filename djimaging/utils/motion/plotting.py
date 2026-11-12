"""Plot helpers for motion QC: example frames (bin means) of the raw and the registered recording vs. the reference."""
from __future__ import annotations

import string
import warnings

import numpy as np


def pick_frame_bins(score: np.ndarray, n_spread: int = 3, n_worst: int = 4, min_sep: int = 2,
                    candidates: np.ndarray | None = None) -> tuple:
    """Bins to show as example frames: ``n_spread`` evenly spaced bins plus the ``n_worst`` bins with the highest
    ``score`` among ``candidates`` (bool, default all; e.g. only bins after stimulus onset), at least ``min_sep`` bins
    away from bins already picked.

    Returns the sorted bin indices and a boolean array marking the ones picked for their score.
    """
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=-np.inf)
    if candidates is not None:
        score = np.where(np.asarray(candidates, bool), score, -np.inf)
    nb = len(score)
    picked = [int(i) for i in np.unique(np.round(np.linspace(0, nb - 1, n_spread)).astype(int))] if n_spread > 0 else []
    worst = []
    for i in np.argsort(-score, kind='stable'):
        if len(worst) >= n_worst or not np.isfinite(score[i]):
            break
        if all(abs(int(i) - j) >= min_sep for j in picked + worst):
            worst.append(int(i))
    idxs = np.array(sorted(picked + worst), dtype=int)
    return idxs, np.isin(idxs, worst)


def bins_at_times(times, edges: np.ndarray, fs: float) -> np.ndarray:
    """Indices of the bins (frame ``edges``) containing the given times [s]."""
    frames = np.asarray(times, dtype=float) * fs
    return np.clip(np.searchsorted(edges, frames, side='right') - 1, 0, len(edges) - 2)


def bin_mean(stack: np.ndarray, i0: int, i1: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(stack[:, :, int(i0):int(i1)], axis=2)


def norm_gamma(img: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Normalized to [0, 1] (1st / 99.5th percentile), then gamma-corrected (``img ** gamma``); NaN -> 0."""
    finite = np.isfinite(img)
    lo, hi = np.percentile(img[finite], [1, 99.5]) if finite.any() else (0., 1.)
    return np.nan_to_num(np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1)) ** gamma


def roi_outline_segments(roi_mask: np.ndarray) -> list:
    """Pixel-edge line segments around the ROIs of an Igor mask (ROIs < 0, background >= 0), in the (x, y)
    coordinates of ``imshow(img.T, origin='lower')``."""
    lab = np.pad(np.where(roi_mask < 0, -roi_mask, 0).astype(int), 1)
    segs = []
    dx = (lab[1:, :] != lab[:-1, :]) & ((lab[1:, :] > 0) | (lab[:-1, :] > 0))  # between (x - 1, y) and (x, y)
    for x, y in zip(*np.nonzero(dx)):
        segs.append([(x - 0.5, y - 1.5), (x - 0.5, y - 0.5)])
    dy = (lab[:, 1:] != lab[:, :-1]) & ((lab[:, 1:] > 0) | (lab[:, :-1] > 0))
    for x, y in zip(*np.nonzero(dy)):
        segs.append([(x - 1.5, y - 0.5), (x - 0.5, y - 0.5)])
    return segs


def _outline(ax, segs, color: str, lw: float, alpha: float) -> None:
    from matplotlib.collections import LineCollection
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=alpha))


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def show_gray(ax, img: np.ndarray, segs: list | None = None, gamma: float = 0.6) -> None:
    ax.imshow(norm_gamma(img, gamma).T, cmap='gray', origin='lower', interpolation='nearest', vmin=0, vmax=1)
    _outline(ax, segs, 'yellow', 0.6, 0.7)
    _clean(ax)


def show_overlay(ax, img: np.ndarray, ref_img: np.ndarray, segs: list | None = None, gamma: float = 0.6) -> None:
    """Reference in green, ``img`` in magenta: aligned structure is grey / white, displaced structure gets a green
    fringe on one side and a magenta fringe on the other; brightness changes (activity) tint a whole cell."""
    r, m = norm_gamma(ref_img, gamma), norm_gamma(img, gamma)
    ax.imshow(np.stack([m, r, m], axis=-1).transpose(1, 0, 2), origin='lower', interpolation='nearest')
    _outline(ax, segs, 'w', 0.4, 0.5)
    _clean(ax)


def overlay_colorbar(fig, axs) -> None:
    """Colorbar explaining the green / magenta code of ``show_overlay``, right of ``axs``."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list('ref_frame', [(0, 1, 0), (1, 1, 1), (1, 0, 1)])
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap), ax=axs, location='right', shrink=0.8, aspect=20,
                      ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['reference\nonly', 'both\n(aligned)', 'frame\nonly'], fontsize=7)


def frame_labels(n: int) -> list:
    return list(string.ascii_uppercase[:n]) if n <= 26 else [str(i + 1) for i in range(n)]


def plot_motion_frames(fig, raw: np.ndarray, reg: np.ndarray, ref_img: np.ndarray, edges: np.ndarray,
                       idxs: np.ndarray, fs: float, roi_mask: np.ndarray | None = None,
                       is_worst: np.ndarray | None = None, stim_onset: float | None = None,
                       disp: np.ndarray | None = None, gamma: float = 0.6) -> list:
    """Example frames vs. the field reference as a 2-row grid in ``fig`` (a figure or subfigure).

    Rows: raw (uncorrected) bin means and slow-registered bin means (what the traces see), each overlaid on the field
    reference (green: reference, magenta: frame; see ``show_overlay``). First column: the reference with the ROI
    outlines, and the registered recording mean. ``disp`` (n_frames, 2): mean displacement per frame for the titles.
    Frames picked for their score are titled in red.

    Returns the labels of the frames (to mark them on time axes with ``mark_frames``).
    """
    n = len(idxs)
    axs = fig.subplots(2, n + 1, squeeze=False)
    labels = frame_labels(n)
    segs = None if roi_mask is None else roi_outline_segments(roi_mask)
    show_gray(axs[0, 0], ref_img, segs, gamma)
    axs[0, 0].set_title('field reference + ROIs', fontsize=8)
    show_overlay(axs[1, 0], bin_mean(reg, 0, reg.shape[2]), ref_img, segs, gamma)
    axs[1, 0].set_title('registered mean', fontsize=8)
    axs[0, 0].set_ylabel('raw vs. reference', fontsize=8)
    axs[1, 0].set_ylabel('registered vs. reference', fontsize=8)
    for k, b in enumerate(idxs):
        i0, i1 = int(edges[b]), int(edges[b + 1])
        show_overlay(axs[0, k + 1], bin_mean(raw, i0, i1), ref_img, segs, gamma)
        show_overlay(axs[1, k + 1], bin_mean(reg, i0, i1), ref_img, segs, gamma)
        pre = ' (pre-stim)' if stim_onset is not None and i0 / fs < stim_onset else ''
        title = f'{labels[k]}: {i0 / fs:.1f}-{i1 / fs:.1f} s{pre}'
        if disp is not None:
            title += f'\nd=({disp[k][0]:+.2f}, {disp[k][1]:+.2f}) px'
        worst = is_worst is not None and is_worst[k]
        axs[0, k + 1].set_title(title, fontsize=8, color='r' if worst else 'k')
    overlay_colorbar(fig, axs[:, 1:])
    return labels


def mark_frames(ax, idxs: np.ndarray, edges: np.ndarray, fs: float, labels: list,
                is_worst: np.ndarray | None = None) -> None:
    """Shade the example-frame bins on a time axis and label them at the top."""
    for k, b in enumerate(idxs):
        c = 'r' if is_worst is not None and is_worst[k] else 'C2'
        t0, t1 = edges[b] / fs, edges[b + 1] / fs
        ax.axvspan(t0, t1, color=c, alpha=0.2, lw=0, zorder=-4)
        ax.text(0.5 * (t0 + t1), 1.0, labels[k], transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=7, color=c)
    if ax.get_title():  # keep the title clear of the frame labels
        ax.set_title(ax.get_title(), pad=14)


def mark_stimulus(ax, triggertimes, label: bool = False) -> None:
    """Grey out the pre-stimulus period and draw the first and last trigger."""
    tt = np.asarray(triggertimes if triggertimes is not None else [], dtype=float).ravel()
    tt = tt[np.isfinite(tt)]
    if tt.size == 0:
        return
    ax.axvspan(0, tt[0], color='0.6', alpha=0.25, lw=0, zorder=-6)
    ax.axvline(tt[0], c='k', ls='--', lw=1, label='first / last trigger' if label else None)
    ax.axvline(tt[-1], c='k', ls='--', lw=1)
