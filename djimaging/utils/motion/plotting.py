"""Plot helpers for motion QC: example frames (bin means) of the raw and the registered recording."""
from __future__ import annotations

import string
import warnings

import numpy as np


def pick_frame_bins(score: np.ndarray, n_spread: int = 3, n_worst: int = 4, min_sep: int = 2) -> tuple:
    """Bins to show as example frames: ``n_spread`` evenly spaced bins plus the ``n_worst`` bins with the highest
    ``score`` (at least ``min_sep`` bins away from bins already picked).

    Returns the sorted bin indices and a boolean array marking the ones picked for their score.
    """
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=-np.inf)
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


def _bin_mean(stack: np.ndarray, i0: int, i1: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(stack[:, :, int(i0):int(i1)], axis=2)


def _show(ax, img: np.ndarray, roi_mask: np.ndarray | None) -> None:
    finite = np.isfinite(img)
    vmin, vmax = np.percentile(img[finite], [1, 99.5]) if finite.any() else (0, 1)
    ax.imshow(img.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
    if roi_mask is not None:
        lab = np.abs(roi_mask.astype(int))
        edge = np.zeros(lab.shape, bool)
        edge[1:] |= lab[1:] != lab[:-1]
        edge[:-1] |= lab[1:] != lab[:-1]
        edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
        edge[:, :-1] |= lab[:, 1:] != lab[:, :-1]
        edge &= lab > 0
        overlay = np.zeros(lab.shape + (4,))
        overlay[edge] = (1., 0.3, 0.1, 0.6)
        ax.imshow(overlay.transpose(1, 0, 2), origin='lower', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])


def frame_labels(n: int) -> list:
    return list(string.ascii_uppercase[:n]) if n <= 26 else [str(i + 1) for i in range(n)]


def plot_motion_frames(fig, raw: np.ndarray, reg: np.ndarray, ref_img: np.ndarray, edges: np.ndarray,
                       idxs: np.ndarray, fs: float, roi_mask: np.ndarray | None = None,
                       is_worst: np.ndarray | None = None, stim_onset: float | None = None) -> list:
    """Example frames as a 2-row grid in ``fig`` (a figure or subfigure).

    Top row: field reference, then the raw (uncorrected) bin means; bottom row: registered recording mean, then the
    slow-registered bin means (what the traces see). ROI contours are drawn at the field-mask (reference) position,
    so misalignment is visible in the raw row. Frames picked for their score are titled in red.

    Returns the labels of the frames (to mark them on time axes with ``mark_frames``).
    """
    n = len(idxs)
    axs = fig.subplots(2, n + 1, squeeze=False)
    labels = frame_labels(n)
    _show(axs[0, 0], ref_img, roi_mask)
    axs[0, 0].set_title('field reference', fontsize=8)
    _show(axs[1, 0], _bin_mean(reg, 0, reg.shape[2]), roi_mask)
    axs[1, 0].set_title('registered mean', fontsize=8)
    axs[0, 0].set_ylabel('raw', fontsize=8)
    axs[1, 0].set_ylabel('registered', fontsize=8)
    for k, b in enumerate(idxs):
        i0, i1 = int(edges[b]), int(edges[b + 1])
        _show(axs[0, k + 1], _bin_mean(raw, i0, i1), roi_mask)
        _show(axs[1, k + 1], _bin_mean(reg, i0, i1), roi_mask)
        t0, t1 = i0 / fs, i1 / fs
        pre = ' (pre-stim)' if stim_onset is not None and t0 < stim_onset else ''
        worst = is_worst is not None and is_worst[k]
        axs[0, k + 1].set_title(f'{labels[k]}: {t0:.1f}-{t1:.1f} s{pre}', fontsize=8, color='r' if worst else 'k')
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


def mark_stimulus(ax, triggertimes, label: bool = False) -> None:
    """Grey out the pre-stimulus period and draw the first and last trigger."""
    tt = np.asarray(triggertimes if triggertimes is not None else [], dtype=float).ravel()
    tt = tt[np.isfinite(tt)]
    if tt.size == 0:
        return
    ax.axvspan(0, tt[0], color='0.6', alpha=0.25, lw=0, zorder=-6)
    ax.axvline(tt[0], c='k', ls='--', lw=1, label='first / last trigger' if label else None)
    ax.axvline(tt[-1], c='k', ls='--', lw=1)
