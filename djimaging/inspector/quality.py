"""Stimulus-aware response-quality filter.

Supported metrics (tables of the ``basic`` block, keyed by the Snippets key = presentation + ROI + preprocessing
(+ ``ch_id`` with several functional channels)):

- chirp (``stim_family='chirp'`` or ``stim_name='chirp'``): ``ChirpQI.qidx``
  = var_t(mean_reps(snippets)) / mean_reps(var_t(snippets)); ``min_qidx`` = 1 / n_reps is shown for reference.
- moving bar (``stim_family='movingbar'`` or ``stim_name='movingbar'``): ``OsDsIndexes.d_qi``, the Baden et al. 2016
  QI per direction (best direction). ``ds_index`` and ``os_index`` are kept as separate, unfiltered metrics.

A threshold keeps ROIs with ``value >= threshold``. ROIs without a value (no row in the QI table, or NaN) are
*missing*, which is reported separately from *below threshold*; both are excluded while the filter is active.
With the filter off every ROI of the mask is eligible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from djimaging.inspector.data import QualityValues

QUALITY_METRICS = {
    'chirp': ('ChirpQI', 'qidx'),
    'movingbar': ('OsDsIndexes', 'd_qi'),
}


def quality_family(stim_name: Optional[str], stim_family: str = '') -> Optional[str]:
    """Return the key of ``QUALITY_METRICS`` for a stimulus, using the same rules as the QI tables' key sources."""
    for family in QUALITY_METRICS:
        if stim_family == family or stim_name == family:
            return family
    return None


def unsupported_reason(stim_name: Optional[str], stim_family: str = '') -> str:
    if stim_name is None:
        return "stimulus not identified (file does not match a Stimulus alias), no quality metric"
    supported = ', '.join(f"{fam} ({tab}.{col})" for fam, (tab, col) in QUALITY_METRICS.items())
    return f"no supported quality metric for stimulus '{stim_name}' (family '{stim_family}'); supported: {supported}"


@dataclass(frozen=True)
class QualityClasses:
    eligible: tuple
    below: tuple
    missing: tuple
    active: bool

    def summary(self) -> str:
        if not self.active:
            return f"filter off: {len(self.eligible)} ROIs ({len(self.missing)} without QI value)"
        return f"{len(self.eligible)} eligible, {len(self.below)} below threshold, {len(self.missing)} missing value"


def classify(roi_ids: Iterable[int], quality: Optional[QualityValues], threshold: Optional[float]) -> QualityClasses:
    """Split ROI IDs into eligible / below threshold / missing value.

    ``threshold=None`` (or no quality values) means filter off: all ROIs are eligible, missing values are still
    counted for information.
    """
    roi_ids = [int(r) for r in roi_ids]
    values = {r: (np.nan if quality is None else quality.value(r)) for r in roi_ids}
    missing = tuple(r for r in roi_ids if not np.isfinite(values[r]))
    if threshold is None or quality is None:
        return QualityClasses(tuple(roi_ids), (), missing, active=False)
    eligible = tuple(r for r in roi_ids if np.isfinite(values[r]) and values[r] >= threshold)
    below = tuple(r for r in roi_ids if np.isfinite(values[r]) and values[r] < threshold)
    return QualityClasses(eligible, below, missing, active=True)
