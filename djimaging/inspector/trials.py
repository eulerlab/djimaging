"""Trial-aligned summaries for the "Trial mean ± SD" panel of the recording inspector.

Alignment is the pipeline's: every ROI trace is cut with ``split_trace_by_reps`` exactly like ``SnippetsTemplate.make``
(``ntrigger_rep``, ``delay = stim_dict['trigger_delay']``, ``allow_drop_last=True``, ``pad_trace = Snippets._pad_trace``;
see ``data.StimulusInfo``). Column ``k`` of the snippets is trial ``k``, which starts at trigger
``k * ntrigger_rep``. Trial-relative times are ``snippet times - onset trigger`` (as ``Snippets.plot``), averaged over
the trials of a summary (as ``Averages`` / the CellViz trial average do).

Conditions: with ``ntrigger_rep == 1`` and a ``trial_info`` (moving bar) trial ``k`` belongs to condition
``k % len(trial_info)`` (the order ``preprocess_mb_snippets`` / ``get_dir_idx`` assume), and every summary uses only
the repeats of one condition, so directions are never pooled. Otherwise all repeats form one condition (chirp).

Summaries: mean and SD (``ddof=1``) across the valid repeats (complete and finite snippets) of a condition. For the
eligible population, the ROIs are averaged within each repeat first and the SD is taken across these per-repeat
population means. With a single valid repeat there is no SD.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Number
from typing import Iterable, Optional

import numpy as np

from djimaging.inspector.data import PresentationData, StimulusInfo, Trace


def condition_label(name) -> str:
    """Short label of a trial condition: numeric names are directions in degrees (moving bar ``trial_info``)."""
    if isinstance(name, Number) and not isinstance(name, bool):
        return f"{float(name):g}°"
    return str(name)


@dataclass(frozen=True)
class TrialLayout:
    """Trials of one presentation on the recording clock (from the triggers and the stimulus, no traces needed)."""
    onsets: np.ndarray            # onset trigger time of trial k (= triggers[k * ntrigger_rep])
    condition: np.ndarray         # condition index of trial k
    repeat: np.ndarray            # repeat number of trial k within its condition (0-based)
    condition_names: tuple        # ('',) for whole repetitions, e.g. ('0°', '180°', ...) for the moving bar

    @property
    def n_conditions(self) -> int:
        return len(self.condition_names)

    def trial_before(self, t: float) -> int:
        """Index of the latest trial starting before ``t``; -1 if ``t`` is before the first onset."""
        return int(np.searchsorted(self.onsets, t, side='left')) - 1


def trial_layout(triggers: np.ndarray, stim: Optional[StimulusInfo]) -> Optional[TrialLayout]:
    """``None`` if the presentation has no repeat structure (not repeated, no triggers, unknown stimulus)."""
    triggers = np.asarray(triggers, dtype=float).ravel()
    if stim is None or not stim.isrepeated or int(stim.ntrigger_rep) < 1 or triggers.size == 0:
        return None
    onsets = triggers[::int(stim.ntrigger_rep)]
    n = onsets.size
    conditions = stim.conditions
    if conditions:
        n_cond = len(conditions)
        return TrialLayout(onsets, np.arange(n) % n_cond, np.arange(n) // n_cond,
                           tuple(condition_label(c) for c in conditions))
    return TrialLayout(onsets, np.zeros(n, dtype=int), np.arange(n), ('',))


@dataclass(frozen=True)
class Snippets:
    """Trial-aligned samples of one ROI (or of the population): columns are trials."""
    values: np.ndarray            # (n_samples, n_trials)
    rel_times: np.ndarray         # (n_samples, n_trials), seconds from the trial's onset trigger
    valid: np.ndarray             # (n_trials,), complete and finite


def roi_snippets(trace: Trace, triggers: np.ndarray, stim: StimulusInfo) -> Snippets:
    """Cut one trace into trials with the pipeline function and parameters of ``SnippetsTemplate.make``."""
    from djimaging.utils.snippet_utils import split_trace_by_reps
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        values, times, trig, _ = split_trace_by_reps(
            trace.values, trace.times, np.asarray(triggers, dtype=float), int(stim.ntrigger_rep),
            delay=float(stim.trigger_delay or 0.), allow_drop_last=True, pad_trace=bool(stim.pad_trace))
    return Snippets(values, times - trig[0][None, :], np.all(np.isfinite(values), axis=0))


def population_snippets(parts: Iterable[Snippets]) -> Optional[Snippets]:
    """Average ROIs within each trial (only ROIs whose snippet of that trial is valid), truncated to the shortest
    snippet length. A trial is valid if at least one ROI contributes to it."""
    parts = [p for p in parts if p.values.size]
    if not parts:
        return None
    n_s = min(p.values.shape[0] for p in parts)
    n_t = max(p.values.shape[1] for p in parts)
    total = np.zeros((n_s, n_t))
    rel = np.zeros((n_s, n_t))
    count = np.zeros(n_t)
    for p in parts:
        k = np.flatnonzero(p.valid)
        total[:, k] += p.values[:n_s, k]
        rel[:, k] += p.rel_times[:n_s, k]
        count[k] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        return Snippets(total / count, rel / count, count > 0)


@dataclass(frozen=True)
class ConditionSummary:
    rel_times: np.ndarray         # (n_samples,), mean trial-relative time
    mean: np.ndarray
    sd: Optional[np.ndarray]      # None with fewer than 2 repeats
    n: int                        # number of valid repeats


@dataclass(frozen=True)
class TrialSummary:
    """Per condition (index of ``TrialLayout.condition_names``): summary, or None without valid repeat."""
    conditions: tuple

    def value_range(self, offset: float = 0.) -> tuple:
        lo, hi = [], []
        for c in self.conditions:
            if c is None:
                continue
            band = (c.mean - c.sd, c.mean + c.sd) if c.sd is not None else (c.mean, c.mean)
            lo.append(np.nanmin(band[0]))
            hi.append(np.nanmax(band[1]))
        return (min(lo) + offset, max(hi) + offset) if lo else (np.nan, np.nan)

    @property
    def t_range(self) -> tuple:
        ts = [c.rel_times for c in self.conditions if c is not None]
        return (min(t[0] for t in ts), max(t[-1] for t in ts)) if ts else (0., 1.)


def summarize(snippets: Snippets, layout: TrialLayout) -> TrialSummary:
    """Mean and SD (ddof=1) over the valid repeats of each condition; conditions are never pooled."""
    n_trials = min(snippets.values.shape[1], layout.condition.size)
    out = []
    for c in range(layout.n_conditions):
        cols = [k for k in range(n_trials) if layout.condition[k] == c and snippets.valid[k]]
        if not cols:
            out.append(None)
            continue
        x = snippets.values[:, cols]
        out.append(ConditionSummary(rel_times=snippets.rel_times[:, cols].mean(axis=1), mean=x.mean(axis=1),
                                    sd=x.std(axis=1, ddof=1) if len(cols) > 1 else None, n=len(cols)))
    return TrialSummary(tuple(out))


class TrialCache:
    """Snippets and summaries of one (presentation, signal), computed on first use and kept across frames.

    Snippets are cached per ROI, summaries per ROI and per eligible population (tuple of ROI IDs).
    """

    def __init__(self, pres: PresentationData, signal: Optional[str]):
        self.key = (id(pres), signal)
        self.pres = pres
        self.layout = trial_layout(pres.triggers, pres.stimulus)
        self.signal = pres.signals.get(signal) if signal else None
        self._snippets, self._errors, self._summaries = {}, {}, {}

    @property
    def available(self) -> bool:
        return self.layout is not None and self.signal is not None

    def unavailable_reason(self) -> str:
        if self.signal is None:
            return 'no traces'
        stim = self.pres.stimulus
        if stim is None:
            return 'stimulus unknown: no repeat structure'
        if not stim.isrepeated:
            return 'stimulus is not repeated'
        return 'no triggers'

    def snippets(self, roi_id: int) -> Optional[Snippets]:
        roi_id = int(roi_id)
        if roi_id not in self._snippets:
            trace = self.signal.traces.get(roi_id) if self.signal is not None else None
            snip = None
            if trace is not None and self.layout is not None:
                try:
                    snip = roi_snippets(trace, self.pres.triggers, self.pres.stimulus)
                except Exception as e:  # noqa: BLE001 (reported per ROI, the panel stays usable)
                    self._errors[roi_id] = f"{type(e).__name__}: {e}"
            self._snippets[roi_id] = snip
        return self._snippets[roi_id]

    def error(self, roi_id: int) -> str:
        return self._errors.get(int(roi_id), '')

    def roi_summary(self, roi_id: int) -> Optional[TrialSummary]:
        key = ('roi', int(roi_id))
        if key not in self._summaries:
            snip = self.snippets(roi_id)
            self._summaries[key] = None if snip is None else summarize(snip, self.layout)
        return self._summaries[key]

    def population_summary(self, roi_ids: Iterable[int]) -> Optional[TrialSummary]:
        ids = tuple(int(r) for r in roi_ids)
        key = ('population', ids)
        if key not in self._summaries:
            pop = population_snippets(s for s in (self.snippets(r) for r in ids) if s is not None)
            self._summaries[key] = None if pop is None else summarize(pop, self.layout)
            while sum(k[0] == 'population' for k in self._summaries) > 8:  # keep a few recent eligible sets
                del self._summaries[next(k for k in self._summaries if k[0] == 'population')]
        return self._summaries[key]
