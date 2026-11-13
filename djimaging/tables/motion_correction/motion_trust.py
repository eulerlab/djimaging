"""
Per-ROI trust over time: can the trace of an ROI be trusted in a time bin / stimulus repeat?

Trust = fraction of the ROI on its cell (xy: residual and uncorrected fast motion, FOV coverage). The plane change
(z) is reported separately per time bin as an approximate |z change| in um (structural channel, field level).
See ``djimaging.utils.motion.trust`` and docs/motion_qc.md.

Example usage:

@schema
class MotionTrust(motion_correction.MotionTrustTemplate):
    motion_roi_mask_table = MotionRoiMask
    stimulus_table = Stimulus

    class Roi(motion_correction.MotionTrustTemplate.Roi):
        motion_roi_mask_table = MotionRoiMask
"""
from __future__ import annotations

from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.dj_storage import load_array
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.motion import compute_trust, fill_artifact_lines, roi_ids_from_mask, trust_per_repeat
from djimaging.utils.motion.image_utils import register_stack
from djimaging.utils.motion.trust import per_repeat, stim_bins
from djimaging.tables.motion_correction.motion_detection import load_presentation_channels


def _none_if_nan(x) -> float | None:
    """NaN (not measurable, e.g. no valid pixels around the ROI) as NULL; DataJoint would otherwise omit it."""
    x = float(x)
    return None if np.isnan(x) else x


class MotionTrustTemplate(dj.Computed):
    """Per-ROI trust (fraction of the ROI on its cell) per time bin and per stimulus repeat, and the field's
    approximate z change. Structure only, not confounded by drugs."""
    database = ""

    @property
    def definition(self) -> str:
        return """
        # Trust of the ROI traces over time (motion QC)
        -> self.motion_roi_mask_table
        ---
        bin_edges : <blob>  # (n_bins + 1,) frame edges of the trust bins
        fast : <blob>  # (n_bins,) max |fast| (uncorrected) motion per bin [px]
        dz_um=null : <blob>  # (n_bins,) approx. |z change| vs. the recording's mean plane [um]; null w/o structural ch.
        rep_starts=null : <blob>  # start times of the stimulus repeats [s], if repeated
        dz_rep=null : <blob>  # (n_repeats,) max dz_um per stimulus repeat
        stim_onset=null : float32  # first trigger [s]; the summaries below only use bins starting after it
        trust_median : float32  # median trust over ROIs and bins after stimulus onset
        frac_low : float32  # fraction of (ROI, bin) after stimulus onset with trust < trust_thresh
        n_rois_flagged : int32  # ROIs with median trust (after stimulus onset) < trust_thresh
        dz_max_um=null : float32  # max dz_um after stimulus onset
        frac_dz_high=null : float32  # fraction of bins after stimulus onset with dz_um > dz_thresh_um
        trust_thresh : float32  # thresholds used for the summaries (MotionParams)
        dz_thresh_um : float32
        """

    class Roi(dj.Part):
        @property
        def definition(self) -> str:
            return """
            # Trust of one ROI
            -> master
            -> self.motion_roi_mask_table.Roi
            ---
            trust : <blob>  # (n_bins,) fraction of the ROI on its cell in [0, 1], all bins incl. pre-stimulus
            trust_rep=null : <blob>  # (n_repeats,) min trust per stimulus repeat
            trust_median : float32  # median trust over the bins after stimulus onset
            res_shift : <blob>  # (n_bins,) |local displacement vs. reference - applied shift| [px]
            coverage : <blob>  # (n_bins,) fraction of the ROI inside the FOV
            corr_self : <blob>  # (n_bins,) diagnostic (not in trust): noise-corrected patch similarity to own mean
            match_corr=null : float32  # noise-corrected similarity of the ROI patch to the field reference (not in trust)
            match_shift_x=null : float32  # remaining displacement of the recording mean vs. reference [px]
            match_shift_y=null : float32
            """

        @property
        @abstractmethod
        def motion_roi_mask_table(self):
            pass

    @property
    @abstractmethod
    def motion_roi_mask_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.motion_roi_mask_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key: dict) -> None:
        mask_tab = self.motion_roi_mask_table
        det = mask_tab.motion_detection_table
        ref_tab = det.motion_reference_table
        params = ref_tab.motion_params_table().get_params(key)
        ref = ref_tab().get_reference(key)
        motion = det().get_motion(key)
        npix, z_edges, z_self_ch1 = (det & key).fetch1('npixartifact_qc', 'z_bin_edges', 'z_self_ch1')
        edges = load_array(z_edges)
        field_mask = mask_tab().get_field_mask(key)
        roi_ids, _, shifts = mask_tab().get_weights(key)
        order = {abs(int(r)): i for i, r in enumerate(roi_ids_from_mask(field_mask))}
        shifts_field_order = np.zeros((len(order), 2))
        for rid, s in zip(roi_ids, shifts):
            shifts_field_order[order[int(rid)]] = s

        data = load_presentation_channels(det.presentation_table, det.userinfo_table, det.raw_params_table, key,
                                          structural=False)
        reg0 = register_stack(fill_artifact_lines(data['ch0'], int(npix)), motion.slow_frames)
        res = compute_trust(
            reg0, ref, field_mask, edges, motion.disp_frames - motion.slow_frames, int(npix),
            zself_ch1=None if z_self_ch1 is None else load_array(z_self_ch1), roi_shifts=shifts_field_order)
        ntrigger_rep = (self.stimulus_table & key).fetch1('ntrigger_rep')
        trust_rep, rep_starts = trust_per_repeat(res.trust, edges, data['fs'], data['triggertimes'], ntrigger_rep)
        dz_rep, _ = per_repeat(res.dz_um, edges, data['fs'], data['triggertimes'], ntrigger_rep, agg=np.max)
        has_z = bool(np.any(np.isfinite(res.dz_um)))

        t_thresh, z_thresh = float(params['trust_thresh']), float(params['dz_thresh_um'])
        sel, onset = stim_bins(edges[:-1], data['fs'], data['triggertimes'])
        trust_stim = res.trust[:, sel]
        roi_median = np.median(trust_stim, axis=1)
        dz_stim = res.dz_um[sel]
        self.insert1(dict(
            key, bin_edges=edges, fast=res.fast.astype(np.float32),
            dz_um=res.dz_um.astype(np.float32) if has_z else None, rep_starts=rep_starts,
            dz_rep=dz_rep.astype(np.float32) if (has_z and dz_rep is not None) else None, stim_onset=onset,
            trust_median=float(np.median(trust_stim)), frac_low=float(np.mean(trust_stim < t_thresh)),
            n_rois_flagged=int(np.sum(roi_median < t_thresh)),
            dz_max_um=float(np.nanmax(dz_stim)) if has_z else None,
            frac_dz_high=float(np.mean(dz_stim > z_thresh)) if has_z else None,
            trust_thresh=t_thresh, dz_thresh_um=z_thresh))
        parts = []
        for i, rid in enumerate(res.roi_ids):
            parts.append(dict(
                key, roi_id=int(abs(rid)), trust=res.trust[i].astype(np.float32),
                trust_rep=None if trust_rep is None else trust_rep[i].astype(np.float32),
                trust_median=float(roi_median[i]), res_shift=res.res_shift[i].astype(np.float32),
                coverage=res.coverage[i].astype(np.float32), corr_self=res.corr_self[i].astype(np.float32),
                match_corr=_none_if_nan(res.match_corr[i]), match_shift_x=_none_if_nan(res.match_shift[i, 0]),
                match_shift_y=_none_if_nan(res.match_shift[i, 1])))
        self.Roi().insert(parts)

    def valid_repeats(self, key: dict, trust_thresh: float | None = None,
                      dz_thresh_um: float | None = None) -> dict:
        """Which stimulus repeats of each ROI to keep: ``{roi_id: bool array (n_repeats,)}``.

        A repeat is kept if at least ``trust_thresh`` of the ROI stays on its cell throughout it (min over its bins) and
        the plane changes by at most ``dz_thresh_um`` (field level; unknown z counts as ok). Defaults: the thresholds
        stored with the entry (MotionParams). Without repeats, the whole recording counts as one repeat (median trust).
        """
        key = get_primary_key(self, key)
        row = (self & key).fetch1()
        t_thresh = row['trust_thresh'] if trust_thresh is None else trust_thresh
        z_thresh = row['dz_thresh_um'] if dz_thresh_um is None else dz_thresh_um
        dz_rep = None if row['dz_rep'] is None else load_array(row['dz_rep'])
        out = {}
        for r in (self.Roi & key).to_dicts(order_by='roi_id'):
            if r['trust_rep'] is None:
                out[r['roi_id']] = np.array([r['trust_median'] >= t_thresh])
                continue
            ok = load_array(r['trust_rep']) >= t_thresh
            if dz_rep is not None:
                ok &= ~(dz_rep > z_thresh)
            out[r['roi_id']] = ok
        return out

    def plot1(self, key: dict | None = None, show_frames: bool = True, frame_times=None, n_spread: int = 3,
              n_worst: int = 4, gamma: float = 0.6) -> None:
        """Trust heatmap (ROI x time), median trust over time and per repeat, and the uncorrected fast motion.

        The pre-stimulus period (before the first trigger, greyed out) does not count in the summaries. With
        ``show_frames``, example frames (raw and registered means of the trust bins, overlaid on the field reference in
        green / magenta, gamma-corrected) are shown below: the bins at ``frame_times`` [s], or ``n_spread`` evenly
        spaced bins plus the ``n_worst`` bins after stimulus onset with the lowest median ROI trust (red). This loads
        the recording.
        """
        from matplotlib import pyplot as plt
        from djimaging.utils.motion.plotting import mark_frames, mark_stimulus
        key = get_primary_key(self, key)
        row = (self & key).fetch1()
        det = self.motion_roi_mask_table.motion_detection_table
        fs = (det.presentation_table.ScanInfo & key).fetch1('scan_frequency')
        triggertimes = load_array((det.presentation_table & key).fetch1('triggertimes'))
        rows = (self.Roi & key).to_dicts(order_by='roi_id')
        trust = np.stack([load_array(r['trust']) for r in rows])
        edges = load_array(row['bin_edges'])
        t = 0.5 * (edges[1:] + edges[:-1]) / fs
        med = np.median(trust, axis=0)

        fig = plt.figure(figsize=(16 if show_frames else 12, 11 if show_frames else 7), layout='constrained')
        sfigs = fig.subfigures(2, 1, height_ratios=[1.5, 1]) if show_frames else [fig]
        axs = sfigs[0].subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
        im = axs[0].imshow(trust, aspect='auto', vmin=0, vmax=1, interpolation='nearest',
                           extent=[0, edges[-1] / fs, len(rows) + 0.5, 0.5])
        sfigs[0].colorbar(im, ax=axs[0], label='trust (fraction of ROI on cell)')
        axs[0].set(ylabel='ROI', title=f"{key}\nafter stimulus onset: median trust {row['trust_median']:.2f}, "
                                       f"{row['n_rois_flagged']} ROIs with median < {row['trust_thresh']:.2f}")
        axs[1].fill_between(t, np.percentile(trust, 10, axis=0), np.percentile(trust, 90, axis=0), color='0.8')
        axs[1].plot(t, med, c='k', label='median ROI trust (10-90 %)')
        axs[1].plot(t, trust.min(axis=0), c='k', lw=0.6, ls=':', label='min ROI trust')
        if row['rep_starts'] is not None and rows[0]['trust_rep'] is not None:
            rep = np.stack([load_array(r['trust_rep']) for r in rows])
            axs[1].step(load_array(row['rep_starts']), np.nanmedian(rep, axis=0), where='post', c='orange',
                        label='per repeat (median ROI)')
        axs[1].axhline(row['trust_thresh'], c='r', lw=0.6, ls='--')
        axs[1].set(ylim=(0, 1.05), xlabel='time [s]', ylabel='trust')
        ax_z = axs[1].twinx()
        ax_z.plot(t, load_array(row['fast']), c='C0', lw=0.8, label='max |fast| [px]')
        if row['dz_um'] is not None:
            ax_z.plot(t, load_array(row['dz_um']), c='m', lw=0.8, label='|dz| approx. [um]')
            ax_z.axhline(row['dz_thresh_um'], c='m', lw=0.6, ls='--')
        ax_z.set(ylabel='fast [px] / dz [um]', ylim=(0, max(6., ax_z.get_ylim()[1])))
        for ax in axs:
            mark_stimulus(ax, triggertimes, label=ax is axs[1])
        h0, l0 = axs[1].get_legend_handles_labels()
        h1, l1 = ax_z.get_legend_handles_labels()
        axs[1].legend(h0 + h1, l0 + l1, fontsize=7, ncol=6, loc='lower left')
        if show_frames:
            idxs, is_worst, labels = det().plot_frames(sfigs[1], key, edges, 1. - med, frame_times=frame_times,
                                                       n_spread=n_spread, n_worst=n_worst, gamma=gamma)
            for ax in axs:
                mark_frames(ax, idxs, edges, fs, labels, is_worst)
        plt.show()
