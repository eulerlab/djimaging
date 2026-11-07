"""
Per-ROI trust over time: can the trace of an ROI be trusted in a time bin / stimulus repeat?

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
from djimaging.utils.motion.trust import fast_per_bin
from djimaging.tables.motion_correction.motion_detection import load_presentation_channels


class MotionTrustTemplate(dj.Computed):
    """Per-ROI trust per time bin and per stimulus repeat (structure only, not confounded by drugs)."""
    database = ""

    @property
    def definition(self) -> str:
        return """
        # Trust of the ROI traces over time (motion QC)
        -> self.motion_roi_mask_table
        ---
        bin_edges : <blob>  # (n_bins + 1,) frame edges of the trust bins
        fast : <blob>  # (n_bins,) max |fast| (uncorrected) motion per bin [px]
        zfield : <blob>  # (n_bins,) z factor from the structural channel in [0, 1]
        rep_starts=null : <blob>  # start times of the stimulus repeats [s], if repeated
        trust_median : float32  # median trust over ROIs and bins
        frac_low : float32  # fraction of (ROI, bin) with trust < 0.5
        n_rois_flagged : int32  # ROIs with median trust < 0.5
        """

    class Roi(dj.Part):
        @property
        def definition(self) -> str:
            return """
            # Trust of one ROI
            -> master
            -> self.motion_roi_mask_table.Roi
            ---
            trust : <blob>  # (n_bins,) trust in [0, 1]
            trust_rep=null : <blob>  # (n_repeats,) min trust per stimulus repeat
            trust_median : float32
            corr_self : <blob>  # (n_bins,) noise-corrected patch similarity to the recording's own mean
            res_shift : <blob>  # (n_bins,) |local displacement vs. reference - applied shift| [px]
            coverage : <blob>  # (n_bins,) fraction of the ROI inside the FOV
            match_corr : float32  # noise-corrected similarity of the ROI patch to the field reference (not in trust)
            match_shift_x : float32  # remaining displacement of the recording mean vs. reference [px]
            match_shift_y : float32
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
        field_key = {k: key[k] for k in det.roi_mask_table.primary_key}
        field_mask = load_array((det.roi_mask_table & field_key).fetch1('roi_mask'))
        roi_ids, _, shifts = mask_tab().get_weights(key)
        order = {abs(int(r)): i for i, r in enumerate(roi_ids_from_mask(field_mask))}
        shifts_field_order = np.zeros((len(order), 2))
        for rid, s in zip(roi_ids, shifts):
            shifts_field_order[order[int(rid)]] = s

        data = load_presentation_channels(det.presentation_table, det.userinfo_table, det.raw_params_table, key,
                                          structural=False)
        reg0 = register_stack(fill_artifact_lines(data['ch0'], int(npix)), motion.slow_frames)
        fast = fast_per_bin(motion.fast_bins, motion.bin_centers, edges)
        res = compute_trust(
            reg0, ref, field_mask, edges, fast, int(npix),
            zself_ch1=None if z_self_ch1 is None else load_array(z_self_ch1), roi_shifts=shifts_field_order,
            corr_lo=params['corr_lo'], corr_hi=params['corr_hi'], res_scale=params['res_scale'],
            fast_scale=params['fast_scale'], z_lo=params['z_lo'], z_hi=params['z_hi'])
        ntrigger_rep = (self.stimulus_table & key).fetch1('ntrigger_rep')
        trust_rep, rep_starts = trust_per_repeat(res.trust, edges, data['fs'], data['triggertimes'], ntrigger_rep)

        roi_median = np.median(res.trust, axis=1)
        self.insert1(dict(key, bin_edges=edges, fast=res.fast.astype(np.float32), zfield=res.zfield.astype(np.float32),
                          rep_starts=rep_starts, trust_median=float(np.median(res.trust)),
                          frac_low=float(np.mean(res.trust < 0.5)), n_rois_flagged=int(np.sum(roi_median < 0.5))))
        parts = []
        for i, rid in enumerate(res.roi_ids):
            parts.append(dict(
                key, roi_id=int(abs(rid)), trust=res.trust[i].astype(np.float32),
                trust_rep=None if trust_rep is None else trust_rep[i].astype(np.float32),
                trust_median=float(roi_median[i]), corr_self=res.corr_self[i].astype(np.float32),
                res_shift=res.res_shift[i].astype(np.float32), coverage=res.coverage[i].astype(np.float32),
                match_corr=float(res.match_corr[i]), match_shift_x=float(res.match_shift[i, 0]),
                match_shift_y=float(res.match_shift[i, 1])))
        self.Roi().insert(parts)

    def plot1(self, key: dict | None = None) -> None:
        """Trust heatmap (ROI x time), median trust over time and per repeat, and the uncorrected fast motion."""
        from matplotlib import pyplot as plt
        key = get_primary_key(self, key)
        row = (self & key).fetch1()
        fs = (self.motion_roi_mask_table.motion_detection_table.presentation_table.ScanInfo & key).fetch1(
            'scan_frequency')
        rows = (self.Roi & key).to_dicts(order_by='roi_id')
        trust = np.stack([load_array(r['trust']) for r in rows])
        edges = load_array(row['bin_edges'])
        t = 0.5 * (edges[1:] + edges[:-1]) / fs

        fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
        im = axs[0].imshow(trust, aspect='auto', vmin=0, vmax=1, interpolation='nearest',
                           extent=[0, edges[-1] / fs, len(rows) + 0.5, 0.5])
        fig.colorbar(im, ax=axs[0], label='trust')
        axs[0].set(ylabel='ROI', title=f"{key}\nmedian trust {row['trust_median']:.2f}, "
                                       f"{row['n_rois_flagged']} ROIs flagged")
        axs[1].fill_between(t, np.percentile(trust, 10, axis=0), np.percentile(trust, 90, axis=0), color='0.8')
        axs[1].plot(t, np.median(trust, axis=0), c='k', label='median ROI trust (10-90 %)')
        if row['rep_starts'] is not None and rows[0]['trust_rep'] is not None:
            rep = np.stack([load_array(r['trust_rep']) for r in rows])
            axs[1].step(load_array(row['rep_starts']), np.nanmedian(rep, axis=0), where='post', c='orange',
                        label='per repeat (median)')
        axs[1].plot(t, load_array(row['zfield']), c='m', lw=0.8, label='z factor (structural)')
        axs[1].plot(t, np.exp(-0.5 * (load_array(row['fast']) / 0.75) ** 2), c='C0', lw=0.8, label='fast-motion factor')
        axs[1].set(ylim=(0, 1.05), xlabel='time [s]', ylabel='trust')
        axs[1].legend(fontsize=7, ncol=4, loc='lower left')
        plt.tight_layout()
        plt.show()
