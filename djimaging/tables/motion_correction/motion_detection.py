"""
Motion correction and motion QC for low-resolution two-photon recordings (method and validation: docs/motion_qc.md).

Tables:
- MotionParams: parameters.
- MotionReference: field reference images from the presentation the field ROI mask was drawn on.
- MotionDetection: per presentation: rigid subpixel displacement (slow part is corrected, fast part flagged),
  relative z proxy, and per-ROI displacements vs. the reference.
- RefinedRoiMask (refined_roi_mask.py): field ROI mask with borders refined (+-1 px) on the registered mean of the
  reference condition; ROI ids kept.
- MotionRoiMask (motion_roi_mask.py): presentation ROI weights (per-ROI between-presentation shift + mild reshape).
  In motion mode this replaces RoiMask.RoiMaskPresentation.
- MotionTrust (motion_trust.py): per-ROI trust over time and per stimulus repeat.

Example usage (see djimaging/schemas/blocks/motion.py):

@schema
class MotionParams(motion_correction.MotionParamsTemplate):
    pass


@schema
class MotionReference(motion_correction.MotionReferenceTemplate):
    motion_params_table = MotionParams
    roi_mask_table = RoiMask
    presentation_table = Presentation
    userinfo_table = UserInfo
    raw_params_table = RawDataParams


@schema
class MotionDetection(motion_correction.MotionDetectionTemplate):
    motion_reference_table = MotionReference
    roi_mask_table = RoiMask
    presentation_table = Presentation
    userinfo_table = UserInfo
    raw_params_table = RawDataParams
"""
from __future__ import annotations

from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.dj_storage import load_array, local_path
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.motion import clean_channels, compute_zproxy, estimate_motion, estimate_roi_shifts, make_reference
from djimaging.utils.motion.image_utils import register_stack
from djimaging.utils.motion.registration import MotionEstimate
from djimaging.utils.motion.trust import stim_bins
from djimaging.utils.scanm import read_utils

DEFAULT_MOTION_PARAMS = dict(
    # registration
    shift_bin_frames=4,  # frames per displacement estimate (~0.5 s at 7.8 Hz)
    tau_slow_s=10.,  # smoothing of the displacement; only the slow part is corrected
    split_at_onset=True,  # smooth before / after the first trigger separately (focus adjusted before the stimulus)
    max_shift=8,  # search window of the rigid registration [px]
    whiten=0.5,  # spectral whitening exponent (0: cross-correlation, 1: phase correlation)
    smooth_sigma=1.,  # smoothing of the correlation surface [px]
    upsample=20,  # subpixel resolution 1/upsample px
    clean_structural=True,  # remove light artifact / indicator bleed-through from the structural channel
    # QC bins
    trust_bin_frames=16,  # frames per trust / z-proxy bin (~2 s)
    fast_thresh=0.5,  # |fast motion| [px] counted as fast-motion bin in the summary
    # between-presentation ROI shifts
    roi_max_shift=3,  # search window of the per-ROI shifts [px]
    pooling='cond1',  # 'cond1': pool presentations of the same condition; 'presentation': each on its own
    local_sigma=4.,  # smoothness of the displacement field [px]
    lam_single=0.5,  # shrinkage of per-ROI deviations if only one presentation is pooled
    # refinement of the field mask on the registered mean of the reference condition (RefinedRoiMask)
    refine_mask=True,  # False: the field mask is used as drawn
    refine_max_px=1,  # ROI borders may move by at most this many pixels (in and out)
    refine_rel_thresh=0.5,  # border pixels are part of the cell if (img - bg) / (peak - bg) >= this
    refine_min_contrast=0.5,  # ROIs dimmer than this (vs. surround, in robust SDs of the image) kept as drawn
    refine_min_keep_frac=0.5,  # ROIs that would keep fewer pixels are kept as drawn
    refine_grow=True,  # border pixels outside the drawn ROI may be added
    # presentation mask reshape
    rel_thresh=0.2, rel_drop=0.3, min_valid_frac=0.95, min_contrast_frac=0.5,
    # trust
    corr_lo=0.5, corr_hi=0.85, res_scale=0.75, fast_scale=0.75, z_lo=0.75, z_hi=0.95,
)


def motion_channel_names(userinfo_table, key: dict) -> tuple[str, str | None]:
    """Functional channel (first functional channel) and structural channel (first other channel, e.g. SR101)."""
    functional = userinfo_table().get_functional_ch_names(key)
    others = [ch for ch in userinfo_table().get_ch_names(key) if ch not in functional]
    return functional[0], (others[0] if others else None)


def load_presentation_channels(presentation_table, userinfo_table, raw_params_table, key: dict,
                               structural: bool = True) -> dict:
    """Functional and structural stacks (x, y, t) of a presentation plus frame rate, artifact lines and triggers."""
    pres_data_file, npixartifact, triggertimes = (presentation_table & key).fetch1(
        'pres_data_file', 'npixartifact', 'triggertimes')
    fs = (presentation_table.ScanInfo & key).fetch1('scan_frequency')
    from_raw_data = (raw_params_table & key).fetch1('from_raw_data')
    func_name, struct_name = motion_channel_names(userinfo_table, key)
    if not structural:
        struct_name = None
    ch_names = (func_name,) if struct_name is None else (func_name, struct_name)
    with local_path(pres_data_file, presentation_table._filepath_store) as filepath:
        stacks, _ = read_utils.load_stacks(filepath, from_raw_data=from_raw_data, ch_names=ch_names)
    return dict(ch0=np.asarray(stacks[func_name], dtype=np.float32),
                ch1=None if struct_name is None else np.asarray(stacks[struct_name], dtype=np.float32),
                fs=float(fs), npixartifact=int(npixartifact), triggertimes=load_array(triggertimes))


def reference_pres_key(roi_mask_table, presentation_table, key: dict) -> dict:
    """Primary key of the presentation the field ROI mask was drawn on (the motion reference)."""
    row = (roi_mask_table & key).fetch1()
    return {k: row[k] for k in presentation_table.primary_key}


class MotionParamsTemplate(dj.Lookup):
    """Parameters of motion correction and motion QC, see ``DEFAULT_MOTION_PARAMS``."""
    database = ""

    @property
    def definition(self) -> str:
        return """
        motion_params_id : int32
        ---
        motion_params : <blob>  # dict, missing entries default to DEFAULT_MOTION_PARAMS
        """

    def add(self, motion_params_id: int = 1, skip_duplicates: bool = False, **params) -> None:
        """Insert a parameter set: the defaults (``DEFAULT_MOTION_PARAMS``) updated by ``params``."""
        unknown = set(params) - set(DEFAULT_MOTION_PARAMS)
        if unknown:
            raise ValueError(f'Unknown motion parameters: {sorted(unknown)}. Valid: {sorted(DEFAULT_MOTION_PARAMS)}')
        if params.get('pooling', 'cond1') not in ('cond1', 'presentation'):
            raise ValueError(f"pooling={params['pooling']!r} must be 'cond1' or 'presentation'")
        self.insert1(dict(motion_params_id=motion_params_id, motion_params={**DEFAULT_MOTION_PARAMS, **params}),
                     skip_duplicates=skip_duplicates)

    def get_params(self, key: dict) -> dict:
        return {**DEFAULT_MOTION_PARAMS, **(self & key).fetch1('motion_params')}


class MotionReferenceTemplate(dj.Computed):
    """Field reference images (registered mean of the presentation the field ROI mask was drawn on)."""
    database = ""

    @property
    def definition(self) -> str:
        return """
        # Motion reference of a field: registered mean images of the ROI-mask presentation
        -> self.roi_mask_table
        -> self.motion_params_table
        ---
        ref_ch0 : <blob>  # reference image of the functional channel
        ref_ch0_a : <blob>  # mean of the even frames (for noise-corrected similarities)
        ref_ch0_b : <blob>  # mean of the odd frames
        ref_ch1=null : <blob>  # reference image of the structural channel (cleaned), if available
        ref_ch1_a=null : <blob>
        ref_ch1_b=null : <blob>
        art_mask=null : <blob>  # structural-channel pixels contaminated by the light artifact (excluded)
        npixartifact_qc : int32  # light-artifact lines used for QC (may be wider than npixartifact)
        k_bleed : float32  # bleed-through of the functional into the structural channel
        """

    @property
    @abstractmethod
    def motion_params_table(self):
        pass

    @property
    @abstractmethod
    def roi_mask_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.roi_mask_table.proj() * self.motion_params_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key: dict) -> None:
        params = self.motion_params_table().get_params(key)
        pres_key = {**key, **reference_pres_key(self.roi_mask_table, self.presentation_table, key)}
        data = load_presentation_channels(self.presentation_table, self.userinfo_table, self.raw_params_table,
                                          pres_key, structural=True)
        ch0, ch1, info = clean_channels(data['ch0'], data['ch1'], data['npixartifact']) \
            if params['clean_structural'] else _fill_only(data)
        ref = make_reference(ch0, ch1, info['npixartifact'], info['art_mask'], shift_bin_frames=params['shift_bin_frames'],
                             **_xcorr_kws(params))
        self.insert1(dict(
            key, ref_ch0=ref['ch0'].astype(np.float32), ref_ch0_a=ref['ch0_halves'][0].astype(np.float32),
            ref_ch0_b=ref['ch0_halves'][1].astype(np.float32),
            ref_ch1=None if ref['ch1'] is None else ref['ch1'].astype(np.float32),
            ref_ch1_a=None if ref['ch1'] is None else ref['ch1_halves'][0].astype(np.float32),
            ref_ch1_b=None if ref['ch1'] is None else ref['ch1_halves'][1].astype(np.float32),
            art_mask=info['art_mask'], npixartifact_qc=info['npixartifact'], k_bleed=info['k_bleed']))

    def get_reference(self, key: dict) -> dict:
        """Reference as used by ``djimaging.utils.motion`` functions."""
        r = (self & key).fetch1()
        ch1 = None if r['ref_ch1'] is None else load_array(r['ref_ch1'])
        return dict(ch0=load_array(r['ref_ch0']), ch1=ch1,
                    ch0_halves=(load_array(r['ref_ch0_a']), load_array(r['ref_ch0_b'])),
                    ch1_halves=None if ch1 is None else (load_array(r['ref_ch1_a']), load_array(r['ref_ch1_b'])),
                    art_mask=None if r['art_mask'] is None else load_array(r['art_mask']).astype(bool),
                    npixartifact=int(r['npixartifact_qc']))

    def plot1(self, key: dict | None = None) -> None:
        from matplotlib import pyplot as plt
        key = get_primary_key(self, key)
        ref = self.get_reference(key)
        imgs = [('functional', ref['ch0'])] + ([('structural', ref['ch1'])] if ref['ch1'] is not None else [])
        fig, axs = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4), squeeze=False)
        for ax, (name, img) in zip(axs[0], imgs):
            ax.imshow(img.T, cmap='gray', origin='lower')
            ax.set_title(f'reference {name}')
        if ref['art_mask'] is not None and ref['ch1'] is not None:
            axs[0, -1].contour(ref['art_mask'].T, levels=[0.5], colors='r', origin='lower')
        fig.suptitle(str(key), fontsize=8)
        plt.tight_layout()
        plt.show()


def _xcorr_kws(params: dict) -> dict:
    return dict(max_shift=params['max_shift'], whiten=params['whiten'], smooth_sigma=params['smooth_sigma'],
                upsample=params['upsample'])


def _fill_only(data: dict):
    from djimaging.utils.motion import fill_artifact_lines
    ch1 = data['ch1']
    return (fill_artifact_lines(data['ch0'], data['npixartifact']),
            None if ch1 is None else fill_artifact_lines(ch1, data['npixartifact']),
            dict(npixartifact=data['npixartifact'], k_bleed=0., art_mask=None))


class MotionDetectionTemplate(dj.Computed):
    """Per presentation: displacement vs. the field reference, z proxy and per-ROI displacements."""
    database = ""

    @property
    def definition(self) -> str:
        return """
        # Motion of a presentation relative to the field reference
        -> self.presentation_table
        -> self.motion_reference_table
        ---
        bin_frames : int32  # frames per displacement estimate
        bin_centers : <blob>  # (n_bins,) frame index of the bin centers
        disp_bins : <blob>  # (n_bins, 2) displacement vs. reference [px]; content at p is at p + disp
        slow_bins : <blob>  # (n_bins, 2) smoothed displacement, this part is corrected
        n_frames : int32  # number of frames
        stim_onset=null : float32  # first trigger [s]; the summaries below only use bins starting after it
        slow_range_px : float32  # max range of the slow displacement after stimulus onset [px]
        fast_rms_px : float32  # rms of the fast (uncorrected) displacement after stimulus onset [px]
        fast_frac : float32  # fraction of bins after stimulus onset with |fast| > fast_thresh
        npixartifact_qc : int32  # light-artifact lines used for QC
        k_bleed : float32  # bleed-through of the functional into the structural channel
        z_bin_edges : <blob>  # (n_zbins + 1,) frame edges of the z-proxy bins
        z_self_ch0 : <blob>  # noise-corrected similarity to the recording's own mean, functional channel
        z_ref_ch0 : <blob>  # same, to the field reference
        z_rec_ref_ch0 : float32  # whole recording vs. field reference, functional channel
        z_self_ch1=null : <blob>  # same for the structural channel
        z_ref_ch1=null : <blob>
        z_rec_ref_ch1=null : float32
        roi_shifts_meas : <blob>  # (n_rois, 2) per-ROI displacement vs. reference after global correction [px]
        roi_shift_quality : <blob>  # (n_rois,) peak correlation of the per-ROI match
        """

    @property
    @abstractmethod
    def motion_reference_table(self):
        pass

    @property
    @abstractmethod
    def roi_mask_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.presentation_table.proj() * self.motion_reference_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key: dict) -> None:
        params = self.motion_reference_table.motion_params_table().get_params(key)
        ref = self.motion_reference_table().get_reference(key)
        data = load_presentation_channels(self.presentation_table, self.userinfo_table, self.raw_params_table, key,
                                          structural=ref['ch1'] is not None)
        ch0, ch1, info = clean_channels(data['ch0'], data['ch1'], data['npixartifact']) \
            if params['clean_structural'] else _fill_only(data)
        npix = max(info['npixartifact'], ref['npixartifact'])
        onset = stim_bins([], data['fs'], data['triggertimes'])[1]
        motion = estimate_motion(ch0, ch1, ref, npix, data['fs'], art_mask=info['art_mask'],
                                 bin_frames=params['shift_bin_frames'], tau_slow_s=params['tau_slow_s'],
                                 split_s=onset if params['split_at_onset'] else None, **_xcorr_kws(params))
        zp = compute_zproxy(ch0, ch1, ref, motion.disp_frames, npix, art_mask=info['art_mask'],
                            bin_frames=params['trust_bin_frames'])
        field_key = {k: key[k] for k in self.roi_mask_table.primary_key}  # field mask: not restricted by presentation
        field_mask = load_array((self.roi_mask_table & field_key).fetch1('roi_mask'))
        reg0 = register_stack(ch0, motion.slow_frames)
        roi_shifts, quality = estimate_roi_shifts(np.nanmean(reg0, axis=2), ref['ch0'], field_mask,
                                                  max_shift=params['roi_max_shift'])
        fast_abs = np.linalg.norm(motion.fast_bins, axis=1)
        sel, onset = stim_bins(motion.bin_centers - (motion.bin_frames - 1) / 2., data['fs'], data['triggertimes'])
        self.insert1(dict(
            key, bin_frames=motion.bin_frames, bin_centers=motion.bin_centers.astype(np.float32),
            disp_bins=motion.disp_bins.astype(np.float32), slow_bins=motion.slow_bins.astype(np.float32),
            n_frames=motion.n_frames, stim_onset=onset,
            slow_range_px=float(np.max(np.ptp(motion.slow_bins[sel], axis=0))),
            fast_rms_px=float(np.sqrt(np.mean(fast_abs[sel] ** 2))),
            fast_frac=float(np.mean(fast_abs[sel] > params['fast_thresh'])),
            npixartifact_qc=npix, k_bleed=info['k_bleed'], z_bin_edges=zp['bin_edges'],
            z_self_ch0=zp['self_ch0'].astype(np.float32), z_ref_ch0=zp['ref_ch0'].astype(np.float32),
            z_rec_ref_ch0=zp['rec_ref_ch0'],
            z_self_ch1=None if zp['self_ch1'] is None else zp['self_ch1'].astype(np.float32),
            z_ref_ch1=None if zp['ref_ch1'] is None else zp['ref_ch1'].astype(np.float32),
            z_rec_ref_ch1=zp['rec_ref_ch1'],
            roi_shifts_meas=roi_shifts.astype(np.float32), roi_shift_quality=quality.astype(np.float32)))

    def get_motion(self, key: dict) -> MotionEstimate:
        bin_frames, centers, disp, slow, n_frames = (self & key).fetch1(
            'bin_frames', 'bin_centers', 'disp_bins', 'slow_bins', 'n_frames')
        return MotionEstimate(bin_frames=int(bin_frames), bin_centers=load_array(centers),
                              disp_bins=load_array(disp).astype(float), slow_bins=load_array(slow).astype(float),
                              n_frames=int(n_frames))

    def load_frames(self, key: dict) -> dict:
        """Raw (artifact lines filled) and slow-registered functional stacks of a presentation, for plotting."""
        from djimaging.utils.motion import fill_artifact_lines
        npix = int((self & key).fetch1('npixartifact_qc'))
        data = load_presentation_channels(self.presentation_table, self.userinfo_table, self.raw_params_table, key,
                                          structural=False)
        raw = fill_artifact_lines(data['ch0'], npix)
        return dict(raw=raw, reg=register_stack(raw, self.get_motion(key).slow_frames), fs=data['fs'])

    def plot_frames(self, fig, key: dict, edges: np.ndarray, score: np.ndarray, frame_times=None,
                    n_spread: int = 3, n_worst: int = 4, gamma: float = 0.6) -> tuple:
        """Example frames (raw and registered means of the bins ``edges``) vs. the field reference into ``fig``, see
        ``plot_motion_frames``.

        Bins: those containing ``frame_times`` [s] if given, else ``n_spread`` evenly spaced bins plus the ``n_worst``
        bins after stimulus onset with the highest ``score``. Loads the recording. Returns (bin indices,
        picked-for-score flags, labels) for ``mark_frames``.
        """
        from djimaging.utils.motion.plotting import bins_at_times, pick_frame_bins, plot_motion_frames
        frames = self.load_frames(key)
        stim_onset = (self & key).fetch1('stim_onset')
        if frame_times is not None:
            idxs = np.unique(bins_at_times(frame_times, edges, frames['fs']))
            is_worst = np.zeros(len(idxs), bool)
        else:
            candidates = None if stim_onset is None else stim_bins(edges[:-1], frames['fs'], [stim_onset])[0]
            idxs, is_worst = pick_frame_bins(score, n_spread=n_spread, n_worst=n_worst, candidates=candidates)
        motion = self.get_motion(key)
        disp = np.array([motion.disp_frames[int(edges[b]):int(edges[b + 1])].mean(axis=0) for b in idxs])
        ref = self.motion_reference_table().get_reference(key)
        field_key = {k: key[k] for k in self.roi_mask_table.primary_key}
        roi_mask = load_array((self.roi_mask_table & field_key).fetch1('roi_mask'))
        labels = plot_motion_frames(fig, frames['raw'], frames['reg'], ref['ch0'], edges, idxs, frames['fs'],
                                    roi_mask=roi_mask, is_worst=is_worst, stim_onset=stim_onset, disp=disp,
                                    gamma=gamma)
        return idxs, is_worst, labels

    def plot1(self, key: dict | None = None, show_frames: bool = True, frame_times=None, n_spread: int = 3,
              n_worst: int = 4, gamma: float = 0.6) -> None:
        """Displacement (raw / slow), fast motion and z proxy of one presentation.

        With ``show_frames``, example frames (raw and registered means of the z-proxy bins, overlaid on the field
        reference in green / magenta, gamma-corrected) are shown below: the bins at ``frame_times`` [s], or
        ``n_spread`` evenly spaced bins plus the ``n_worst`` bins after stimulus onset with the most motion
        (|fast| + distance from the median displacement; red). This loads the recording.
        """
        from matplotlib import pyplot as plt
        from djimaging.utils.motion.plotting import mark_frames, mark_stimulus
        key = get_primary_key(self, key)
        motion = self.get_motion(key)
        row = (self & key).fetch1()
        fs = (self.presentation_table.ScanInfo & key).fetch1('scan_frequency')
        triggertimes = load_array((self.presentation_table & key).fetch1('triggertimes'))
        t = motion.bin_centers / fs
        edges = load_array(row['z_bin_edges'])
        tz = 0.5 * (edges[1:] + edges[:-1]) / fs

        fig = plt.figure(figsize=(16, 8 if show_frames else 3.5), layout='constrained')
        sfigs = fig.subfigures(2, 1, height_ratios=[1, 1.25]) if show_frames else [fig]
        axs = sfigs[0].subplots(1, 3)
        for ax in axs:
            for tt in triggertimes[:300]:
                ax.axvline(tt, c='0.85', lw=0.5, zorder=-5)
            mark_stimulus(ax, triggertimes, label=ax is axs[0])
            ax.set_xlabel('time [s]')
        for k, (c, lab) in enumerate((('C0', 'x'), ('C3', 'y'))):
            axs[0].plot(t, motion.disp_bins[:, k], c=c, lw=0.6, alpha=0.5)
            axs[0].plot(t, motion.slow_bins[:, k], c=c, lw=2, label=f'{lab} (slow, corrected)')
        axs[0].set(ylabel='displacement [px]', title='displacement vs. field reference')
        axs[0].legend(fontsize=7)
        axs[1].plot(t, np.linalg.norm(motion.fast_bins, axis=1), c='k', lw=0.7)
        axs[1].axhline(self.motion_reference_table.motion_params_table().get_params(key)['fast_thresh'], c='r', ls='--')
        axs[1].set(ylabel='|fast| [px]', title='uncorrected fast motion')
        for name, c in (('ch0', 'g'), ('ch1', 'm')):
            if row[f'z_self_{name}'] is None:
                continue
            axs[2].plot(tz, load_array(row[f'z_self_{name}']), c=c, label=f'{name} vs own mean')
            axs[2].plot(tz, load_array(row[f'z_ref_{name}']), c=c, ls=':', label=f'{name} vs field reference')
        axs[2].axhline(1, c='k', lw=0.5)
        axs[2].set(ylim=(0.4, 1.1), ylabel='similarity', title='z proxy (noise-corrected)')
        axs[2].legend(fontsize=7, ncol=2)
        if show_frames:
            sel = stim_bins(motion.bin_centers - (motion.bin_frames - 1) / 2., fs, triggertimes)[0]
            dev = np.linalg.norm(motion.disp_bins - np.median(motion.disp_bins[sel], axis=0), axis=1)
            amount = dev + np.linalg.norm(motion.fast_bins, axis=1)
            score = np.array([amount[(motion.bin_centers >= edges[b]) & (motion.bin_centers < edges[b + 1])].max(
                initial=0.) for b in range(len(edges) - 1)])
            idxs, is_worst, labels = self.plot_frames(sfigs[1], key, edges, score, frame_times=frame_times,
                                                      n_spread=n_spread, n_worst=n_worst, gamma=gamma)
            for ax in axs:
                mark_frames(ax, idxs, edges, fs, labels, is_worst)
        fig.suptitle(str(key), fontsize=8)
        plt.show()
