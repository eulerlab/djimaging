"""
Presentation ROI masks of the motion mode: the field ROI mask with a per-ROI between-presentation shift and a mild
border reshape, as subpixel weights in the frame of the (slowly) registered stack. In motion mode this table replaces
RoiMask.RoiMaskPresentation (RoiMask then only stores the field mask, see RoiMaskTemplate._presentation_masks).

Example usage:

@schema
class MotionRoiMask(motion_correction.MotionRoiMaskTemplate):
    motion_detection_table = MotionDetection
    roi_table = Roi

    class Roi(motion_correction.MotionRoiMaskTemplate.Roi):
        roi_table = Roi
"""
from __future__ import annotations

from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.dj_storage import load_array
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.mask_utils import shift_array
from djimaging.utils.motion import fill_artifact_lines, pool_roi_shifts, presentation_roi_weights, roi_ids_from_mask
from djimaging.utils.motion.image_utils import register_stack
from djimaging.utils.motion.roi_shifts import crop_weights, roi_centroids, uncrop_weights, weights_to_label
from djimaging.tables.motion_correction.motion_detection import load_presentation_channels, reference_pres_key


class MotionRoiMaskTemplate(dj.Computed):
    """Presentation ROI weights: per-ROI shift (pooled per condition or per presentation) + mild reshape."""
    database = ""

    @property
    def definition(self) -> str:
        return """
        # ROI mask of a presentation in motion mode (replaces RoiMask.RoiMaskPresentation)
        -> self.motion_detection_table
        ---
        roi_mask : <blob>  # label mask (Igor format), raw presentation coordinates: weights >= 0.5 + global shift
        shift_dx : int32  # rounded global shift, as in RoiMaskPresentation: shift_array(roi_mask, (dx, dy)) ~ field mask
        shift_dy : int32
        is_reference : bool  # presentation of the field ROI mask (no shifts)
        pooling : varchar(16)  # 'cond1' or 'presentation'
        n_pooled : int32  # number of presentations the per-ROI shifts were pooled over
        lam : float32  # shrinkage of per-ROI deviations from the smooth displacement field (0: field only, 1: per ROI)
        """

    class Roi(dj.Part):
        @property
        def definition(self) -> str:
            return """
            # Weights of one ROI
            -> master
            -> self.roi_table
            ---
            shift_x : float32  # applied between-presentation shift [px] (frame of the registered stack)
            shift_y : float32
            weights : <blob>  # weights cropped to their bounding box (frame of the registered stack)
            x0 : int32  # position of the bounding box
            y0 : int32
            n_px_ref : int32  # pixels of the ROI in the field mask
            n_px : int32  # pixels with weight >= 0.5 after reshape
            low_contrast : bool  # ROI lost more than half of its contrast (not reshaped)
            at_limit : bool  # shift at the search limit (likely a mismatched cell)
            inconsistent : bool  # shift differs between the pooled presentations (> 0.5 px)
            low_quality : bool  # poor patch match, field value used
            """

        @property
        @abstractmethod
        def roi_table(self):
            pass

    @property
    @abstractmethod
    def motion_detection_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.motion_detection_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    def _reference_table(self):
        return self.motion_detection_table.motion_reference_table

    def _pooling_group(self, key: dict, params: dict) -> list:
        """Presentation keys whose per-ROI shifts are pooled with ``key`` (without the reference presentation)."""
        det = self.motion_detection_table
        pres_tab = det.presentation_table
        field_key = {k: key[k] for k in self.motion_detection_table.roi_mask_table.primary_key}
        ref_key = reference_pres_key(det.roi_mask_table, pres_tab, field_key)
        if params['pooling'] == 'presentation':
            group = [{k: key[k] for k in pres_tab.primary_key}]
        else:
            restr = dict(field_key)
            if 'cond1' in pres_tab.primary_key:
                restr['cond1'] = key['cond1']
            group = (pres_tab & restr).proj().to_dicts()
        group = [g for g in group if not all(g[k] == ref_key[k] for k in pres_tab.primary_key)]
        missing = [g for g in group if len(det & g & dict(motion_params_id=key['motion_params_id'])) == 0]
        if missing:
            raise ValueError(f'Populate MotionDetection for all presentations of the pooling group first; '
                             f'missing:\n{missing}')
        return group

    def make(self, key: dict) -> None:
        det = self.motion_detection_table
        ref_tab = self._reference_table
        params = ref_tab.motion_params_table().get_params(key)
        field_key = {k: key[k] for k in det.roi_mask_table.primary_key}
        field_mask = load_array((det.roi_mask_table & field_key).fetch1('roi_mask'))
        roi_ids = roi_ids_from_mask(field_mask)
        n = len(roi_ids)
        ref_key = reference_pres_key(det.roi_mask_table, det.presentation_table, field_key)
        is_reference = all(key[k] == ref_key[k] for k in det.presentation_table.primary_key)

        flags = dict(at_limit=np.zeros(n, bool), inconsistent=np.zeros(n, bool), low_quality=np.zeros(n, bool))
        if is_reference:
            shifts, lam, n_pooled = np.zeros((n, 2)), np.nan, 0
        else:
            group = self._pooling_group(key, params)
            rows = [(det & g & dict(motion_params_id=key['motion_params_id'])).fetch1(
                'roi_shifts_meas', 'roi_shift_quality') for g in group]
            shifts, info = pool_roi_shifts(
                roi_centroids(field_mask, roi_ids), [load_array(r[0]) for r in rows], [load_array(r[1]) for r in rows],
                sigma=params['local_sigma'], lam_single=params['lam_single'], max_shift=params['roi_max_shift'])
            lam, n_pooled = info['lam'], info['n_presentations']
            flags = {k: info[k] for k in flags}

        ref = ref_tab().get_reference(key)
        npix = int((det & key).fetch1('npixartifact_qc'))
        data = load_presentation_channels(det.presentation_table, det.userinfo_table, det.raw_params_table, key,
                                          structural=False)
        motion = det().get_motion(key)
        reg0 = register_stack(fill_artifact_lines(data['ch0'], npix), motion.slow_frames)
        weights, winfo = presentation_roi_weights(
            np.nanmean(reg0, axis=2), np.mean(np.isfinite(reg0), axis=2), ref['ch0'], field_mask, shifts, npix,
            rel_thresh=params['rel_thresh'], rel_drop=params['rel_drop'], min_valid_frac=params['min_valid_frac'],
            min_contrast_frac=params['min_contrast_frac'])

        glob = np.round(np.median(motion.slow_bins, axis=0)).astype(int)
        label_raw = shift_array(weights_to_label(weights, roi_ids), shift=(int(glob[0]), int(glob[1])), cval=1)

        self.insert1(dict(key, roi_mask=label_raw, shift_dx=int(-glob[0]), shift_dy=int(-glob[1]),
                          is_reference=is_reference, pooling=params['pooling'], n_pooled=int(n_pooled),
                          lam=float(lam) if np.isfinite(lam) else np.nan))
        parts = []
        for i, rid in enumerate(roi_ids):
            cropped, x0, y0 = crop_weights(weights[i])
            parts.append(dict(key, roi_id=int(abs(rid)), shift_x=float(shifts[i, 0]), shift_y=float(shifts[i, 1]),
                              weights=cropped, x0=x0, y0=y0, n_px_ref=winfo[int(rid)]['n_px_ref'],
                              n_px=winfo[int(rid)]['n_px'], low_contrast=winfo[int(rid)]['low_contrast'],
                              at_limit=bool(flags['at_limit'][i]), inconsistent=bool(flags['inconsistent'][i]),
                              low_quality=bool(flags['low_quality'][i])))
        self.Roi().insert(parts)

    def get_weights(self, key: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ROI ids, weights (n_rois, x, y) and shifts (n_rois, 2) in the frame of the registered stack."""
        shape = load_array((self & key).fetch1('roi_mask')).shape
        rows = (self.Roi & key).to_dicts(order_by='roi_id')
        roi_ids = np.array([r['roi_id'] for r in rows], dtype=int)
        weights = np.stack([uncrop_weights(load_array(r['weights']), r['x0'], r['y0'], shape) for r in rows]) \
            if rows else np.zeros((0,) + shape, np.float32)
        shifts = np.array([[r['shift_x'], r['shift_y']] for r in rows]).reshape(-1, 2)
        return roi_ids, weights, shifts

    def plot1(self, key: dict | None = None, gamma: float = 0.7) -> None:
        """Registered mean image with field mask (yellow), presentation mask (green) and per-ROI shifts (x3)."""
        from matplotlib import pyplot as plt
        key = get_primary_key(self, key)
        det = self.motion_detection_table
        roi_ids, weights, shifts = self.get_weights(key)
        field_key = {k: key[k] for k in det.roi_mask_table.primary_key}
        field_mask = load_array((det.roi_mask_table & field_key).fetch1('roi_mask'))
        data = load_presentation_channels(det.presentation_table, det.userinfo_table, det.raw_params_table, key,
                                          structural=False)
        img = np.nanmean(register_stack(data['ch0'], det().get_motion(key).slow_frames), axis=2)
        img = (img - np.nanmin(img)) / max(np.nanmax(img) - np.nanmin(img), 1e-12)
        label = weights_to_label(weights, -roi_ids)
        flagged = (self.Roi & key & 'low_contrast OR at_limit OR inconsistent').to_arrays('roi_id')

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(np.nan_to_num(img).T ** gamma, cmap='gray', origin='lower')
        for mask, color, lw in ((field_mask, 'y', 0.4), (label, 'lime', 0.8)):
            for rid in roi_ids_from_mask(mask):
                ax.contour((mask == rid).T, levels=[0.5], colors=[color], linewidths=lw, origin='lower')
        for rid in flagged:
            ax.contour((label == -rid).T, levels=[0.5], colors=['c'], linewidths=1.5, origin='lower')
        cent = roi_centroids(field_mask, -roi_ids)
        ax.quiver(cent[:, 0], cent[:, 1], shifts[:, 0], shifts[:, 1], color='orange', angles='xy', scale_units='xy',
                  scale=1 / 3, width=0.004)
        ax.set_title(f'{key}\nyellow: field mask, green: presentation mask, cyan: flagged, arrows: shift x3',
                     fontsize=8)
        plt.tight_layout()
        plt.show()
