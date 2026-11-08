"""
Refined field ROI mask of the motion mode: the field ROI mask (RoiMask, drawn on one unregistered presentation) with
its ROI borders adjusted (within +-1 px) on the registered mean of all presentations of the reference condition.
ROI ids are kept; ROIs are never added, removed or merged, so Roi and its downstream tables stay valid.
MotionRoiMask and MotionTrust start from this mask. See ``djimaging.utils.motion.refine`` and docs/motion_qc.md.

Example usage:

@schema
class RefinedRoiMask(motion_correction.RefinedRoiMaskTemplate):
    motion_detection_table = MotionDetection
    roi_table = Roi

    class Roi(motion_correction.RefinedRoiMaskTemplate.Roi):
        roi_table = Roi
"""
from __future__ import annotations

from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.dj_storage import load_array
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.motion import fill_artifact_lines, roi_ids_from_mask
from djimaging.utils.motion.image_utils import register_stack
from djimaging.utils.motion.refine import REFINE_STATUS, pooled_mean_image, refine_roi_mask
from djimaging.tables.motion_correction.motion_detection import load_presentation_channels, reference_pres_key


class RefinedRoiMaskTemplate(dj.Computed):
    """Field ROI mask with borders refined on the registered mean of the reference condition."""
    database = ""

    @property
    def definition(self) -> str:
        return """
        # Field ROI mask refined on the registered mean of the reference condition (motion mode)
        -> self.motion_detection_table.motion_reference_table
        ---
        roi_mask : <blob>  # refined field mask (Igor format), same frame and ROI ids as RoiMask.roi_mask
        ref_img : <blob>  # registered mean of the reference condition (functional channel), refined on
        n_presentations : int32  # presentations pooled into ref_img
        n_rois_refined : int32  # ROIs whose border changed
        """

    class Roi(dj.Part):
        @property
        def definition(self) -> str:
            return f"""
            # Refinement of one ROI
            -> master
            -> self.roi_table
            ---
            n_px_orig : int32  # pixels in the drawn ROI
            n_px : int32  # pixels in the refined ROI
            n_added : int32  # pixels added (within refine_max_px outside the drawn border)
            n_removed : int32  # pixels removed (within refine_max_px inside the drawn border)
            contrast : float32  # (median inside - local background) / robust SD of ref_img
            status : enum({', '.join(repr(s) for s in REFINE_STATUS)})  # 'low_contrast'/'too_small': kept as drawn
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
            return self.motion_detection_table.motion_reference_table.proj()
        except (AttributeError, TypeError):
            pass

    def reference_condition_keys(self, key: dict) -> list:
        """Presentation keys of the reference condition (same ``cond1`` as the presentation of the field mask)."""
        det = self.motion_detection_table
        pres_tab = det.presentation_table
        field_key = {k: key[k] for k in det.roi_mask_table.primary_key}
        ref_key = reference_pres_key(det.roi_mask_table, pres_tab, field_key)
        restr = dict(field_key)
        if 'cond1' in pres_tab.primary_key:
            restr['cond1'] = ref_key['cond1']
        group = (pres_tab & restr).proj().to_dicts()
        missing = [g for g in group if len(det & g & dict(motion_params_id=key['motion_params_id'])) == 0]
        if missing:
            raise ValueError(f'Populate MotionDetection for all presentations of the reference condition first; '
                             f'missing:\n{missing}')
        return group

    def make(self, key: dict) -> None:
        det = self.motion_detection_table
        params = det.motion_reference_table.motion_params_table().get_params(key)
        field_key = {k: key[k] for k in det.roi_mask_table.primary_key}
        field_mask = load_array((det.roi_mask_table & field_key).fetch1('roi_mask'))
        npix_field = int((det.motion_reference_table & key).fetch1('npixartifact_qc'))

        group = self.reference_condition_keys(key)
        means, n_frames = [], []
        for g in group:
            g = {**g, 'motion_params_id': key['motion_params_id']}
            npix = int((det & g).fetch1('npixartifact_qc'))
            data = load_presentation_channels(det.presentation_table, det.userinfo_table, det.raw_params_table, g,
                                              structural=False)
            reg0 = register_stack(fill_artifact_lines(data['ch0'], npix), det().get_motion(g).slow_frames)
            means.append(np.nanmean(reg0, axis=2))
            n_frames.append(reg0.shape[2])
            npix_field = max(npix_field, npix)
        ref_img = pooled_mean_image(means, n_frames).astype(np.float32)

        if params['refine_mask']:
            refined, info = refine_roi_mask(
                ref_img, field_mask, npix_field, max_px=params['refine_max_px'],
                rel_thresh=params['refine_rel_thresh'], min_contrast=params['refine_min_contrast'],
                min_keep_frac=params['refine_min_keep_frac'], grow=params['refine_grow'])
        else:
            refined = field_mask.copy()
            info = {int(rid): dict(n_px_orig=int((field_mask == rid).sum()), n_px=int((field_mask == rid).sum()),
                                   n_added=0, n_removed=0, contrast=np.nan, status='unchanged')
                    for rid in roi_ids_from_mask(field_mask)}

        self.insert1(dict(key, roi_mask=refined, ref_img=ref_img, n_presentations=len(group),
                          n_rois_refined=int(sum(d['status'] == 'refined' for d in info.values()))))
        self.Roi().insert([dict(key, roi_id=int(abs(rid)), n_px_orig=d['n_px_orig'], n_px=d['n_px'],
                                n_added=d['n_added'], n_removed=d['n_removed'],
                                contrast=float(d['contrast']) if np.isfinite(d['contrast']) else np.nan,
                                status=d['status']) for rid, d in info.items()])

    def get_mask(self, key: dict) -> np.ndarray:
        return load_array((self & key).fetch1('roi_mask'))

    def plot1(self, key: dict | None = None, gamma: float = 0.7) -> None:
        """Registered mean of the reference condition with the drawn (yellow) and the refined (green) mask."""
        from matplotlib import pyplot as plt
        key = get_primary_key(self, key)
        det = self.motion_detection_table
        field_key = {k: key[k] for k in det.roi_mask_table.primary_key}
        field_mask = load_array((det.roi_mask_table & field_key).fetch1('roi_mask'))
        refined = self.get_mask(key)
        img = load_array((self & key).fetch1('ref_img'))
        img = (img - np.nanmin(img)) / max(np.nanmax(img) - np.nanmin(img), 1e-12)
        kept = (self.Roi & key & "status in ('low_contrast', 'too_small')").to_arrays('roi_id')

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(np.nan_to_num(img).T ** gamma, cmap='gray', origin='lower')
        for mask, color, lw in ((field_mask, 'y', 0.5), (refined, 'lime', 0.9)):
            for rid in roi_ids_from_mask(mask):
                ax.contour((mask == rid).T, levels=[0.5], colors=[color], linewidths=lw, origin='lower')
        for rid in kept:
            ax.contour((refined == -rid).T, levels=[0.5], colors=['c'], linewidths=1.5, origin='lower')
        ax.set_title(f'{key}\nyellow: drawn, green: refined, cyan: kept as drawn (low contrast / too small)',
                     fontsize=8)
        plt.tight_layout()
        plt.show()
