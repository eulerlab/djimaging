"""Motion estimation, motion QC (z proxy, per-ROI trust) and between-presentation ROI shifts.

Used by the tables in ``djimaging.tables.motion_correction``; method and validation in ``docs/motion_qc.md``.
"""
from djimaging.utils.motion.channel_cleaning import clean_channels, fill_artifact_lines
from djimaging.utils.motion.image_utils import register_stack
from djimaging.utils.motion.registration import MotionEstimate, estimate_motion, make_reference, xcorr_disp
from djimaging.utils.motion.roi_shifts import (
    estimate_roi_shifts, extract_traces_weighted, pool_roi_shifts, presentation_roi_weights, roi_ids_from_mask,
    smooth_shift_field)
from djimaging.utils.motion.trust import compute_trust, trust_per_repeat
from djimaging.utils.motion.zproxy import compute_zproxy
