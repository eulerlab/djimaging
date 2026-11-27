from djimaging.schemas.blocks.core import *
from djimaging.tables import motion_correction


# Motion correction and motion QC ("motion mode", see docs/motion_qc.md).
# make_project.py inserts this block before Traces and switches the core tables into motion mode: RoiMask only stores
# the field mask (no RoiMaskPresentation), MotionRoiMask provides the presentation ROI masks, and Traces are extracted
# from the registered stack with these masks.
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


@schema
class MotionRoiMask(motion_correction.MotionRoiMaskTemplate):
    motion_detection_table = MotionDetection
    roi_table = Roi

    class Roi(motion_correction.MotionRoiMaskTemplate.Roi):
        roi_table = Roi


@schema
class MotionTrust(motion_correction.MotionTrustTemplate):
    motion_roi_mask_table = MotionRoiMask
    stimulus_table = Stimulus

    class Roi(motion_correction.MotionTrustTemplate.Roi):
        motion_roi_mask_table = MotionRoiMask
