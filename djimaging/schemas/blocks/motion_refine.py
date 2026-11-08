from djimaging.schemas.blocks.motion import *


# Optional refinement of the field ROI mask on the registered mean of the reference condition (see docs/motion_qc.md:
# no measurable gain on the tested data, hence not part of the motion block). make_project.py inserts this table before
# MotionRoiMask and sets MotionRoiMask.refined_roi_mask_table = RefinedRoiMask, so that the presentation masks and
# MotionTrust start from the refined instead of the drawn field mask.
@schema
class RefinedRoiMask(motion_correction.RefinedRoiMaskTemplate):
    motion_detection_table = MotionDetection
    roi_table = Roi

    class Roi(motion_correction.RefinedRoiMaskTemplate.Roi):
        roi_table = Roi
