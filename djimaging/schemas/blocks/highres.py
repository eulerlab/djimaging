from djimaging.schemas.blocks.core import *
from djimaging.tables import misc


# HighRes scan aligned to the field ROI mask (HighRes itself is part of the core block)
@schema
class HighResAlignment(misc.HighResAlignmentTemplate):
    highres_table = HighRes
    roi_mask_table = RoiMask
    presentation_table = Presentation
    userinfo_table = UserInfo
