from djimaging.schemas.blocks.location import *
from djimaging.schemas.blocks.rf_sta import *
from djimaging.schemas.blocks.stimulator import *
from djimaging.tables import location, receptivefield


# Receptive field position relative to the ROI, using the stimulator geometry of the experiment
@schema
class RelativeRoiLocationWrtField(location.RelativeRoiLocationWrtFieldTemplate):
    roi_table = Roi
    roi_mask_table = RoiMask  # Field mask
    field_table = Field
    presentation_table = Presentation


@schema
class RfOffset(receptivefield.RfOffsetTemplate):
    stimulus_tab = Stimulus
    rf_split_tab = SplitRF
    rf_fit_tab = FitGauss2DRF


@schema
class RfRoiOffset(receptivefield.RfRoiOffsetTemplate):
    rf_offset_tab = RfOffset
    roi_pos_wrt_field_tab = RelativeRoiLocationWrtField
    userinfo_tab = UserInfo
    experiment_stimulator_tab = ExperimentStimulator
    stimulus_tab = Stimulus
    pres_tab = Presentation
    roimask_tab = RoiMask.RoiMaskPresentation
    rf_fit_tab = FitGauss2DRF
