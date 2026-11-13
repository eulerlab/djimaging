"""Setup-dependent parameters of recordings (stimulator delay, scan width at zoom 1, light artifact rows).

Tables that load ScanM recordings (Field, Presentation, HighRes, OutlineAbs) inherit `RecordingSetupMixin`. If the
schema binds `experiment_stimulator_table` (ExperimentStimulator) and `experiment_imaging_setup_table`
(ExperimentImagingSetup), the values are taken from these tables (dated, per experiment). If not (older schemas), the
setupid constants of `djimaging.utils.scanm.setup_utils` are used, as before.
"""


class RecordingSetupMixin:
    experiment_stimulator_table = None  # Optional: ExperimentStimulator (stimulator delay)
    experiment_imaging_setup_table = None  # Optional: ExperimentImagingSetup (scan width, light artifact rows)

    def recording_setup_kwargs(self, key: dict) -> dict:
        """Keyword arguments for `ScanMRecording` (setup_id, date and, if bound, the values of the setup tables)."""
        setupid = (self.experiment_table().ExpInfo & key).fetch1('setupid')
        kwargs = dict(setup_id=setupid, date=key['date'])

        if self.experiment_stimulator_table is not None:
            kwargs['stimulator_delay'] = self.experiment_stimulator_table().get_stimulator_delay_s(key)

        if self.experiment_imaging_setup_table is not None:
            kwargs.update(self.experiment_imaging_setup_table().get_recording_params(key))

        return kwargs
