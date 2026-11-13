"""
Geometry of mouse cam (MC, natural movie) presentations.

Maps frames of the QDSpy export of the movie (x = column, y = row from the top, as shown with imshow) to the field
frame of the recording (rows along +XCoord_um, columns along +YCoord_um), using only the stimulator of the experiment.
The version (v1, v2, v2new, v3) is read from the movie names in the QDSpy log and checked against the setup.

Example usage:

@schema
class MouseCamGeometry(stimulator.MouseCamGeometryTemplate):
    presentation_table = Presentation
    stimulus_table = Stimulus
    experiment_stimulator_table = ExperimentStimulator
    expinfo_table = Experiment.ExpInfo
    presentation_log_table = PresentationLog  # optional
    stim_log_table = QdsPyLog.StimLog  # optional

movie_field = MouseCamGeometry().to_field(key, movie)  # movie: (time, rows, cols[, colors]) as exported by QDSpy
"""
from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.tables.stimulator.mb_direction import fetch_stim_log
from djimaging.utils.stimulator import (
    QDSPY_EXPORT_ARRAY_TO_SCREEN, array_to_field, array_to_field_matrix, combine_confidence, join_reasons,
    mc_confidence, mc_version_from_params)


class MouseCamGeometryTemplate(dj.Computed):
    database = ""
    _stim_family = 'mouse_cam'

    presentation_log_table = None
    stim_log_table = None

    @property
    def definition(self):
        definition = """
        # Orientation of mouse cam movies in the recording field
        -> self.presentation_table
        -> self.experiment_stimulator_table
        ---
        mc_version = null      : varchar(16)    # v1, v2, v2new, v3 (NULL: unknown)
        export_to_field        : <blob>         # 2x2 map from QDSpy export frame (x=col, y=row) to (XCoord, YCoord)
        confidence             : enum('high', 'medium', 'low')
        reason                 : varchar(1023)
        """
        return definition

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def experiment_stimulator_table(self):
        pass

    @property
    @abstractmethod
    def expinfo_table(self):
        pass

    @property
    def key_source(self):
        try:
            return ((self.presentation_table & (self.stimulus_table & dict(stim_family=self._stim_family))).proj()
                    * self.experiment_stimulator_table.proj())
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        geometry = self.experiment_stimulator_table().get_geometry(key)
        setupid = (self.expinfo_table & key).fetch1('setupid')
        stim_log, log_reason = fetch_stim_log(self.presentation_log_table, self.stim_log_table, key)
        params = stim_log.get('params') if stim_log else None
        version = mc_version_from_params(params)
        version_conf, version_reason = mc_confidence(version, setupid)

        self.insert1(dict(
            **key, mc_version=version,
            export_to_field=array_to_field_matrix(
                QDSPY_EXPORT_ARRAY_TO_SCREEN, geometry['screen_x_axis'], geometry['screen_y_axis']),
            confidence=combine_confidence(geometry['confidence'], version_conf),
            reason=join_reasons(geometry['reason'], log_reason if stim_log is None else '', version_reason)[:1023]))

    def to_field(self, key: dict, movie: np.ndarray) -> np.ndarray:
        """Reorient a QDSpy export movie (time, rows, cols) or (time, rows, cols, colors) to the field frame."""
        geometry = self.experiment_stimulator_table().get_geometry(key)
        movie = np.asarray(movie)
        has_colors = movie.ndim == 4
        if has_colors:
            movie = np.moveaxis(movie, -1, 0)
        out = array_to_field(movie, QDSPY_EXPORT_ARRAY_TO_SCREEN, geometry['screen_x_axis'], geometry['screen_y_axis'])
        return np.moveaxis(out, 0, -1) if has_colors else out
