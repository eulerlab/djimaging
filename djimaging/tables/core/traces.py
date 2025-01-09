import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.core.preprocesstraces import plot_left_right_clipping
from djimaging.utils.scanm.traces_and_triggers_utils import roi2trace_from_stack, check_valid_triggers_rel_to_tracetime
from djimaging.utils.scanm.read_h5_utils import load_roi2trace
from djimaging.utils import plot_utils, math_utils, trace_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_trace_and_trigger, prep_long_title


class TracesTemplate(dj.Computed):
    database = ""
    _include_motion_correction = False
    _mc_fupsample = 10
    _mc_f_cutoff = 3

    @property
    def definition(self):
        definition = """
        # Raw Traces for each roi under a specific presentation
        -> self.presentation_table
        -> self.roi_table
        -> self.raw_params_table
        """
        if self._include_motion_correction:
            definition += """
        -> self.motion_detection_table
        """

        definition += """
        ---
        trace          :longblob              # array of raw trace
        trace_t0       :float                 # numerical array of trace times
        trace_dt       :float                 # time between frames
        trace_valid    :tinyint unsigned      # Are values in trace correct (1) or not (0)?
        trigger_valid  :tinyint unsigned      # Are triggertimes inside trace_times (1) or not (0)?
        """
        return definition

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def roi_mask_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    def motion_detection_table(self):
        return None

    @property
    def key_source(self):
        try:
            key_source = self.presentation_table.proj() \
                         * self.raw_params_table.proj() \
                         * self.roi_mask_table.RoiMaskPresentation.proj()

            if self._include_motion_correction:
                key_source *= self.motion_detection_table.proj()

            return key_source
        except (AttributeError, TypeError):
            pass

    def make(self, key, verboselvl=0):
        include_artifacts, compute_from_stack, trace_precision, from_raw_data = (self.raw_params_table & key).fetch1(
            "include_artifacts", "compute_from_stack", "trace_precision", "from_raw_data")

        if self._include_motion_correction and not compute_from_stack:
            raise ValueError("Motion correction only supported for compute_from_stack")
        if from_raw_data and not compute_from_stack:
            raise ValueError("from_raw_data=True only supported for compute_from_stack=True")

        filepath = (self.presentation_table & key).fetch1("pres_data_file")
        triggertimes = (self.presentation_table & key).fetch1("triggertimes")
        roi_ids = (self.roi_table & key).fetch("roi_id")

        if compute_from_stack:
            roi2trace, frame_dt = self._compute_roi2trace_from_stack(
                key, filepath, roi_ids, trace_precision, from_raw_data, verboselvl=verboselvl)
        else:
            roi2trace, frame_dt = load_roi2trace(filepath, roi_ids)

        for roi_id, roi_data in roi2trace.items():
            if not include_artifacts and roi_data.get('incl_artifact', False):
                continue

            try:
                trace_key = key.copy()
                trace_key['roi_id'] = roi_id
                trace_key['trace'] = roi_data['trace']
                trace_key['trace_t0'] = roi_data['trace_times'][0]
                trace_key['trace_dt'] = frame_dt
                trace_key['trace_valid'] = roi_data['trace_valid']
                trace_key['trigger_valid'] = check_valid_triggers_rel_to_tracetime(
                    trace_valid=roi_data['trace_valid'], trace_times=roi_data['trace_times'], triggertimes=triggertimes)

                self.insert1(trace_key)
            except Exception as e:
                if roi_data['trace_valid']:
                    raise e
                else:
                    warnings.warn(f"Skipping invalid trace for {key} and roi_id={roi_id}")

    def _compute_roi2trace_from_stack(self, key, filepath, roi_ids, trace_precision, from_raw_data, verboselvl=0):
        data_stack_name = (self.userinfo_table & key).fetch1("data_stack_name")
        roi_mask = (self.roi_mask_table.RoiMaskPresentation & key).fetch1("roi_mask")
        n_artifact = (self.presentation_table & key).fetch1("npixartifact")

        if (self.roi_mask_table.RoiMaskPresentation & key).fetch1('as_field_mask') == 'different':
            raise ValueError(f'Tried to populate traces with inconsistent roi mask for key=\n{key}\n' +
                             'Compare ROI mask of Field and Presentation.')

        if self._include_motion_correction and (self.motion_detection_table is not None):
            shifts_x, shifts_y = (self.motion_detection_table & key).fetch1('shifts_x', 'shifts_y')
            fs = (self.presentation_table.ScanInfo & key).fetch1('scan_frequency')
            roi2trace, frame_dt = roi2trace_from_stack(
                filepath=filepath, roi_ids=roi_ids, roi_mask=roi_mask,
                data_stack_name=data_stack_name, precision=trace_precision, from_raw_data=from_raw_data,
                shifts_x=shifts_x, shifts_y=shifts_y,
                shift_kws=dict(fs=fs, fupsample=self._mc_fupsample, f_cutoff=self._mc_f_cutoff)
            )
        else:
            roi2trace, frame_dt = roi2trace_from_stack(
                filepath=filepath, roi_ids=roi_ids, roi_mask=roi_mask,
                data_stack_name=data_stack_name, precision=trace_precision, from_raw_data=from_raw_data)

        for roi_id, roi_data in roi2trace.items():
            if np.any(-roi_mask[:n_artifact, :] == roi_id):
                if verboselvl > 0:
                    print('Found light artifact in :', key, roi_id)
            roi2trace[roi_id]['incl_artifact'] = np.any(-roi_mask[:n_artifact, :] == roi_id)

        return roi2trace, frame_dt

    def gui_clip_trace(self, key):
        """GUI to clip traces. Note that this can not be easily undone for now."""
        trace, trace_t0, trace_dt, valid_trace, valid_trigger = (self & key).fetch1(
            'trace', 'trace_t0', 'trace_dt', 'trace_valid', 'trigger_valid')
        trace_t = np.arange(trace.size) * trace_dt + trace_t0

        import ipywidgets as widgets

        w_left = widgets.IntSlider(0, min=0, max=trace.size - 1, step=1,
                                   layout=widgets.Layout(width='800px'))
        w_right = widgets.IntSlider(trace.size - 1, min=0, max=trace.size - 1, step=1,
                                    layout=widgets.Layout(width='800px'))
        w_save = widgets.Checkbox(False)

        title = 'Not saved\n' + prep_long_title(key)

        @widgets.interact(left=w_left, right=w_right, save=w_save)
        def plot_fit(left=0, right=trace.size - 1, save=False):
            nonlocal title, key

            plot_left_right_clipping(trace, trace_t, left, right, title)

            if save:
                i0, i1 = (right, left + 1) if right < left else (left, right + 1)
                self.add_or_update(
                    key, trace=trace[i0:i1], trace_t0=trace_t0 + left * trace_dt, trace_dt=trace_dt,
                    valid_trace=valid_trace, valid_trigger=valid_trigger)
                title = f'SAVED: left={left}, right={right}\n{prep_long_title(key)}'
                w_save.value = False

    def add_or_update(self, key, trace, trace_t0, trace_dt, valid_trace, valid_trigger):
        entry = dict(**key, trace=trace, trace_t0=trace_t0, trace_dt=trace_dt,
                     trace_valid=valid_trace, trigger_valid=valid_trigger)
        self.update1(entry)

    def plot1(self, key=None, xlim=None, ylim=None):
        key = get_primary_key(table=self, key=key)
        trace_t0, trace_dt, trace = (self & key).fetch1("trace_t0", "trace_dt", "trace")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")
        trace_times = np.arange(len(trace)) * trace_dt + trace_t0

        ax = plot_trace_and_trigger(
            time=trace_times, trace=trace, triggertimes=triggertimes, title=str(key))
        ax.set(xlim=xlim, ylim=ylim)

    def plot(self, restriction=None, sort=True):
        if restriction is None:
            restriction = dict()

        traces = (self & restriction).fetch("trace")

        traces = math_utils.padded_vstack(traces, cval=np.nan)
        n = traces.shape[0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 1 + np.minimum(n * 0.1, 10)))
        if len(restriction) > 0:
            plot_utils.set_long_title(fig=fig, title=restriction)

        sort_idxs = trace_utils.argsort_traces(traces, ignore_nan=True) if sort else np.arange(n)

        ax.set_title('traces')
        plot_utils.plot_signals_heatmap(ax=ax, signals=traces[sort_idxs, :], symmetric=False)
        plt.show()
