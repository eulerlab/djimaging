from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.scanm_utils import roi2trace_from_h5_file, roi2trace_from_stack, \
    check_valid_triggers_rel_to_tracetime
from djimaging.utils import plot_utils, math_utils, trace_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_trace_and_trigger


class TracesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Raw Traces for each roi under a specific presentation
        -> self.presentation_table
        -> self.roi_table
        -> self.raw_params_table
        ---
        trace          :longblob              # array of raw trace
        trace_times    :longblob              # numerical array of trace times
        trace_flag     :tinyint unsigned      # Are values in trace correct (1) or not (0)?
        trigger_flag   :tinyint unsigned      # Are triggertimes inside trace_times (1) or not (0)?
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
    def key_source(self):
        try:
            return self.presentation_table.proj() \
                * self.raw_params_table.proj() \
                * self.roi_mask_table.RoiMaskPresentation.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        include_artifacts, compute_from_stack, trace_precision, from_raw_data = (self.raw_params_table & key).fetch1(
            "include_artifacts", "compute_from_stack", "trace_precision", "from_raw_data")

        filepath = (self.presentation_table & key).fetch1("pres_data_file")
        triggertimes = (self.presentation_table & key).fetch1("triggertimes")
        roi_ids = (self.roi_table & key).fetch("roi_id")
        n_artifact = (self.presentation_table & key).fetch1("npixartifact")

        if compute_from_stack:
            data_stack_name = (self.userinfo_table & key).fetch1("data_stack_name")
            roi_mask = (self.roi_mask_table.RoiMaskPresentation & key).fetch1("roi_mask")

            if (self.roi_mask_table.RoiMaskPresentation & key).fetch1('as_field_mask') == 'different':
                raise ValueError(f'Tried to populate traces with inconsistent roi mask for key=\n{key}\n' +
                                 'Compare ROI mask of Field and Presentation.')

            roi2trace = roi2trace_from_stack(
                filepath=filepath, roi_ids=roi_ids, roi_mask=roi_mask,
                data_stack_name=data_stack_name, precision=trace_precision, from_raw_data=from_raw_data)

            for roi_id, roi_data in roi2trace.items():
                if np.any(-roi_mask[:n_artifact, :] == roi_id):
                    print(key, roi_id)
                roi2trace[roi_id]['incl_artifact'] = np.any(-roi_mask[:n_artifact, :] == roi_id)

        else:
            assert not from_raw_data, "from_raw_data=True only supported for compute_from_stack=True"
            roi2trace = roi2trace_from_h5_file(filepath, roi_ids)

        for roi_id, roi_data in roi2trace.items():
            if not include_artifacts and roi_data.get('incl_artifact', False):
                continue

            trace_key = key.copy()
            trace_key['roi_id'] = roi_id
            trace_key['trace'] = roi_data['trace']
            trace_key['trace_times'] = roi_data['trace_times']
            trace_key['trace_flag'] = roi_data['valid_flag']
            trace_key['trigger_flag'] = check_valid_triggers_rel_to_tracetime(
                trace_flag=trace_key['trace_flag'], trace_times=trace_key['trace_times'], triggertimes=triggertimes)

            self.insert1(trace_key)

    def plot1(self, key=None, xlim=None, ylim=None):
        key = get_primary_key(table=self, key=key)
        trace_times, trace = (self & key).fetch1("trace_times", "trace")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

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
