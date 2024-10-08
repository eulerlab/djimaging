from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils import plot_utils, math_utils, trace_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_trace_and_trigger
from djimaging.utils.scanm_utils import roi2trace_from_stack, roi2trace_from_h5_file, \
    check_valid_triggers_rel_to_tracetime


class TracesTemplate(dj.Computed):
    database = ""
    _ignore_incompatible_roi_masks = False

    @property
    def definition(self):
        definition = """
        # Raw Traces for each roi under a specific presentation
        -> self.presentation_table
        -> self.roi_table
        -> self.params_table
        ---
        trace          :longblob              # array of raw trace
        trace_times    :longblob              # numerical array of trace times
        trace_flag     :tinyint unsigned      # Are values in trace correct (1) or not (0)?
        trigger_flag   :tinyint unsigned      # Are triggertimes inside trace_times (1) or not (0)?
        """
        return definition

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
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
            return self.presentation_table().proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        include_artifacts, compute_from_stack, trace_precision = (self.params_table() & key).fetch1(
            "include_artifacts", "compute_from_stack", "trace_precision")
        filepath = (self.presentation_table() & key).fetch1("h5_header")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")
        roi_ids = (self.roi_table() & key).fetch("roi_id")

        if not self._ignore_incompatible_roi_masks:
            if (self.presentation_table.RoiMask & key).fetch1('pres_and_field_mask') == 'different':
                raise ValueError(f'Tried to populate traces with inconsistent roi mask for key=\n{key}\n' +
                                 'Compare ROI mask of Field and Presentation.' +
                                 'To ignore this error set self._ignore_incompatible_roi_masks=True')

        if compute_from_stack:
            data_stack_name = (self.userinfo_table() & key).fetch1("data_stack_name")
            roi_mask = (self.presentation_table.RoiMask() & key).fetch1("roi_mask")

            roi2trace = roi2trace_from_stack(
                filepath=filepath, roi_ids=roi_ids, roi_mask=roi_mask,
                data_stack_name=data_stack_name, precision=trace_precision)
        else:
            roi2trace = roi2trace_from_h5_file(filepath, roi_ids)

        for roi_id in roi_ids:
            if not include_artifacts:
                if (self.roi_table() & key & dict(roi_id=roi_id)).fetch1("artifact_flag") == 1:
                    continue

            roi_data = roi2trace[roi_id]

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
