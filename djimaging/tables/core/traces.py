from abc import abstractmethod

import datajoint as dj

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_trace_and_trigger
from djimaging.utils.scanm_utils import load_traces_from_h5_file


class TracesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Raw Traces for each roi under a specific presentation
    
        -> self.presentation_table
        -> self.roi_table
    
        ---
        trace          :longblob              # array of raw trace
        trace_times    :longblob              # numerical array of trace times
        trace_flag     :tinyint unsigned      # Are values in trace correct (1) or not (0)?
        trigger_flag   :tinyint unsigned      # Are triggertimes inside trace_times (1) or not (0)?
        """
        return definition

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    _include_artifacts = False

    @property
    def key_source(self):
        roi_restriction = dict() if self._include_artifacts else 'artifact_flag=0'

        try:
            return ((self.roi_table() & roi_restriction) * self.presentation_table()).proj()
        except TypeError:
            pass

    def make(self, key):

        # get all params we need for creating trace
        filepath = (self.presentation_table() & key).fetch1("h5_header")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")
        roi_ids = (self.roi_table() & key).fetch("roi_id")

        roi2trace = load_traces_from_h5_file(filepath, roi_ids)

        for roi_id, roi_data in roi2trace.items():
            trace_key = key.copy()
            trace_key['roi_id'] = roi_id
            trace_key['trace'] = roi_data['trace']
            trace_key['trace_times'] = roi_data['trace_times']
            trace_key['trace_flag'] = roi_data['valid_flag']

            if trace_key['trace_flag']:
                if triggertimes[0] < trace_key['trace_times'][0]:
                    trace_key["trigger_flag"] = 0
                elif trace_key['trace_flag'] and triggertimes[-1] > trace_key['trace_times'][-1]:
                    trace_key["trigger_flag"] = 0
                else:
                    trace_key["trigger_flag"] = 1
            else:
                trace_key["trigger_flag"] = 0

            self.insert1(trace_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        trace_times, trace = (self & key).fetch1("trace_times", "trace")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        plot_trace_and_trigger(
            time=trace_times, trace=trace, triggertimes=triggertimes, title=str(key))
