from abc import abstractmethod

from djimaging.tables.response.response_quality import RepeatQITemplate


class SineSpotQITemplate(RepeatQITemplate):
    _stim_family = "sinespot"
    _stim_name = "sinespot"

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass
