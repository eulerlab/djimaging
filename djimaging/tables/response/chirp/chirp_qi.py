from abc import abstractmethod

from djimaging.tables.response.response_quality import RepeatQITemplate


class ChirpQITemplate(RepeatQITemplate):
    _stim_family = "chirp"
    _stim_name = "chirp"

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass
