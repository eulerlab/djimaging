from abc import abstractmethod

import datajoint as dj


class PretrainedClassifierTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.classifier_training_data_table
        classifier_params_hash : varchar(63)  # Make consistent with trained classifiers
        ---
        classifier_file : varchar(255)
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.classifier_training_data_table().proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def classifier_training_data_table(self):
        pass

    def add(self, training_data_hash, classifier_params_hash, classifier_file):
        self.insert1(dict(
            training_data_hash=training_data_hash,
            classifier_params_hash=classifier_params_hash,
            classifier_file=classifier_file))
