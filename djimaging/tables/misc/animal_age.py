from abc import abstractmethod
from datetime import datetime

import datajoint as dj


class AnimalAgeTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.experiment_table
        ---
        age : int  # age of the animal in days
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.experiment_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    def make(self, key):
        dod, dob = (self.experiment_table.Animal & key).fetch1('date', 'animdob')

        # convert string to date
        dod = date_str_to_date(dod) if isinstance(dod, str) else dod
        dob = date_str_to_date(dob) if isinstance(dob, str) else dob

        if dob is not None and dod is not None:
             # Compute age in days
            age = (dod - dob).days
            self.insert1(dict(key, age=age))

def date_str_to_date(date_str: str):
    if not date_str:  # Handle empty entries
        return None
    date_formats = [
        "%Y%m%d",
        "%Y-%m-%d",
        "%Y_%m_%d",
        "%d.%m.%Y",
        "%Y.%m.%d",
        "%d-%m-%Y"
    ]
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            # Test if year is not between 2000 and 2050
            if date_obj.year < 2000 or date_obj.year > 2050:
                raise ValueError
            return date_obj.date()
        except ValueError:
            continue
    return None




