from abc import abstractmethod
from datetime import date, datetime
from typing import Optional

import datajoint as dj


class AnimalAgeTemplate(dj.Computed):
    """DataJoint computed table template that stores the age of an animal in days."""

    database = ""

    @property
    def definition(self) -> str:
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

    def make(self, key: dict) -> None:
        """Compute and insert the animal age in days for a given experiment key.

        Args:
            key: DataJoint primary key dict identifying the experiment entry.
        """
        dod, dob = (self.experiment_table.Animal & key).fetch1('date', 'animdob')

        # convert string to date
        dod = date_str_to_date(dod) if isinstance(dod, str) else dod
        dob = date_str_to_date(dob) if isinstance(dob, str) else dob

        if dob is not None and dod is not None:
             # Compute age in days
            age = (dod - dob).days
            self.insert1(dict(key, age=age))

def date_str_to_date(date_str: str) -> Optional[date]:
    """Parse a date string into a ``datetime.date`` object.

    Tries several common date formats and returns ``None`` if the string is
    empty, unparseable, or the parsed year is outside 2000–2050.

    Args:
        date_str: A date string to parse (e.g. ``"20210101"`` or ``"01.01.2021"``).

    Returns:
        A ``datetime.date`` instance, or ``None`` if parsing fails.
    """
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




