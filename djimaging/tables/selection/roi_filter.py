import datajoint as dj

from djimaging.utils.dj_utils import make_hash


class CellFilterParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        cell_filter_params_hash         :  varchar(32)         # hash of the classifier params config
        ---
        qi_thres_chirp                  :  float               # QI threshold for full-field chirp response
        qi_thres_bar                    :  float               # QI threshold for moving bar response
        cell_selection_constraint       :  enum("and", "or")   # constraint flag (and, or) for QI
        condition = 'control'           :  varchar(255)        # Condition to classify. Empty strings = all conditions. 
        """
        return definition

    def add_parameters(self, condition: str, qi_thres_chirp: float, qi_thres_bar: float,
                       cell_selection_constraint: str, skip_duplicates: bool = False) -> None:
        insert_dict = dict(qi_thres_chirp=qi_thres_chirp, qi_thres_bar=qi_thres_bar, condition=condition,
                           cell_selection_constraint=cell_selection_constraint)
        cell_filter_params_hash = make_hash(insert_dict)
        insert_dict.update(dict(cell_filter_params_hash=cell_filter_params_hash))
        self.insert1(insert_dict, skip_duplicates=skip_duplicates)
