import os
import warnings
from copy import deepcopy
from datetime import datetime

import datajoint as dj
import numpy as np

from djimaging.utils.data_utils import read_config_dict
from djimaging.utils.dj_utils import PlaceholderTable


def find_header_files(data_dir: str) -> list:
    """
    Search for header files in folder in given path.
    :param data_dir: Root folder.
    :return: List of header files.
    """
    os_walk_output = []
    for folder, subfolders, files in os.walk(data_dir):
        if np.any([f.endswith('.ini') for f in files]):
            os_walk_output.append(folder)
    return os_walk_output


class ExperimentTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Header-File location, name and path
        -> self.userinfo_table
        date                        :date                     # date of recording
        exp_num                     :mediumint                # experiment number in a day
        ---
        header_path                 :varchar(255)             # path to header file
        header_name                 :varchar(255)             # name of header file
        """
        return definition

    userinfo_table = PlaceholderTable

    def make(self, key: dict) -> None:
        data_dir, pre_data_dir, raw_data_dir = (self.userinfo_table() & key).fetch1(
            "data_dir", "pre_data_dir", "raw_data_dir")
        self.__add_experiments(key=key, data_dir=data_dir, pre_data_dir=pre_data_dir, raw_data_dir=raw_data_dir,
                               only_new=False, restrictions=None, verboselvl=1, suppress_errors=False)

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 1, suppress_errors: bool = False) -> None:
        """Scan filesystem for new experiments and add them to the database.
        :param restrictions: Restriction to users table, e.g. to scan only for specific user(s)
        :param verboselvl: Print (0) no / (1) only new data / (2) all data information
        :param suppress_errors: Stop on errors or only print?
        """

        if restrictions is None:
            restrictions = dict()

        for row in (self.userinfo_table() & restrictions):
            key = dict()
            key['experimenter'] = row['experimenter']

            if verboselvl > 0:
                print(f"Scanning for experimenter: {key['experimenter']}")

            self.__add_experiments(
                key=key, data_dir=row["data_dir"],
                pre_data_dir=row["pre_data_dir"], raw_data_dir=row["raw_data_dir"],
                only_new=True, restrictions=restrictions, verboselvl=verboselvl, suppress_errors=suppress_errors)

    def __add_experiments(self, key, data_dir, pre_data_dir, raw_data_dir,
                          only_new, restrictions, verboselvl, suppress_errors):

        os_walk_output = find_header_files(data_dir)

        for header_path in os_walk_output:
            try:
                self.__add_experiment(
                    key=key, header_path=header_path, pre_data_dir=pre_data_dir, raw_data_dir=raw_data_dir,
                    only_new=only_new, restrictions=restrictions, verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    print("Suppressed Error:", e, '\n\tfor key:', key)
                else:
                    raise e

    def __add_experiment(self, key, header_path, pre_data_dir, raw_data_dir, only_new, restrictions, verboselvl):

        if verboselvl > 0:
            print('\theader_path:', header_path)

        header_names = [s for s in os.listdir(header_path) if s.endswith('.ini') and (not s.startswith('.'))]
        if len(header_names) != 1:
            # Check if they are equal
            header_dicts = [read_config_dict(header_path + "/" + hn) for hn in header_names]
            if any([hd != header_dicts[0] for hd in header_dicts[1:]]):
                raise ValueError(f'Found {len(header_names)} header files in {header_path} with differences.')

        header_name = header_names[0]

        if verboselvl > 0:
            print('\t\theader_name:', header_name)

        header_dict = read_config_dict(header_path + "/" + header_name)

        primary_key = deepcopy(key)

        try:
            primary_key["date"] = datetime.strptime(header_path.split("/")[-2], '%Y%m%d')
        except ValueError:
            if verboselvl >= 0:
                warnings.warn(f'Failed to convert `{header_path.split("/")[-2]}` to date. Skip this folder.')
            return

        primary_key["exp_num"] = int(header_path.split("/")[-1])

        if only_new:
            search = (self & restrictions & primary_key)
            if len(search) > 0:
                if verboselvl > 1:
                    print('\t\tAlready present:', primary_key)
                return

        for k, v in restrictions.items():
            if k not in primary_key:
                warnings.warn(f'Restriction not in primary key, ignoring {k}')
                continue

            if primary_key[k] != v:
                if verboselvl > 0:
                    warnings.warn(f'Skipping {primary_key} because of restriction: {k}={v}.')
                return

        pre_data_path = header_path + "/" + pre_data_dir + "/"
        assert os.path.isdir(pre_data_path), f"{pre_data_dir} not found {header_path}"

        raw_data_path = header_path + "/" + raw_data_dir + "/"
        if not os.path.isdir(raw_data_path):
            print(f"WARNING: Folder {raw_data_dir} not found in {header_path}")

        exp_key = deepcopy(primary_key)
        exp_key["header_path"] = header_path + "/"
        exp_key["header_name"] = header_name

        # Populate ExpInfo table for this experiment
        expinfo_key = deepcopy(primary_key)
        expinfo_key["eye"] = header_dict["eye"]
        expinfo_key["projname"] = header_dict["projname"]
        expinfo_key["setupid"] = header_dict["setupid"]
        expinfo_key["prep"] = header_dict["prep"]
        expinfo_key["preprem"] = header_dict["preprem"]
        expinfo_key["darkadapt_hrs"] = header_dict["darkadapt_hrs"]
        expinfo_key["slicethickness_um"] = header_dict["slicethickness_um"]
        expinfo_key["bathtemp_degc"] = header_dict["bathtemp_degc"]
        expinfo_key["prepwmorient"] = header_dict["prepwmorient"] if header_dict["prepwmorient"] != "" else -1

        # find optic disk information if available
        odx, ody, odz, od_ini_flag = 0, 0, 0, 0

        odpos_string = header_dict["prepwmopticdiscpos"].strip('() ')
        if len(odpos_string) > 0:
            odpos_list = odpos_string.split(";" if ';' in odpos_string else ',')

            if len(odpos_list) >= 2:
                if len(odpos_list[0]) > 0 or len(odpos_list[1]) > 0:

                    odx = float(odpos_list[0])
                    ody = float(odpos_list[1])
                    od_ini_flag = 1

                    try:
                        odz = float(odpos_list[2])
                    except ValueError:
                        odz = 0
                    except IndexError:
                        odz = 0

        expinfo_key["odx"] = odx
        expinfo_key["ody"] = ody
        expinfo_key["odz"] = odz
        expinfo_key["od_ini_flag"] = od_ini_flag

        # Populate Animal table for this experiment
        animal_key = deepcopy(primary_key)
        animal_key["genline"] = header_dict["genline"]
        animal_key["genbkglinerem"] = header_dict["genbkglinerem"]
        animal_key["genline_reporter"] = header_dict["genline_reporter"]
        animal_key["genline_reporterrem"] = header_dict["genline_reporterrem"]
        animal_key["animspecies"] = header_dict["animspecies"].lower() if header_dict["animspecies"] != "" else "mouse"
        animal_key["animgender"] = header_dict["animgender"]
        animal_key["animdob"] = header_dict["animdob"]
        animal_key["animrem"] = header_dict["animrem"]

        # Populate Indicator table for this experiment
        indicator_key = deepcopy(primary_key)
        indicator_key["isepored"] = header_dict["isepored"]
        indicator_key["eporrem"] = header_dict["eporrem"]
        indicator_key["epordye"] = header_dict["epordye"]
        indicator_key["isvirusinject"] = header_dict["isvirusinject"]
        indicator_key["virusvect"] = header_dict["virusvect"]
        indicator_key["virusserotype"] = header_dict["virusserotype"]
        indicator_key["virustransprotein"] = header_dict["virustransprotein"]
        indicator_key["virusinjectq"] = header_dict["virusinjectq"]
        indicator_key["virusinjectrem"] = header_dict["virusinjectrem"]
        indicator_key["tracer"] = header_dict["tracer"]
        indicator_key["isbraininject"] = header_dict["isbraininject"]
        indicator_key["braininjectrem"] = header_dict["braininjectrem"]
        indicator_key["braininjectq"] = header_dict["braininjectq"]

        # Populate Pharmacology table for this experiment
        pharminfo_key = deepcopy(primary_key)
        drug = header_dict.get("pharmdrug", "").lower()
        if drug.lower() in ["", "none"]:
            drug = "none"
        pharminfo_key['drug'] = drug
        pharminfo_key["pharmaflag"] = 0 if drug == 'none' else 1
        pharminfo_key['pharmconc'] = header_dict.get("pharmdrugconc_um", "")
        pharminfo_key['preapptime'] = header_dict.get("pretime", "")
        pharminfo_key['pharmcom'] = header_dict.get("pharmrem", "")

        if verboselvl > 0:
            print('\t\tAdding:', primary_key)

        self.insert1(exp_key, allow_direct_insert=True)
        self.ExpInfo().insert1(expinfo_key)
        self.Animal().insert1(animal_key)
        self.Indicator().insert1(indicator_key)
        self.PharmInfo().insert1(pharminfo_key)

    class ExpInfo(dj.Part):
        @property
        def definition(self):
            definition = """
               # General preparation details set by user in preprocessing
               -> master
               ---
               eye                :enum("left", "right", "unknown") # left or right eye of the animal
               projname           :varchar(255)                     # name of experimental project
               setupid            :varchar(255)                     # setup 1-3
               prep="wholemount"  :enum("wholemount", "slice")      # preparation type of the retina
               preprem            :varchar(255)                     # comments on the preparation
               darkadapt_hrs      :varchar(255)                     # time spent dark adapting animal before disection
               slicethickness_um  :varchar(255)                     # thickness of each slice in slice preparation
               bathtemp_degc      :varchar(255)                     # temperature of bath chamber
               prepwmorient       :smallint         # retina orientation in chamber (0° = dorsal away from experimenter)
               odx                :float            # x location of optic disk as read in from .ini file
               ody                :float            # y location of optic disk as read in from .ini file (if available)
               odz                :float            # z location of optic disk as read in from .ini file (if available)
               od_ini_flag        :tinyint unsigned # flag (0, 1) indicating whether (1) or whether not (0)
                                                    # the optic disk position was documented in .ini file and
                                                    # is valid to use
               """
            return definition

    class Animal(dj.Part):
        @property
        def definition(self):
            definition = """
               # Animal info and genetic background set by user in preprocessing
               -> master
               ---
               genline                   :varchar(255)                     # Genetic background line
               genbkglinerem             :varchar(255)                     # Comments about background line
               genline_reporter          :varchar(255)                     # Genetic reporter line
               genline_reporterrem       :varchar(255)                     # Comments about reporter line
               animspecies="mouse"       :varchar(255)                     # animal species
               animgender                :varchar(255)                     # gender.
               animdob                   :varchar(255)                     # Whether to have this or DOB?
               animrem                   :varchar(255)                     # Comments about animal
               """
            return definition

    class Indicator(dj.Part):
        @property
        def definition(self):
            definition = """
               # Indicator used for imaging set by user in preprocessing
               -> master
               ---
               isepored                    :varchar(255)     # whether the retina was electroporated
               eporrem                     :varchar(255)     # comments about the electroporation
               epordye                     :varchar(255)     # which dye was used for the electroporation
               isvirusinject               :varchar(5)       # whether the retina was injected
               virusvect                   :varchar(255)     # what vector was used in the injection
               virusserotype               :varchar(255)     # what serotype was used in the injection
               virustransprotein           :varchar(255)     # the viral transprotein
               virusinjectrem              :varchar(255)     # comments about the injection
               virusinjectq                :varchar(255)     # numerical rating of the injection quality
               isbraininject               :varchar(5)       # whether the retina was injected
               tracer                      :varchar(255)     # which tracer has been used in the brain injection
               braininjectq                :varchar(255)     # numerical rating of the brain injection quality
               braininjectrem              :varchar(255)     # comments on the brain injection
               """
            return definition

    class PharmInfo(dj.Part):
        @property
        def definition(self):
            definition = """
            # Pharmacology Info
            -> master
            ---
            pharmaflag      :tinyint unsigned # 1 there was pharma, 0 no pharma
            drug            :varchar(255)     # which drug was applied
            pharmconc       :varchar(255)     # concentration used in micromolar
            preapptime      :varchar(255)     # preapplication time
            pharmcom        :varchar(255)     # experimenter comments
            """
            return definition

