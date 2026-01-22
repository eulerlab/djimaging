import os
import warnings
from abc import abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Optional

import datajoint as dj

from djimaging.utils.scanm.read_h5_utils import read_config_dict
from djimaging.utils.filesystem_utils import find_folders_with_file_of_type


class ExperimentTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Header-File location, name and path
        -> self.userinfo_table
        date                        :date                     # date of recording
        exp_num                     :tinyint unsigned         # experiment number in a day
        ---
        header_path                 :varchar(191)             # path to header file
        header_name                 :varchar(63)              # name of header file
        """
        return definition

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    def make(self, key: dict) -> None:
        data_dir, pre_data_dir, raw_data_dir = (self.userinfo_table() & key).fetch1(
            "data_dir", "pre_data_dir", "raw_data_dir")
        self.add_experiments(key=key, data_dir=data_dir, pre_data_dir=pre_data_dir, raw_data_dir=raw_data_dir,
                             only_new=False, restrictions=None, verboselvl=1, suppress_errors=False)

    @property
    def key_source(self):
        try:
            return self.userinfo_table.proj()
        except (AttributeError, TypeError):
            pass

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 1, suppress_errors: bool = False,
                          restr_headers: Optional[list] = None) -> None:
        """Scan filesystem for new experiments and add them to the database.
        :param restrictions: Restriction to users table, e.g. to scan only for specific user(s)
        :param verboselvl: Print (0) no / (1) only new data / (2) all data information
        :param suppress_errors: Stop on errors or only print?
        :param restr_headers: List of headers to be included
        """

        if restrictions is None:
            restrictions = dict()

        for key in (self.key_source & restrictions).fetch(as_dict=True):
            if verboselvl > 0:
                print(f"Scanning for experimenter: {key['experimenter']}")

            data_dir = (self.userinfo_table & key).fetch1("data_dir")
            pre_data_dir = (self.userinfo_table & key).fetch1("pre_data_dir")
            raw_data_dir = (self.userinfo_table & key).fetch1("raw_data_dir")

            self.add_experiments(
                key=key, data_dir=data_dir, pre_data_dir=pre_data_dir, raw_data_dir=raw_data_dir,
                only_new=True, restrictions=restrictions, restr_headers=restr_headers,
                verboselvl=verboselvl, suppress_errors=suppress_errors)

    def add_experiments(self, key, data_dir, pre_data_dir, raw_data_dir,
                        only_new, restrictions, restr_headers=None, verboselvl=0, suppress_errors=False):

        header_paths = find_folders_with_file_of_type(data_dir, ending='.ini', ignore_hidden=True)

        for header_path in header_paths:
            if restr_headers is not None and header_path not in restr_headers:
                if verboselvl > 1:
                    print('\t\t\tSkipping:', header_path)
                continue
            try:
                self.add_experiment(
                    key=key, header_path=header_path, pre_data_dir=pre_data_dir, raw_data_dir=raw_data_dir,
                    only_new=only_new, restrictions=restrictions, verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    print("Suppressed Error:", e, '\n\tfor key:', key)
                else:
                    raise e

    def add_experiment(self, key, header_path, pre_data_dir, raw_data_dir, only_new, restrictions, verboselvl):

        if verboselvl > 0:
            print('\theader_path:', header_path)

        header_names = [s for s in os.listdir(header_path) if s.endswith('.ini') and (not s.startswith('.'))]
        if len(header_names) > 1:
            # Check if they are equal
            header_dicts = [read_config_dict(header_path + "/" + hn) for hn in header_names]
            if any([hd != header_dicts[0] for hd in header_dicts[1:]]):
                raise ValueError(f'Found {len(header_names)} header files in {header_path} with differences.')
        elif len(header_names) == 0:
            raise ValueError(f'No header file found in {header_path}. This should not happen.')

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

        exp_num = int(header_path.split("/")[-1])
        primary_key["exp_num"] = exp_num

        if only_new:
            search = (self & restrictions & primary_key)
            if len(search) > 0:
                if verboselvl > 1:
                    print('\t\tAlready present:', primary_key)
                return

        pre_data_path = header_path + "/" + pre_data_dir + "/"
        if not os.path.isdir(pre_data_path):
            warnings.warn(f"Folder `{pre_data_dir}` not found in {header_path}")

        raw_data_path = header_path + "/" + raw_data_dir + "/"
        if not os.path.isdir(raw_data_path):
            warnings.warn(f"Folder `{raw_data_dir}` not found in {header_path}")

        exp_entry = deepcopy(primary_key)
        exp_entry["header_path"] = header_path + "/"
        exp_entry["header_name"] = header_name

        # Populate ExpInfo table for this experiment
        info_entry = deepcopy(primary_key)
        info_keys = [
            "projname", "setupid", "prep", "preprem", "darkadapt_hrs", "slicethickness_um", "bathtemp_degc"]
        info_entry.update({k: header_dict.get(k, "") for k in info_keys})

        if info_entry["prep"] == "":
            info_entry["prep"] = "wholemount"

        eye = header_dict.get("eye", "unknown")
        if 'eye' == '':
            eye = 'unknown'

        if (
                (eye in ["Right", "right"] and (exp_num == 1 or 'left' in header_name.lower())) or
                (eye in ["Left", "left"] and (exp_num == 2 or 'right' in header_name.lower()))
        ):
            warnings.warn(
                f"Eye is set to {eye} in .ini file, "
                f"but exp_num is {exp_num} and header_file_name is '{header_name}'. "
                f"Use exp_num=1 for left eye and exp_num=2 for right eye. "
                f"To overwrite this, use all-caps in .ini file which is then used.")
        info_entry["eye"] = eye

        prepwmorient_str = header_dict.get("prepwmorient", "")
        if prepwmorient_str == "":
            prepwmorient = 0  # Default to zero if not specified
        elif prepwmorient_str == "unknown" or prepwmorient_str == "111":
            prepwmorient = -1
        else:
            prepwmorient = int(prepwmorient_str)
        info_entry["prepwmorient"] = prepwmorient

        # find optic disk information if available
        odx, ody, odz, od_ini_flag = 0, 0, 0, 0

        odpos_string = header_dict.get("prepwmopticdiscpos", "").strip('() ')
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

        info_entry["odx"] = odx
        info_entry["ody"] = ody
        info_entry["odz"] = odz
        info_entry["od_ini_flag"] = od_ini_flag

        # Create animal entry
        animal_entry = deepcopy(primary_key)
        animal_keys = [
            "genline", "genbkglinerem", "genline_reporter", "genline_reporterrem",
            "animspecies", "animgender", "animdob", "animrem"]
        animal_entry.update({k: header_dict.get(k, "") for k in animal_keys})

        # Create indicator entry
        indicator_entry = deepcopy(primary_key)
        indicator_keys = [
            "isepored", "eporrem", "epordye",
            "isvirusinject", "virusvect", "virusserotype", "virustransprotein", "virusinjectq", "virusinjectrem",
            "tracer", "isbraininject", "braininjectq", "braininjectrem"]
        indicator_entry.update({k: header_dict.get(k, "") for k in indicator_keys})

        # Create pharmacology entry
        pharminfo_entry = deepcopy(primary_key)
        drug = header_dict.get("pharmdrug", "").lower()
        if drug.lower() in ["", "none"]:
            drug = "none"
        pharminfo_entry['drug'] = drug
        pharminfo_entry["pharmaflag"] = 0 if drug == 'none' else 1
        pharminfo_entry['pharmconc'] = header_dict.get("pharmdrugconc_um", "")
        pharminfo_entry['preapptime'] = header_dict.get("pretime", "")
        pharminfo_entry['pharmcom'] = header_dict.get("pharmrem", "")

        if verboselvl > 0:
            print('\t\tAdding:', primary_key)

        self.insert1(exp_entry, allow_direct_insert=True)
        self.ExpInfo().insert1(info_entry)
        self.Animal().insert1(animal_entry)
        self.Indicator().insert1(indicator_entry)
        self.PharmInfo().insert1(pharminfo_entry)

    class ExpInfo(dj.Part):
        @property
        def definition(self):
            definition = """
               # General preparation details set by user in preprocessing
               -> master
               ---
               eye                :enum("left", "right", "unknown") # left or right eye of the animal
               projname           :varchar(191)                     # name of experimental project
               setupid            :varchar(191)                     # setup 1-3
               prep="wholemount"  :enum("wholemount", "slice")      # preparation type of the retina
               preprem            :varchar(191)                     # comments on the preparation
               darkadapt_hrs      :varchar(191)                     # time spent dark adapting animal before disection
               slicethickness_um  :varchar(191)                     # thickness of each slice in slice preparation
               bathtemp_degc      :varchar(191)                     # temperature of bath chamber
               prepwmorient       :smallint         # retina orientation in chamber (0° = dorsal away from experimenter). Defaults to 0. Use -1, 111 or "unkown" to encode unknown.
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
               genline                   :varchar(191)                     # Genetic background line
               genbkglinerem             :varchar(191)                     # Comments about background line
               genline_reporter          :varchar(191)                     # Genetic reporter line
               genline_reporterrem       :varchar(191)                     # Comments about reporter line
               animspecies="mouse"       :varchar(191)                     # animal species
               animgender                :varchar(191)                     # gender.
               animdob                   :varchar(191)                     # Whether to have this or DOB?
               animrem                   :varchar(191)                     # Comments about animal
               """
            return definition

    class Indicator(dj.Part):
        @property
        def definition(self):
            definition = """
               # Indicator used for imaging set by user in preprocessing
               -> master
               ---
               isepored                    :varchar(191)     # whether the retina was electroporated
               eporrem                     :varchar(191)     # comments about the electroporation
               epordye                     :varchar(191)     # which dye was used for the electroporation
               isvirusinject               :varchar(63)       # whether the retina was injected
               virusvect                   :varchar(191)     # what vector was used in the injection
               virusserotype               :varchar(191)     # what serotype was used in the injection
               virustransprotein           :varchar(191)     # the viral transprotein
               virusinjectrem              :varchar(191)     # comments about the injection
               virusinjectq                :varchar(191)     # numerical rating of the injection quality
               isbraininject               :varchar(63)       # whether the retina was injected
               tracer                      :varchar(191)     # which tracer has been used in the brain injection
               braininjectq                :varchar(191)     # numerical rating of the brain injection quality
               braininjectrem              :varchar(191)     # comments on the brain injection
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
            drug            :varchar(191)     # which drug was applied
            pharmconc       :varchar(191)     # concentration used in micromolar
            preapptime      :varchar(191)     # preapplication time
            pharmcom        :varchar(191)     # experimenter comments
            """
            return definition
