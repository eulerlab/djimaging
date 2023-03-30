import os


def check_shared_alias_str(str1, str2, case_sensitive=False, sep='_'):
    if not case_sensitive:
        str1 = str1.lower()
        str2 = str2.lower()

    list1 = str1.split(sep)
    list2 = str2.split(sep)

    return check_shared_alias_list(list1, list2)


def check_shared_alias_list(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    if '' in set1:
        set1.remove('')
    if '' in set2:
        set2.remove('')

    return bool(set1 & set2)


def match_file(file, pattern, pattern_loc=None, ftype=None, case_sensitive=False, sep='_', mode='full'):
    file = os.path.split(file)[1]
    if not case_sensitive:
        pattern = pattern.lower()
        file = file.lower()

    if ftype is None:
        ftype = os.path.splitext(file)[1][1:]

    infos = file.replace(f'.{ftype}', '').split(sep)
    if pattern_loc is not None:
        if len(infos) <= pattern_loc:
            return False
        infos = [infos[pattern_loc]]

    for info in infos:
        for a in pattern.split(sep):
            if mode == 'full' and a == info:
                return True
            elif mode == 'contains' and a in info:
                return True
    return False


def match_files(files, pattern, pattern_loc=None, ftype=None, case_sensitive=False, sep='_', mode='full'):
    matching_files = [file for file in files
                      if match_file(file, pattern=pattern, pattern_loc=pattern_loc,
                                    ftype=ftype, case_sensitive=case_sensitive, sep=sep, mode=mode)]
    return matching_files


def get_field_files(folder, field, field_loc=None, ftype='h5', incl_hidden=False):
    files = [file for file in sorted(os.listdir(folder))
             if file.endswith(f'.{ftype}') and (incl_hidden or not file.startswith('.'))]
    matching_files = match_files(
        files, pattern=field, pattern_loc=field_loc, ftype=ftype, case_sensitive=False, sep='_', mode='full')
    return matching_files
