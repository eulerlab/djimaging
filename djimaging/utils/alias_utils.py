import os


def check_shared_alias_str(str1: str, str2: str, case_sensitive: bool = False, sep: str = '_') -> bool:
    """Check whether two strings share any alias tokens after splitting.

    Parameters
    ----------
    str1 : str
        First string to compare.
    str2 : str
        Second string to compare.
    case_sensitive : bool, optional
        If False (default), comparison is case-insensitive.
    sep : str, optional
        Separator used to split the strings into tokens. Default is '_'.

    Returns
    -------
    bool
        True if the two strings share at least one common non-empty token.
    """
    if not case_sensitive:
        str1 = str1.lower()
        str2 = str2.lower()

    list1 = str1.split(sep)
    list2 = str2.split(sep)

    return check_shared_alias_list(list1, list2)


def check_shared_alias_list(list1: list, list2: list) -> bool:
    """Check whether two lists share any common non-empty elements.

    Parameters
    ----------
    list1 : list
        First list of tokens.
    list2 : list
        Second list of tokens.

    Returns
    -------
    bool
        True if the two lists have at least one common non-empty element.
    """
    set1 = set(list1)
    set2 = set(list2)

    if '' in set1:
        set1.remove('')
    if '' in set2:
        set2.remove('')

    return bool(set1 & set2)


def match_file(file: str, pattern: str, pattern_loc: int | None = None, ftype: str | None = None,
               case_sensitive: bool = False, sep: str = '_', mode: str = 'full') -> bool:
    """Check whether a filename matches a given pattern.

    Parameters
    ----------
    file : str
        Path or filename to test.
    pattern : str
        Pattern string to match against (split by `sep`).
    pattern_loc : int | None, optional
        If provided, only the token at this position in the filename is checked.
    ftype : str | None, optional
        File extension to strip before matching. If None, inferred from `file`.
    case_sensitive : bool, optional
        If False (default), comparison is case-insensitive.
    sep : str, optional
        Separator used to tokenise the filename and pattern. Default is '_'.
    mode : str, optional
        Matching mode: ``'full'`` requires exact token equality; ``'contains'``
        requires the pattern token to be a substring. Default is ``'full'``.

    Returns
    -------
    bool
        True if the file matches the pattern according to the specified mode.
    """
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


def match_files(files: list, pattern: str, pattern_loc: int | None = None, ftype: str | None = None,
                case_sensitive: bool = False, sep: str = '_', mode: str = 'full') -> list:
    """Filter a list of files to those matching a given pattern.

    Parameters
    ----------
    files : list
        List of file paths or filenames to filter.
    pattern : str
        Pattern string to match against.
    pattern_loc : int | None, optional
        If provided, only the token at this position in each filename is checked.
    ftype : str | None, optional
        File extension to strip before matching. If None, inferred from each file.
    case_sensitive : bool, optional
        If False (default), comparison is case-insensitive.
    sep : str, optional
        Separator used to tokenise filenames and the pattern. Default is '_'.
    mode : str, optional
        Matching mode: ``'full'`` or ``'contains'``. Default is ``'full'``.

    Returns
    -------
    list
        Subset of `files` whose names match the pattern.
    """
    matching_files = [file for file in files
                      if match_file(file, pattern=pattern, pattern_loc=pattern_loc,
                                    ftype=ftype, case_sensitive=case_sensitive, sep=sep, mode=mode)]
    return matching_files


def get_field_files(folder: str, field: str, field_loc: int | None = None,
                    ftype: str = 'h5', incl_hidden: bool = False) -> list:
    """Return files in a folder that match a given field name pattern.

    Parameters
    ----------
    folder : str
        Directory path to search.
    field : str
        Field name pattern to match.
    field_loc : int | None, optional
        Token position within the filename to restrict matching to.
    ftype : str, optional
        File extension to look for. Default is ``'h5'``.
    incl_hidden : bool, optional
        If False (default), files starting with ``'.'`` are excluded.

    Returns
    -------
    list
        Sorted list of matching filenames (not full paths).
    """
    files = [file for file in sorted(os.listdir(folder))
             if file.endswith(f'.{ftype}') and (incl_hidden or not file.startswith('.'))]
    matching_files = match_files(
        files, pattern=field, pattern_loc=field_loc, ftype=ftype, case_sensitive=False, sep='_', mode='full')
    return matching_files
