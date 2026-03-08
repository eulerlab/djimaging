"""Inspired by: sinzlab/nnfabrik: https://github.com/sinzlab/nnfabrik """

import importlib


def extract_class_info(full_path: str) -> tuple[str, str]:
    """Separate a fully qualified name into module path and class name.

    Parameters
    ----------
    full_path : str
        Fully qualified name, e.g. ``'package.module.ClassName'``.

    Returns
    -------
    tuple[str, str]
        ``(module_path, class_name)`` where ``module_path`` is everything
        before the last dot and ``class_name`` is the final component.
    """
    components = full_path.split('.')
    class_identifier = components[-1]
    module_path = '.'.join(components[:-1])
    return module_path, class_identifier


def load_class(full_module_path: str, target_class: str) -> type:
    """Dynamically load a class from a module.

    Parameters
    ----------
    full_module_path : str
        Dotted import path to the module containing the class.
    target_class : str
        Name of the class attribute to retrieve from the module.

    Returns
    -------
    type
        The class object loaded from the specified module.
    """
    loaded_module = importlib.import_module(full_module_path)
    return getattr(loaded_module, target_class)
