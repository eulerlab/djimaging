"""Inspired by: sinzlab/nnfabrik: https://github.com/sinzlab/nnfabrik """

import importlib


def extract_class_info(full_path):
    """
    Separates a fully qualified name into module path and class name

    Args:
        full_path (str): The fully qualified name (e.g. 'package.module.ClassName')

    Returns:
        tuple: (module_path, class_name)
    """
    components = full_path.split('.')
    class_identifier = components[-1]
    module_path = '.'.join(components[:-1])
    return module_path, class_identifier


def load_class(full_module_path, target_class):
    """
    Dynamically loads a class from a module

    Args:
        full_module_path (str): The path to the module
        target_class (str): The name of the class to load

    Returns:
        class: The loaded class object
    """
    loaded_module = importlib.import_module(full_module_path)
    return getattr(loaded_module, target_class)
