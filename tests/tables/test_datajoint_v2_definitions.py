import importlib
import inspect
import pkgutil
import re

import datajoint as dj
from datajoint.declare import NATIVE_TYPES, attribute_parser, is_foreign_key, match_type

import djimaging.tables


def _table_classes():
    classes = set()
    for module_info in pkgutil.walk_packages(djimaging.tables.__path__, djimaging.tables.__name__ + "."):
        module = importlib.import_module(module_info.name)
        for _, table_class in inspect.getmembers(module, inspect.isclass):
            if table_class.__module__ == module.__name__ and issubclass(table_class, dj.Table):
                classes.add(table_class)
                classes.update(
                    value
                    for value in table_class.__dict__.values()
                    if inspect.isclass(value) and issubclass(value, dj.Table)
                )
    return classes


def _definition(table_class):
    if table_class.__name__ in {"PresentationTemplate", "HighResTemplate"}:
        field_flags = type(
            "FieldFlags",
            (),
            {"incl_region": True, "incl_cond1": True, "incl_cond2": False, "incl_cond3": False},
        )
        table_class = type(f"Test{table_class.__name__}", (table_class,), {"field_table": field_flags})
    return table_class().definition


def test_every_table_definition_uses_datajoint_v2_types():
    definitions = {table_class.__name__: _definition(table_class) for table_class in _table_classes()}
    assert len(definitions) >= 75

    for table_name, definition in definitions.items():
        for line in re.split(r"\s*\n\s*", definition.strip()):
            if (
                not line
                or line.startswith(("#", "---"))
                or is_foreign_key(line)
                or re.match(r"^(unique\s+)?index\s*\(", line, re.I)
            ):
                continue

            parsed = attribute_parser.parse_string(line + "#", parse_all=True)
            attribute_type = parsed.type.strip()
            category = match_type(attribute_type)
            assert category not in NATIVE_TYPES or attribute_type.lower() == "time", (
                table_name,
                parsed.name,
                attribute_type,
            )
