"""Minimal stand-ins for DataJoint relations, to call table methods (make, compute_entry, ...) without a database.

Only the small subset of the relation API used by the table templates is implemented: restriction (``&``),
``proj``, calling the table class, ``fetch1``, ``to_arrays``, ``to_dicts`` and iteration.
"""
import datajoint as dj
import numpy as np


class FakeRel:
    """A relation with fixed rows; restrictions are ignored."""

    def __init__(self, row: dict | None = None, rows: list | None = None, **attrs):
        self.rows = [dict(r) for r in rows] if rows is not None else [dict(row or {})]
        for k, v in attrs.items():
            setattr(self, k, v)

    def __and__(self, other):
        return self

    def __call__(self):
        return self

    def proj(self, *args, **kwargs):
        return self

    def fetch1(self, *attrs):
        if len(self.rows) != 1:
            raise dj.DataJointError(f"fetch1 should only return one tuple. {len(self.rows)} tuples found")
        row = self.rows[0]
        if not attrs:
            return dict(row)
        missing = [a for a in attrs if a not in row]
        if missing:
            raise dj.DataJointError(f"Attributes {missing} not found")
        values = tuple(row[a] for a in attrs)
        return values[0] if len(values) == 1 else values

    def to_arrays(self, *attrs):
        arrays = []
        for a in attrs:
            values = [r[a] for r in self.rows]
            try:
                arr = np.array(values)
            except ValueError:
                arr = np.empty(len(values), dtype=object)
                arr[:] = values
            arrays.append(arr)
        return arrays[0] if len(arrays) == 1 else tuple(arrays)

    def to_dicts(self):
        return [dict(r) for r in self.rows]

    def __iter__(self):
        return iter(self.to_dicts())

    def __len__(self):
        return len(self.rows)


class InsertRecorder:
    """Records ``insert1`` calls."""

    def __init__(self):
        self.inserted = []

    def insert1(self, row, **kwargs):
        self.inserted.append(dict(row))


def make_fake(template_cls, method_names, **attrs):
    """Instance of a plain class that borrows `method_names` from `template_cls` and records inserts."""
    namespace = {name: getattr(template_cls, name) for name in method_names}
    fake_cls = type(f"Fake{template_cls.__name__}", (InsertRecorder,), namespace)
    fake = fake_cls()
    for k, v in attrs.items():
        setattr(fake, k, v)
    return fake


class KeyedRel(FakeRel):
    """Relation whose restriction by a dict selects one of several FakeRel by the value of `field`."""

    def __init__(self, field: str, rels: dict):
        super().__init__()
        self.field = field
        self.rels = rels

    def __and__(self, other):
        if isinstance(other, dict) and self.field in other:
            return self.rels[other[self.field]]
        return self
