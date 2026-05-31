"""Helpers for building NumPy-style docstrings."""

from __future__ import annotations

import types
from typing import Iterable


def doc(
    summary: str,
    *,
    params: Iterable[str] = (),
    returns: str | None = None,
    examples: Iterable[str] = (),
) -> str:
    """Build a NumPy-style docstring."""
    lines = [summary.rstrip(), ""]
    params = list(params)
    if params:
        lines.extend(["Parameters", "----------", *params, ""])
    if returns:
        lines.extend(["Returns", "-------", returns.rstrip(), ""])
    examples = list(examples)
    if examples:
        lines.extend(["Examples", "--------", *examples, ""])
    return "\n".join(lines).rstrip() + "\n"


def inject(obj, name: str, text: str) -> None:
    """Set ``__doc__`` on ``obj.name``, wrapping PyO3 methods when needed."""
    try:
        target = getattr(obj, name)
    except AttributeError:
        return
    try:
        target.__doc__ = text
        return
    except (AttributeError, TypeError):
        pass

    if name == "__doc__":
        try:
            obj.__doc__ = text
        except (AttributeError, TypeError):
            pass
        return

    raw = target
    if isinstance(raw, property):
        setattr(
            obj,
            name,
            property(raw.fget, raw.fset, raw.fdel, text),
        )
        return

    if isinstance(raw, types.GetSetDescriptorType):
        def _getter(self, _desc=raw, _cls=obj):
            return _desc.__get__(self, _cls)

        setattr(obj, name, property(_getter, doc=text))
        return

    if isinstance(raw, (classmethod, staticmethod)):
        raw = raw.__func__

    def _wrapper(*args, **kwargs):
        return raw(*args, **kwargs)

    _wrapper.__doc__ = text
    _wrapper.__name__ = name
    setattr(obj, name, _wrapper)


def inject_many(obj, mapping: dict[str, str]) -> None:
    for name, text in mapping.items():
        inject(obj, name, text)
