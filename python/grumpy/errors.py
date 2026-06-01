"""Structured error reporting for the Grumpy Python layer.

Rust kernels format errors as::

    grumpy.<Code>: <summary>
      cause: …
      context_key: …
      fix: …

Python code should follow the same shape so users get consistent, actionable
messages from constructors, streaming, and the compiler.
"""

from __future__ import annotations

from typing import Any


def format_grumpy_error(
    code: str,
    summary: str,
    *,
    cause: str | None = None,
    fix: str | None = None,
    **context: Any,
) -> str:
    """Build a multi-line Grumpy error string."""
    lines = [f"grumpy.{code}: {summary}"]
    if cause:
        lines.append(f"  cause: {cause}")
    for key, value in context.items():
        lines.append(f"  {key}: {value}")
    if fix:
        lines.append(f"  fix: {fix}")
    return "\n".join(lines)


def raise_grumpy_error(
    code: str,
    summary: str,
    *,
    cause: str | None = None,
    fix: str | None = None,
    exc: type[Exception] = ValueError,
    **context: Any,
) -> None:
    """Raise a structured Grumpy error (subclass of ValueError by default)."""
    raise exc(
        format_grumpy_error(code, summary, cause=cause, fix=fix, **context)
    ) from None


def arg_invalid(name: str, detail: str, *, fix: str) -> None:
    raise_grumpy_error(
        "ArgumentInvalid",
        f"invalid argument '{name}': {detail}",
        fix=fix,
    )


def arg_one_of(name: str, value: object, allowed: tuple[str, ...]) -> None:
    allowed_s = ", ".join(repr(a) for a in allowed)
    raise_grumpy_error(
        "ArgumentInvalid",
        f"invalid argument '{name}': got {value!r}",
        cause=f"expected one of: {allowed_s}.",
        fix=f"pass {name} as one of {allowed_s}.",
    )


def index_out_of_range(index: int, length: int, *, at: str = "on this axis") -> None:
    raise_grumpy_error(
        "IndexOutOfBounds",
        f"index {index} is out of range for length {length} {at}",
        cause=f"valid indices are 0..{length}.",
        fix=f"use an index in [0, {length}) or check len(stream) / batch bounds.",
        exc=IndexError,
    )
