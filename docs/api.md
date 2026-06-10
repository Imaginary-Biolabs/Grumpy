# API Reference

Reference documentation is generated from Python docstrings in the `grumpy` package. For narrative tutorials, start at [Home](index.md) and follow the section links at the bottom of each page.

## Top-level API

::: grumpy
    options:
      show_root_heading: false
      heading_level: 3
      members_order: alphabetical
      filters:
        - "!^GrumpyArray$"
        - "!^DType$"
        - "!^OpenDataFrame$"
        - "!^OpenColumn$"
        - "!^CompiledTransform$"
        - "!^compile$"
        - "!^_"

## Core types

::: grumpy._core
    options:
      show_root_heading: false
      heading_level: 3
      allow_inspection: true
      force_inspection: true
      members:
        - GrumpyArray
        - DType
        - GrumpyDataFrame
        - OpenDataFrame
        - OpenColumn
      filters:
        - "!^__"

## Compilation

::: grumpy.compiler
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - compile
        - CompiledTransform

---

**Next:** [Developer](developer.md) — repository layout, implementation notes, and error handling.
