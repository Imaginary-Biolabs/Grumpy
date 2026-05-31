# API Reference

Reference documentation is generated from Python docstrings in the `grumpy` package.

## Top-level API

::: grumpy
    options:
      show_root_heading: false
      heading_level: 3
      members_order: alphabetical
      filters:
        - "!^GrumpyArray$"
        - "!^DType$"
        - "!^Stream$"
        - "!^StreamApply$"
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
      filters:
        - "!^__"

## Streaming

::: grumpy.stream
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - Stream
        - StreamApply

## Compilation

::: grumpy.compiler
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - compile
        - CompiledTransform
