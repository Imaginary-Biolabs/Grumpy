"""Pytest configuration for compiler/coverage instrumentation tests."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.parent.name == "coverage":
            item.add_marker(pytest.mark.coverage)
