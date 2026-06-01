"""Edge cases for UnionScalarList: nulls, empty lists, broadcast, reduce."""

from __future__ import annotations

import pytest

import grumpy as gr


def test_union_sum_mean_skips_null_scalars():
    x = gr.array([1, None, [2, 3], None], dtype=gr.int64)
    assert x.sum() == 6
    assert x.mean() == 2.0


def test_union_sum_mean_skips_empty_list_elements():
    x = gr.array([1, [], [2, 3], []], dtype=gr.int64)
    assert x.sum() == 6
    assert x.mean() == 2.0


def test_union_sum_dim0_null_scalars_errors():
    x = gr.array([1, None, [2, 3], None], dtype=gr.int64)
    with pytest.raises(ValueError, match="grumpy\\.ReduceEmpty"):
        x.sum(dim=0)


def test_union_sum_dim0_empty_lists_errors():
    x = gr.array([1, [], [2, 3], []], dtype=gr.int64)
    with pytest.raises(ValueError, match="grumpy\\.ReduceEmpty"):
        x.sum(dim=0)


def test_union_broadcast_incompatible_outer_lengths():
    a = gr.array([1, [2, 3], 4], dtype=gr.int64)
    b = gr.array([10, [20], 30, 40], dtype=gr.int64)
    with pytest.raises(ValueError, match="grumpy\\.BroadcastFailed"):
        _ = (a + b).to_list()


def test_union_broadcast_incompatible_layout_kind():
    a = gr.array([1, [2, 3], 4], dtype=gr.int64)
    b = gr.array([10, 20], dtype=gr.int64)
    with pytest.raises(ValueError, match="grumpy\\.BroadcastFailed"):
        _ = (a + b).to_list()


def test_union_sum_dim0():
    x = gr.array([1, [2, 3], 4, [5]], dtype=gr.int64)
    assert x.sum(dim=0).to_list() == [1, 5, 4, 5]


def test_union_sum_dim_minus_one():
    x = gr.array([1, [2, 3], 4, [5, 6]], dtype=gr.int64)
    assert x.sum(dim=-1).to_list() == [1, 5, 4, 11]
