import pytest
import grumpy as gr


def test_elementwise_add_deep_ragged_matches_python_reference():
    a = [[[1, 2], [3]], [[4], [5, 6, 7]]]
    b = [[[10, 20], [30]], [[40], [50, 60, 70]]]
    x = gr.array(a, dtype=gr.int32)
    y = gr.array(b, dtype=gr.int32)
    out = (x + y).to_list()
    assert out == [[[11, 22], [33]], [[44], [55, 66, 77]]]


def test_elementwise_scalar_broadcast_deep():
    a = [[[1, 2], [3]], [[4], [5, 6, 7]]]
    x = gr.array(a, dtype=gr.int32)
    out = (x * 2).to_list()
    assert out == [[[2, 4], [6]], [[8], [10, 12, 14]]]


def test_elementwise_axis0_broadcast_deep():
    a = [[[1, 2], [3]], [[4], [5, 6, 7]]]
    b = [[[10, 20], [30]]]  # len==1 on axis0 should broadcast
    x = gr.array(a, dtype=gr.int32)
    y = gr.array(b, dtype=gr.int32)
    out = (x + y).to_list()
    assert out == [[[11, 22], [33]], [[14, 24], [35, 36, 37]]]


def test_elementwise_union_same_structure_supported():
    # This construction triggers UnionScalarList internally (variable depth) and has nulls.
    a = [[1, 2, 3], [[None, 5], [6]]]
    x = gr.array(a, dtype=gr.int64)
    out = (x + x).to_list()
    assert out == [[2, 4, 6], [[None, 10], [12]]]


def test_elementwise_modulo_by_zero_errors_not_panics():
    x = gr.array([1, 2, 3], dtype=gr.int32)
    y = gr.array([0, 1, 0], dtype=gr.int32)
    with pytest.raises(ValueError, match="Modulo by zero"):
        _ = (x % y).to_list()


def test_elementwise_float16_supported():
    x = gr.array([[1.0, 2.0], [3.0]], dtype=gr.float16)
    y = gr.array([[0.5, 1.5], [2.0]], dtype=gr.float16)
    out = (x + y).to_list()
    # float16 is approximate; compare as python floats within tolerance.
    assert out[0][0] == pytest.approx(1.5, abs=1e-3)
    assert out[0][1] == pytest.approx(3.5, abs=1e-3)
    assert out[1][0] == pytest.approx(5.0, abs=1e-3)


def test_optional_awkward_parity_for_add_if_available():
    ak = pytest.importorskip("awkward")
    a = [[[1, 2], [3]], [[4], [5, 6, 7]]]
    b = [[[10, 20], [30]], [[40], [50, 60, 70]]]
    x = gr.array(a, dtype=gr.int32)
    y = gr.array(b, dtype=gr.int32)
    gr_out = (x + y).to_list()
    ak_out = ak.to_list(ak.Array(a) + ak.Array(b))
    assert gr_out == ak_out


def test_reductions_deep_dim2_sum():
    # 3D ragged: reduce inner-most list axis
    a = [[[1, 2], [3]], [[4], [5, 6, 7]]]
    x = gr.array(a, dtype=gr.int32)
    assert x.sum(dim=2).to_list() == [[3, 3], [4, 18]]


def test_reductions_deep_dim0_sum_strict_placeholder_semantics():
    # dim=0 requires all rows to have a value at that position and it must be non-null,
    # otherwise the output at that position is None.
    a = [[[1, 2], [3]], [[4], [5, 6, 7]]]
    x = gr.array(a, dtype=gr.int32)
    assert x.sum(dim=0).to_list() == [[5, None], [8, None, None]]


