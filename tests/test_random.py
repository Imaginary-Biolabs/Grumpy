import pytest

import grumpy as gr


def test_rng_choice_dim1_ragged():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    rng = gr.rng(42)
    out = rng.choice(x, size=2, replace=False, dim=1)
    assert out.to_list() == [[1, 2], [5, 4]]
    for src, dst in zip(x.to_list(), out.to_list()):
        assert len(dst) == 2
        assert set(dst).issubset(set(src))
        assert len(set(dst)) == 2


def test_rng_choice_dim1_raises_when_too_large():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    rng = gr.rng(0)
    with pytest.raises(ValueError, match="replace=False"):
        rng.choice(x, size=3, replace=False, dim=1)


def test_rng_choice_dim0():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    rng = gr.rng(7)
    out = rng.choice(x, size=1, replace=False, dim=0)
    assert out.to_list() in ([1, 2, 3], [4, 5])


def test_rng_choice_fraction_dim0():
    x = gr.array([[1, 2, 3], [4, 5], [6, 7, 8]], dtype=gr.int32)
    rng = gr.rng(1)
    out = rng.choice(x, size=0.5, replace=False, dim=0)
    assert out.to_list() in x.to_list()


def test_rng_choice_per_row_sizes():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    rng = gr.rng(99)
    out = rng.choice(x, size=[2, 1], replace=False, dim=1)
    assert [len(r) for r in out.to_list()] == [2, 1]
    assert set(out.to_list()[0]).issubset({1, 2, 3})
    assert out.to_list()[1][0] in {4, 5}


def test_array_choice_dot_notation():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    out = x.choice(size=2, replace=False, dim=1, seed=42)
    assert out.to_list() == gr.rng(42).choice(x, size=2, replace=False, dim=1).to_list()


def test_integers_like_shape_and_range():
    template = gr.array([[1, 2], [3, 4, 5]], dtype=gr.int32)
    rng = gr.rng(0)
    out = rng.integers_like(template, 0, 10, dtype=gr.int32)
    assert out.to_list() == [[1, 3], [7, 0, 8]]
    flat = [v for row in out.to_list() for v in row]
    assert len(flat) == 5
    assert all(0 <= v < 10 for v in flat)


def test_uniform_like_and_random_like():
    template = gr.array([[0.0, 1.0], [2.0]], dtype=gr.float64)
    rng = gr.rng(3)
    u = rng.uniform_like(template, 2.0, 4.0)
    r = rng.random_like(template)
    for row in u.to_list():
        for v in row:
            assert 2.0 <= v < 4.0
    for row in r.to_list():
        for v in row:
            assert 0.0 <= v < 1.0
    u2 = template.uniform_like(2.0, 4.0, seed=3)
    assert u2.to_list() == u.to_list()


def test_normal_like():
    template = gr.array([1.0, 2.0, 3.0], dtype=gr.float64)
    rng = gr.rng(11)
    out = rng.normal_like(template, loc=0.0, scale=1.0)
    vals = out.to_list()
    assert len(vals) == 3
    assert all(isinstance(v, float) for v in vals)


def test_integers_1d():
    rng = gr.rng(5)
    out = rng.integers(0, 100, size=8, dtype=gr.int32)
    vals = out.to_list()
    assert len(vals) == 8
    assert all(0 <= v < 100 for v in vals)
    assert gr.rng(5).integers(0, 100, size=8, dtype=gr.int32).to_list() == vals


def test_permutation_dim0():
    x = gr.array([[1, 2], [3, 4], [5, 6]], dtype=gr.int32)
    rng = gr.rng(2)
    out = rng.permutation(x, dim=0)
    assert sorted(out.to_list(), key=lambda r: r[0]) == x.to_list()


def test_permutation_dim1_ragged():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    rng = gr.rng(4)
    out = rng.permutation(x, dim=1)
    for src_row, dst_row in zip(x.to_list(), out.to_list()):
        assert sorted(dst_row) == sorted(src_row)


def test_shuffle_dim0_in_place():
    x = gr.array([[1, 2], [3, 4], [5, 6]], dtype=gr.int32)
    before = x.to_list()
    x.shuffle(dim=0, seed=2)
    after = x.to_list()
    assert sorted(after, key=lambda r: r[0]) == before
    assert after != before


def test_shuffle_via_generator():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    rng = gr.rng(8)
    before = [row[:] for row in x.to_list()]
    rng.shuffle(x, dim=1)
    after = x.to_list()
    for b, a in zip(before, after):
        assert sorted(a) == sorted(b)


def test_array_permutation_dot_notation():
    x = gr.array([10, 20, 30], dtype=gr.int32)
    out = x.permutation(dim=0, seed=6)
    assert sorted(out.to_list()) == [10, 20, 30]
    assert out.to_list() == gr.rng(6).permutation(x, dim=0).to_list()
