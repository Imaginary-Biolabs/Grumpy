import grumpy as gr


def test_setitem_axis0_replaces_list_with_scalar():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    x[0] = 100
    assert x.to_list() == [100, [4, 5]]


def test_setitem_per_row_single_int():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    x[[0, 1]] = [10, 20]
    assert x.to_list() == [[10, 2, 3], [4, 20]]


def test_setitem_per_row_single_int_broadcast_scalar():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    x[[0, 1]] = 10
    assert x.to_list() == [[10, 2, 3], [4, 10]]


def test_setitem_coordinate_fancy():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    x[[0, 1], [0, 0]] = [10, 20]
    assert x.to_list() == [[10, 2, 3], [20, 5]]


def test_setitem_coordinate_fancy_broadcast_scalar():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    x[[0, 1], 0] = 9
    assert x.to_list() == [[9, 2, 3], [9, 5]]


def test_setitem_nested_index_list_of_lists():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    x[[[0], [1]]] = [10, 20]
    assert x.to_list() == [[10, 2, 3], [4, 20]]


