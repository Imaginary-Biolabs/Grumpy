import grumpy as gr


def test_flatten_examples():
    y = gr.array([[1, 2, 3], [[None, 5], [6]]], dtype=gr.int32)

    assert y.flatten().to_list() == [1, 2, 3, None, 5, 6]
    assert y.flatten(dim=2).to_list() == [[1, 2, 3], [None, 5, 6]]
    assert y.flatten(but=-1).to_list() == [1, 2, 3, [None, 5], [6]]
    assert y.flatten(dim=[1, 2]).to_list() == [1, 2, 3, None, 5, 6]


def test_unflatten_dim0_sizes_list():
    x = gr.array([1, 2, 3, 4, 5, 6], dtype=gr.int32)
    y = x.unflatten(sizes=[4, 2], dim=0)
    assert y.to_list() == [[1, 2, 3, 4], [5, 6]]


def test_unflatten_dim1_nested_sizes():
    x = gr.array([[1, 2, 3], [4, 5, 6]], dtype=gr.int32)
    y = x.unflatten(sizes=[[2, 1], [1, 2]], dim=1)
    assert y.to_list() == [[[1, 2], [3]], [[4], [5, 6]]]


def test_unflatten_sizes_from_shape():
    base = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    flat = base.flatten()
    sizes = base.shape(dim=1)  # GrumpyArray[int64]
    y = flat.unflatten(sizes=sizes, dim=0)
    assert y.to_list() == [[1, 2, 3], [4, 5]]


