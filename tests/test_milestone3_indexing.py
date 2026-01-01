import grumpy as gr


def test_indexing_examples():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)

    # coordinate indexing
    a = x[0]
    assert hasattr(a, "to_list")
    assert a.to_list() == [1, 2, 3]

    # array-indexing outer fancy selection (len != outer len)
    assert x[[0]].to_list() == [[1, 2, 3]]

    # array-indexing per-row selection (len == outer len)
    assert x[[0, 1]].to_list() == [[1], [5]]
    assert x[[slice(None, 2), 1]].to_list() == [[1, 2], [5]]

    # coordinate indexing via tuple (note trailing comma)
    assert x[[0, 1],].to_list() == [[1, 2, 3], [4, 5]]

    # scalar coordinate indexing
    assert x[0, 0] == 1

    # fancy coordinate indexing
    assert x[[0, 1], [0, 0]].to_list() == [1, 4]
    assert x[[0, 1, 0, 1], [0, 0, 1, 1]].to_list() == [1, 4, 2, 5]

    # broadcast scalar coordinate with fancy coords
    assert x[[0, 1], 0].to_list() == [1, 4]

    # boolean indexing (array-indexing mask on axis 0)
    assert x[[True, False]].to_list() == [[1, 2, 3]]


