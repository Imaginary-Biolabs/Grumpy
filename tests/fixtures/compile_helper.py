"""Helper module for compiler source-retrieval tests."""


class _CompileHelperMarker:
    """Ensures the linecache parser sees a non-function node first."""


def batch_transform(batch):
    batch = batch + 1
    return batch
