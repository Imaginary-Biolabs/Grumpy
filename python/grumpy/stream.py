"""Streaming iterators and parallel batch transforms for saved Grumpy datasets.

This module provides :class:`Stream` and :class:`StreamApply` for axis-0 batching
over Zarr-backed stores written by :func:`grumpy.save`.

Known limitations
-----------------
- Batching is axis-0 only; ``batch_on``, shuffle, DDP sharding, and random access
  indexing are not implemented yet.
- :meth:`Stream.__iter__` loads one batch at a time via ``load_slice`` (not the full
  dataset in memory), but ``load_slice`` still reads the on-disk layout tree for
  each batch. True chunked I/O without traversing parent layouts is future work.
- ``UnionScalarList`` layouts are not supported for streaming slice loads.
- Compiled Rust scheduling supports a restricted opcode set (see ``compiler.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union

from concurrent.futures import ThreadPoolExecutor
import warnings

T = TypeVar("T")


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Stream:
    """
    Iterator over batches of a saved :class:`~grumpy.GrumpyArray` or dataframe.

    Parameters
    ----------
    path:
        Path passed to :func:`grumpy.save` (Zarr directory store).
    batch_size:
        Maximum number of axis-0 elements per yielded batch.
    drop_last:
        If ``True``, drop the final partial batch when ``len(data) % batch_size != 0``.

    Notes
    -----
    ``__len__`` uses on-disk metadata (``stored_len``) without loading leaf buffers.
    Each batch is loaded with ``load_slice`` so repeated iteration does not keep the
    full dataset in memory; per-batch I/O still depends on the stored layout.

    Examples
    --------
    >>> import grumpy as gr
    >>> gr.save(gr.array(list(range(100))), 'data.gr')
    >>> st = gr.stream('data.gr', batch_size=32)
    >>> len(st)
    4
    >>> next(iter(st)).to_list()[:3]
    [0, 1, 2]
    """

    path: str
    batch_size: int
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def __len__(self) -> int:
        """
        Return the number of batches without loading leaf data.

        Returns
        -------
        int
            Batch count derived from on-disk axis-0 metadata.

        Examples
        --------
        >>> import grumpy as gr
        >>> gr.save(gr.array(list(range(10))), 'tmp.gr')
        >>> len(gr.stream('tmp.gr', batch_size=4))
        3
        """
        from ._core import stored_len

        n = stored_len(self.path)
        if self.drop_last:
            return n // self.batch_size
        return _ceil_div(n, self.batch_size)

    def __iter__(self) -> Iterator:
        """
        Yield consecutive axis-0 batches loaded from disk.

        Yields
        ------
        GrumpyArray or GrumpyDataFrame
            Batch covering a slice of axis 0.

        Examples
        --------
        >>> import grumpy as gr
        >>> for batch in gr.stream('data.gr', batch_size=32):
        ...     train(batch)
        """
        from ._core import load_slice, stored_len

        n = stored_len(self.path)
        bs = self.batch_size
        end = n - (n % bs) if (self.drop_last and n % bs != 0) else n
        for i in range(0, end, bs):
            yield load_slice(self.path, i, min(i + bs, end))

    def apply(
        self,
        fns: Union[Callable[[T], T], Sequence[Callable[[T], T]]],
        cpu: int = 1,
        prefetch: Optional[int] = None,
        compile: Union[bool, str] = "auto",
        scheduler: str = "auto",
    ) -> "StreamApply[T]":
        """
        Apply one or more batch transforms, optionally compiled and parallelized.

        Parameters
        ----------
        fns:
            Callable or sequence of callables ``fn(batch) -> batch``.
        cpu:
            Worker count for parallel apply (``1`` = serial).
        prefetch:
            Max in-flight batches for threaded scheduling (default ``2 * cpu``).
        compile:
            ``True``/``'force'``, ``False``/``'never'``, or ``'auto'`` (compile when possible).
        scheduler:
            ``'auto'``, ``'python'`` (thread pool), or ``'rust'`` (Rayon for fully compiled ops).

        Returns
        -------
        StreamApply
            Lazy iterable of transformed batches.

        Examples
        --------
        >>> import grumpy as gr
        >>> st = gr.stream('data.gr', batch_size=32)
        >>> out = st.apply(lambda b: b * 2, cpu=4, compile='auto')
        """
        if cpu < 1:
            raise ValueError("cpu must be >= 1")
        if callable(fns):
            fns = [fns]
        else:
            fns = list(fns)
        if len(fns) == 0:
            raise ValueError("apply requires at least one transform.")
        return StreamApply(self, fns, cpu=cpu, prefetch=prefetch, compile=compile, scheduler=scheduler)


@dataclass(frozen=True)
class StreamApply(Iterable[T]):
    """
    Lazy iterable of transformed batches produced from a :class:`Stream`.

    Parameters
    ----------
    base : Stream
        Source stream over a saved dataset.
    fns : list[callable]
        Batch transforms applied in order.
    cpu : int, default 1
        Worker count for parallel apply.
    prefetch : int, optional
        Max in-flight batches (defaults to ``2 * cpu``).
    compile : bool or str, default ``'auto'``
        Compilation mode passed to the compiler.
    scheduler : str, default ``'auto'``
        ``'auto'``, ``'python'``, or ``'rust'`` scheduling backend.

    Examples
    --------
    >>> import grumpy as gr
    >>> st = gr.stream('data.gr', batch_size=32)
    >>> for batch in st.apply(lambda b: b * 2):
    ...     process(batch)
    """

    base: Stream
    fns: list[Callable[[T], T]]
    cpu: int = 1
    prefetch: Optional[int] = None
    compile: Union[bool, str] = "auto"
    scheduler: str = "auto"

    def __iter__(self) -> Iterator[T]:
        """
        Yield transformed batches from the underlying stream.

        Yields
        ------
        GrumpyArray or GrumpyDataFrame
            Transformed batch.

        Examples
        --------
        >>> import grumpy as gr
        >>> st = gr.stream('data.gr', batch_size=32)
        >>> for batch in st.apply(lambda b: b * 2):
        ...     process(batch)
        """
        compile_mode = self.compile
        if compile_mode is True:
            compile_mode = "force"
        if compile_mode is False:
            compile_mode = "never"
        if compile_mode not in ("auto", "never", "force"):
            raise ValueError("compile must be one of: True/False/'auto'/'never'/'force'")

        scheduler = self.scheduler
        if scheduler not in ("auto", "python", "rust"):
            raise ValueError("scheduler must be one of: 'auto'/'python'/'rust'")

        # Build the runner (possibly compiled).
        run_all = None
        pipeline_info = None
        if compile_mode != "never":
            from .compiler import compile_pipeline_info

            pipeline_info = compile_pipeline_info(self.fns)
            if compile_mode == "auto" and (not pipeline_info.fully_compiled):
                pipeline_info = None
            run_all = pipeline_info.run_all if pipeline_info is not None else None
        if run_all is None:
            def run_all(x: T) -> T:
                for fn in self.fns:
                    x = fn(x)
                return x

        # Rust scheduling path: only supported for a fully fused compiled pipeline.
        if self.cpu > 1 and pipeline_info is not None and pipeline_info.fully_compiled and pipeline_info.fused_ops is not None:
            # Restrict to op types supported by Rust scheduling (array-only MVP).
            supported = True
            for d in pipeline_info.fused_ops:
                op = d.get("op")
                if op not in (
                    "add_scalar",
                    "sub_scalar",
                    "mul_scalar",
                    "div_scalar",
                    "mod_scalar",
                    "neighbors_knn_self",
                    "reduce",
                    "df_get",
                    "reduce_tmp",
                    "df_set",
                ):
                    supported = False
                    break
            if scheduler == "rust" and not supported:
                warnings.warn(
                    "Stream.apply(scheduler='rust'): pipeline contains ops that are not supported by Rust scheduling yet; falling back to Python scheduling.",
                    category=UserWarning,
                    stacklevel=2,
                )
            if (scheduler == "auto" and supported) or (scheduler == "rust" and supported):
                from . import _core
                pre = self.prefetch if self.prefetch is not None else (2 * self.cpu)
                if pre < 1:
                    pre = 1
                # This returns an iterator over transformed batches.
                yield from _core.compiled_stream_apply(
                    self.base.path,
                    self.base.batch_size,
                    self.base.drop_last,
                    self.cpu,
                    pre,
                    pipeline_info.fused_ops,
                )
                return

        if scheduler == "rust" and self.cpu > 1:
            warnings.warn(
                "Stream.apply(scheduler='rust'): could not use Rust scheduling (pipeline not fully compiled or unsupported ops); using Python scheduling.",
                category=UserWarning,
                stacklevel=2,
            )

        if self.cpu == 1:
            for b in self.base:
                yield run_all(b)
            return

        max_in_flight = self.prefetch if self.prefetch is not None else (2 * self.cpu)
        if max_in_flight < 1:
            max_in_flight = 1

        it = iter(self.base)
        with ThreadPoolExecutor(max_workers=self.cpu) as ex:
            futures = []

            # prime pipeline
            for _ in range(max_in_flight):
                try:
                    b = next(it)
                except StopIteration:
                    break
                futures.append(ex.submit(run_all, b))

            # keep order: pop from front, refill
            while futures:
                head = futures.pop(0)
                yield head.result()
                try:
                    b = next(it)
                except StopIteration:
                    continue
                futures.append(ex.submit(run_all, b))
