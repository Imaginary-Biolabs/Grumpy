"""Streaming iterators and parallel batch transforms for saved Grumpy datasets.

This module provides :class:`Stream` and :class:`StreamApply` for batching
over Zarr-backed stores written by :func:`grumpy.save`.

Features
--------
- Axis-0 batching with optional ``batch_on`` schema-level packing (list-chain and union layouts)
- Reproducible batch-order shuffle and within-batch shuffle on a schema level
- DDP sharding via ``world_size`` / ``rank``
- I/O prefetch via ``workers`` (distinct from ``StreamApply`` transform parallelism)
- Partial batch reads (leaf ranges only) via the Rust ``StreamBatchesIter``
- Compact union partial I/O: slice tags/index and referenced scalar/list pools only
- Subset iteration via ``st[index]`` (int, slice, or sequence of batch indices)

Notes
-----
- ``Indexed`` layouts are not yet supported for streaming slice loads.
- Compiled Rust scheduling supports a restricted opcode set (see ``compiler.py``); scalar
  elementwise opcodes work on ``UnionScalarList`` batches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union

import contextvars
from concurrent.futures import ThreadPoolExecutor
import warnings

from .errors import arg_invalid, arg_one_of, index_out_of_range, raise_grumpy_error

T = TypeVar("T")

# Default GPU mode for gr.neighbors when called inside Stream.apply (see Stream.gpu).
_STREAM_GPU: contextvars.ContextVar[str] = contextvars.ContextVar("grumpy_stream_gpu", default="never")


def _normalize_gpu(gpu: Union[bool, str]) -> str:
    if gpu is True:
        return "auto"
    if gpu is False:
        return "never"
    if gpu not in ("auto", "never", "force"):
        arg_one_of("gpu", gpu, ("True", "False", "'auto'", "'never'", "'force'"))
    return gpu


def current_stream_gpu() -> str:
    """Return the active stream GPU mode for nested :func:`grumpy.neighbors` calls."""
    return _STREAM_GPU.get()


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
        Maximum number of axis-0 elements (or ``batch_on`` entities) per batch.
    drop_last:
        If ``True``, drop the final partial batch.
    batch_on:
        Optional schema level name (e.g. ``'molecule'``) to pack batches by entity
        count at that nesting depth instead of axis 0.
    shuffle:
        If set (e.g. ``'molecule'``), shuffle batch order with ``seed`` and optionally
        shuffle within each batch on that schema axis after loading.
    seed:
        Random seed for ``shuffle`` (required for reproducible training).
    workers:
        Number of background I/O prefetch slots and parallel loader threads (``0`` = synchronous loads).
    in_memory:
        If ``True``, load the entire dataset into RAM once at stream open; batches are zero-copy slices.
    world_size:
        DDP world size; batches are partitioned as ``index % world_size == rank``.
    rank:
        DDP rank in ``[0, world_size)``.

    Examples
    --------
    >>> import grumpy as gr
    >>> gr.save(gr.array(list(range(100))), 'data.gr')
    >>> st = gr.stream('data.gr', batch_size=32)
    >>> len(st)
    4
    """

    path: str
    batch_size: int
    drop_last: bool = False
    batch_on: Optional[str] = None
    shuffle: Optional[Union[str, bool]] = None
    seed: Optional[int] = None
    workers: int = 0
    in_memory: bool = False
    gpu: Union[bool, str] = "auto"
    world_size: int = 1
    rank: int = 0
    batch_indices: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            arg_invalid("batch_size", f"got {self.batch_size}", fix="pass batch_size > 0 (number of axis-0 or batch_on entities per batch).")
        if self.workers < 0:
            arg_invalid("workers", f"got {self.workers}", fix="pass workers >= 0; use 0 for synchronous I/O.")
        if self.world_size <= 0:
            arg_invalid("world_size", f"got {self.world_size}", fix="pass world_size >= 1 (DDP process count).")
        if self.rank < 0 or self.rank >= self.world_size:
            raise_grumpy_error(
                "ArgumentInvalid",
                f"rank {self.rank} is invalid for world_size={self.world_size}",
                cause="rank must satisfy 0 <= rank < world_size for DDP sharding.",
                fix="set rank to your process index in [0, world_size).",
            )
        if self.shuffle is not None and self.seed is None:
            raise_grumpy_error(
                "ArgumentInvalid",
                "shuffle is set but seed is None",
                cause="shuffled batch order requires a fixed seed for reproducible training.",
                fix="pass seed=<int> when shuffle is True or a schema level name.",
            )

    def _batch_indices_arg(self) -> Optional[list[int]]:
        if self.batch_indices is None:
            return None
        return list(self.batch_indices)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> "Stream":
        """Return a stream over a subset of batches (after DDP sharding)."""
        n = len(self)
        if isinstance(index, int):
            if index < 0:
                index += n
            if index < 0 or index >= n:
                index_out_of_range(index, n, at="in this stream's batch sequence")
            indices = (index,)
        elif isinstance(index, slice):
            indices = tuple(range(*index.indices(n)))
        else:
            indices = tuple(int(i) for i in index)
        return replace(self, batch_indices=indices)

    def __len__(self) -> int:
        """Return the number of batches (after DDP sharding, before shuffle)."""
        from ._core import stream_len

        return stream_len(
            self.path,
            self.batch_size,
            self.drop_last,
            self.batch_on,
            self.world_size,
            self.rank,
            self._batch_indices_arg(),
            self.in_memory,
        )

    def __iter__(self) -> Iterator:
        """Yield consecutive batches loaded from disk."""
        from ._core import stream_batches

        shuffle_arg: Optional[str]
        if self.shuffle is True:
            shuffle_arg = "true"
        elif self.shuffle is False or self.shuffle is None:
            shuffle_arg = None
        else:
            shuffle_arg = str(self.shuffle)

        return stream_batches(
            self.path,
            self.batch_size,
            self.drop_last,
            self.batch_on,
            shuffle_arg,
            self.seed,
            self.workers,
            self.world_size,
            self.rank,
            self._batch_indices_arg(),
            self.in_memory,
        )

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
            ``True``/``'force'``, ``False``/``'never'``, or ``'auto'``.
        scheduler:
            ``'auto'``, ``'python'``, or ``'rust'`` (Rayon for fully compiled ops).

        Returns
        -------
        StreamApply
            Lazy iterable of transformed batches.
        """
        if cpu < 1:
            arg_invalid("cpu", f"got {cpu}", fix="pass cpu >= 1 for parallel batch transforms.")
        if callable(fns):
            fns = [fns]
        else:
            fns = list(fns)
        if len(fns) == 0:
            raise_grumpy_error(
                "ArgumentInvalid",
                "apply requires at least one transform",
                cause="an empty fn list would leave batches unchanged with no work to schedule.",
                fix="pass a callable or non-empty sequence of callables, e.g. st.apply(lambda b: ...).",
            )
        return StreamApply(
            self,
            fns,
            cpu=cpu,
            prefetch=prefetch,
            compile=compile,
            scheduler=scheduler,
            gpu=self.gpu,
        )


@dataclass(frozen=True)
class StreamApply(Iterable[T]):
    """Lazy iterable of transformed batches produced from a :class:`Stream`."""

    base: Stream
    fns: list[Callable[[T], T]]
    cpu: int = 1
    prefetch: Optional[int] = None
    compile: Union[bool, str] = "auto"
    scheduler: str = "auto"
    gpu: Union[bool, str] = "auto"

    def __iter__(self) -> Iterator[T]:
        gpu_mode = _normalize_gpu(self.gpu)
        token = _STREAM_GPU.set(gpu_mode)
        try:
            yield from self._iter_batches(gpu_mode)
        finally:
            _STREAM_GPU.reset(token)

    def _iter_batches(self, gpu_mode: str) -> Iterator[T]:
        compile_mode = self.compile
        if compile_mode is True:
            compile_mode = "force"
        if compile_mode is False:
            compile_mode = "never"
        if compile_mode not in ("auto", "never", "force"):
            arg_one_of("compile", compile_mode, ("True", "False", "'auto'", "'never'", "'force'"))
        scheduler = self.scheduler
        if scheduler not in ("auto", "python", "rust"):
            arg_one_of("scheduler", scheduler, ("'auto'", "'python'", "'rust'"))

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

        if (
            self.cpu > 1
            and pipeline_info is not None
            and pipeline_info.fully_compiled
            and pipeline_info.fused_ops is not None
        ):
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
                shuffle_arg: Optional[str]
                if self.base.shuffle is True:
                    shuffle_arg = "true"
                elif self.base.shuffle is False or self.base.shuffle is None:
                    shuffle_arg = None
                else:
                    shuffle_arg = str(self.base.shuffle)
                yield from _core.compiled_stream_apply(
                    self.base.path,
                    self.base.batch_size,
                    self.base.drop_last,
                    self.cpu,
                    pre,
                    pipeline_info.fused_ops,
                    self.base.batch_on,
                    shuffle_arg,
                    self.base.seed,
                    self.base.world_size,
                    self.base.rank,
                    self.base._batch_indices_arg(),
                    gpu_mode,
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

        # Parallel transform: prefetch from base stream when workers > 0.
        max_in_flight = self.prefetch if self.prefetch is not None else (2 * self.cpu)
        if max_in_flight < 1:
            max_in_flight = 1

        it = iter(self.base)
        with ThreadPoolExecutor(max_workers=self.cpu) as ex:
            futures = []
            for _ in range(max_in_flight):
                try:
                    b = next(it)
                except StopIteration:
                    break
                futures.append(ex.submit(run_all, b))

            while futures:
                head = futures.pop(0)
                yield head.result()
                try:
                    b = next(it)
                except StopIteration:
                    continue
                futures.append(ex.submit(run_all, b))
