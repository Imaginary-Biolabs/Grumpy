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
    Minimal streaming iterator over a saved GrumpyArray / GrumpyDataFrame.

    Notes:
    - This currently slices along axis-0 only.
    - This is a correctness-first implementation; it loads the dataset once per iterator.
    """

    path: str
    batch_size: int
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def __len__(self) -> int:
        from ._core import load as _load

        obj = _load(self.path)
        n = len(obj)
        if self.drop_last:
            return n // self.batch_size
        return _ceil_div(n, self.batch_size)

    def __iter__(self) -> Iterator:
        from ._core import load as _load

        obj = _load(self.path)
        n = len(obj)
        bs = self.batch_size
        end = n - (n % bs) if (self.drop_last and n % bs != 0) else n
        for i in range(0, end, bs):
            yield obj[i : i + bs]

    def apply(
        self,
        fns: Union[Callable[[T], T], Sequence[Callable[[T], T]]],
        cpu: int = 1,
        prefetch: Optional[int] = None,
        compile: Union[bool, str] = "auto",
        scheduler: str = "auto",
    ) -> "StreamApply[T]":
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
    base: Stream
    fns: list[Callable[[T], T]]
    cpu: int = 1
    prefetch: Optional[int] = None
    compile: Union[bool, str] = "auto"
    scheduler: str = "auto"

    def __iter__(self) -> Iterator[T]:
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
            if compile_mode == "auto" and (not pipeline_info.run_all):
                pipeline_info = None
            run_all = pipeline_info.run_all
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


