## Benchmarks

Run the indexing/setitem benchmark (after `maturin develop`):

```bash
python benchmarks/benchmark_indexing.py --nrows 4096 --ncols 64 --nfancy 4096
```

Notes:
- NumPy uses a typed `int32` array.
- Grumpy constructs from nested Python lists (rectangular data treated as ragged, but still a pure list-chain layout).
- For fairness, the benchmark includes both scalar-loop and fancy-index operations.


