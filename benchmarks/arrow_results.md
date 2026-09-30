# Arrow backend benchmark results

Command: `PYTHONPATH=/Users/arun/src/networkx python benchmarks/bench_arrow_memory.py --n {5000,20000,200000}`
(date: 2026-09-30, Python 3.14.3, pyarrow 24.0.0, branch `pyarrow`)

Workload: N nodes with `color` (4 distinct strings) + `size` (int) attrs,
2N random edges with `weight` (float) attr; then 2000 incremental
single-edge writes. "DiGraph dicts" is a `sys.getsizeof` struct estimate
(`_node`/`_succ`/`_pred` + per-node/per-succ values); "Arrow tables" is
`nodes_table().nbytes + edges_table().nbytes`.

## n = 5000 (9999 edges)

| metric | Arrow (columnar) | DiGraph (dicts) |
|---|---|---|
| resident tables / struct | 0.36 MB | 2.39 MB (**~6.7x smaller**) |
| bulk build peak (tracemalloc) | 3.5 MB | 5.7 MB |
| bulk build time | 0.09 s | 0.02 s (~4x slower) |
| 2000 single-edge writes | 0.011 s | 0.001 s (~11x slower) |

## n = 20000 (39999 edges)

| metric | Arrow (columnar) | DiGraph (dicts) |
|---|---|---|
| resident tables / struct | 1.49 MB | 9.55 MB (**~6.4x smaller**) |
| bulk build peak (tracemalloc) | 14.3 MB | 23.0 MB |
| bulk build time | 0.78 s | 0.10 s (~8x slower) |
| 2000 single-edge writes | 0.038 s | 0.001 s (~38x slower) |

## n = 200000 (399996 edges)

| metric | Arrow (columnar) | DiGraph (dicts) |
|---|---|---|
| resident tables / struct | 15.94 MB | 109.16 MB (**~6.8x smaller**) |
| bulk build peak (tracemalloc) | 158.2 MB | 243.6 MB |
| bulk build time | 56.70 s | 1.15 s (~49x slower) |
| 2000 single-edge writes | 0.350 s | 0.002 s (~175x slower) |

## Takeaway

- Reads/memory scale with the columnar layout (~6.5x smaller footprint,
  lower build peak) — this is what the `arrow` backend's native
  algorithms (`in/out_degree_centrality`, `topological_sort`) exploit.
- Incremental writes stay dict-backed territory (lazy `_sync()` keeps
  small edits acceptable, but bulk mutation is 4–175x slower, and the
  Arrow build itself scales superlinearly (~49x slower at n=200k vs ~8x
  at n=20k — needs profiling before larger runs). Write-heavy
  path: `gd = g.as_writeable()` → mutate → `ArrowDiGraph.from_networkx(gd)`,
  or set `NETWORKX_FALLBACK_TO_NX=True` and let dispatch convert back.
