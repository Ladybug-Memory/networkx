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
| bulk build peak (tracemalloc) | 3.6 MB | 5.7 MB |
| bulk build time (incremental API) | 0.10 s | 0.02 s (~4x slower) |
| bulk load `ArrowDiGraph(nx_graph)` | 0.01 s | — (**11.5x faster than incremental**) |
| 2000 single-edge writes | 0.011 s | 0.002 s (~6x slower) |

## n = 20000 (39999 edges)

| metric | Arrow (columnar) | DiGraph (dicts) |
|---|---|---|
| resident tables / struct | 1.49 MB | 9.55 MB (**~6.4x smaller**) |
| bulk build peak (tracemalloc) | 14.8 MB | 23.0 MB |
| bulk build time (incremental API) | 0.81 s | 0.10 s (~8x slower) |
| bulk load `ArrowDiGraph(nx_graph)` | 0.05 s | — (**17x faster than incremental, 2x faster than dict build**) |
| 2000 single-edge writes | 0.038 s | 0.001 s (~38x slower) |

## n = 200000 (399996 edges)

| metric | Arrow (columnar) | DiGraph (dicts) |
|---|---|---|
| resident tables / struct | 15.94 MB | 109.16 MB (**~6.8x smaller**) |
| bulk build peak (tracemalloc) | 163.1 MB | 243.6 MB |
| bulk build time (incremental API) | 52.97 s | 1.12 s (~47x slower) |
| bulk load `ArrowDiGraph(nx_graph)` | 0.83 s | — (**64x faster than incremental, faster than dict build**) |
| 2000 single-edge writes | 0.335 s | 0.002 s (~168x slower) |

## Takeaway

- Reads/memory scale with the columnar layout (~6.5–6.8x smaller
  footprint, lower build peak) — this is what the `arrow` backend's
  native algorithms (`in/out_degree_centrality`, `topological_sort`)
  exploit. Reserved `node_type`/`edge_type` columns are materialized
  only when actually used, so untyped graphs pay nothing for the feature.
- Bulk loads (`ArrowDiGraph(nx_graph)`, `add_nodes_from`/`add_edges_from`,
  `from_arrow`) merge in list-level passes and build each table once:
  11–64x faster than the incremental API, and at 200k faster than the
  dict build itself (0.82s vs 1.12s). One caveat: edge order follows the
  input iteration order, so an incrementally built graph and a
  bulk-loaded `DiGraph` hold the same edge *set* in different row order.
- Incremental writes stay dict-backed territory (bulk mutation is 4–175x
  slower). Write-heavy
  path: `gd = g.as_writeable()` → mutate → `ArrowDiGraph.from_networkx(gd)`,
  or set `NETWORKX_FALLBACK_TO_NX=True` and let dispatch convert back.
