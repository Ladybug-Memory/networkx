"""``arrow`` NetworkX backend: columnar graph storage on ``pyarrow.Table``.

This module reframes ``ArrowDiGraph`` (see :mod:`arrow_digraph`) as a
*backend graph object* instead of a ``DiGraph`` replacement. NetworkX
dispatches to it via the standard plugin protocol:

- ``convert_from_nx`` / ``convert_to_nx`` (required) translate between the
  dict-backed reference implementation and Arrow tables. Extra attributes
  are always preserved (a superset of what the caller requests), which is
  harmless for algorithm correctness.
- ``can_run`` / ``should_run`` gate partial algorithm support: the backend
  only volunteers for algorithms with native Arrow implementations.
  Anything else falls back to NetworkX when
  ``NETWORKX_FALLBACK_TO_NX=True``.
- Native implementations (``in_degree_centrality``,
  ``out_degree_centrality``, ``topological_sort``) run directly on Arrow
  tables via ``pyarrow.compute``.

Usage::

    nx.out_degree_centrality(G, backend="arrow")  # explicit, converts + caches
    NETWORKX_BACKEND_PRIORITY=arrow python script.py  # automatic

Registered under the ``networkx.backends`` entry-point name ``arrow``.
"""

import pyarrow.compute as pc

from .arrow_digraph import ArrowDiGraph

__all__ = ["ArrowDiGraph", "ArrowBackendInterface", "backend_interface"]


class ArrowBackendInterface:
    """BackendInterface object for the ``arrow`` backend."""

    @staticmethod
    def convert_from_nx(
        graph,
        *,
        edge_attrs=None,
        node_attrs=None,
        preserve_edge_attrs=None,
        preserve_node_attrs=None,
        preserve_graph_attrs=None,
        preserve_all_attrs=False,
        name=None,
        graph_name=None,
    ):
        if isinstance(graph, ArrowDiGraph):
            return graph
        return ArrowDiGraph.from_networkx(graph)

    @staticmethod
    def convert_to_nx(result, *, name=None):
        if isinstance(result, ArrowDiGraph):
            return result.to_networkx()
        if isinstance(result, dict):
            return {
                (s.decode() if isinstance(s, bytes) else s): v
                for s, v in result.items()
            }
        if isinstance(result, (list, tuple)) and not isinstance(result, str):
            out = []
            for item in result:
                out.append(
                    item.to_networkx() if isinstance(item, ArrowDiGraph) else item
                )
            return type(result)(out) if isinstance(result, tuple) else out
        return result

    @staticmethod
    def can_run(name, args, kwargs):
        return True

    @staticmethod
    def should_run(name, args, kwargs):
        # Only volunteer for algorithms with native Arrow implementations
        # (plus graph construction); everything else stays on the NetworkX
        # path unless the user opts into backend priority + fallback.
        return name in {
            "graph__new__",
            "digraph__new__",
            "in_degree_centrality",
            "out_degree_centrality",
            "topological_sort",
        }

    # -- graph construction: nx.Graph(backend="arrow") / nx.DiGraph(...) --
    # NOTE: ArrowDiGraph is directed; nx.Graph(backend="arrow") returns it
    # as-is (directed semantics). There is no undirected Arrow class yet.

    @staticmethod
    def graph__new__(cls, incoming_graph_data=None, **attr):
        attr.pop("backend", None)  # consumed by dispatch machinery
        return ArrowDiGraph(incoming_graph_data, **attr)

    @staticmethod
    def digraph__new__(cls, incoming_graph_data=None, **attr):
        attr.pop("backend", None)  # consumed by dispatch machinery
        return ArrowDiGraph(incoming_graph_data, **attr)

    # -- native Arrow implementations (signatures mirror the nx originals) --

    @staticmethod
    def in_degree_centrality(G):
        return ArrowBackendInterface._centrality(G, "target")

    @staticmethod
    def out_degree_centrality(G):
        return ArrowBackendInterface._centrality(G, "source")

    @staticmethod
    def _centrality(G, col):
        G._sync()
        nodes = G.nodes
        n = len(nodes)
        if n <= 1:
            return {v: 0 for v in nodes}
        t = G.edges_table()
        scale = 1 / (n - 1)
        centrality = dict.fromkeys(nodes, 0.0)
        if t.num_rows == 0:
            return centrality
        grouped = t.group_by(col).aggregate([(col, "count")])
        keys = grouped.column(col).to_pylist()
        vals = grouped.column(f"{col}_count").to_pylist()
        # map string keys back to original node objects
        rev = {str(v): v for v in nodes}
        for k, c in zip(keys, vals):
            if k in rev:
                centrality[rev[k]] = c * scale
        return centrality

    @staticmethod
    def topological_sort(G):
        G._sync()
        nodes = G.nodes
        succ = {v: [] for v in nodes}
        indeg = dict.fromkeys(nodes, 0)
        rev = {str(v): v for v in nodes}
        for row in G.edges_table().to_pylist():
            u, v = rev[row["source"]], rev[row["target"]]
            succ[u].append(v)
            indeg[v] += 1
        zero = sorted(
            [v for v in nodes if indeg[v] == 0],
            key=lambda v: str(v),
        )
        order = []
        while zero:
            u = zero.pop(0)
            order.append(u)
            for w in succ[u]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    zero.append(w)
        if len(order) != len(nodes):
            raise RuntimeError("Graph contains a cycle; topological sort not possible")
        return iter(order)


backend_interface = ArrowBackendInterface()
