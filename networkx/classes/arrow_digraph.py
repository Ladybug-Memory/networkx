"""Directed graph backed by ``pyarrow.Table`` for nodes and edges.

Nodes: ``node`` key column + one typed column per homogeneous attribute.
Edges: ``source``/``target`` key columns + one typed column per attribute.
Missing values are null. Columns are (dictionary-)encoded by Arrow, so
large homogeneous attrs use far less memory than per-object Python dicts.
"""

import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["ArrowDiGraph"]


def _typed_column(values):
    try:
        return pa.array(values)
    except Exception:
        return pa.array([None if v is None else str(v) for v in values])


class ArrowDiGraph:
    """Minimal directed graph with nodes/edges + attrs in Arrow tables."""

    __networkx_backend__ = "arrow"

    def __init__(self, incoming_graph_data=None, **attr):
        attr.pop("backend", None)  # consumed by dispatch machinery
        self.graph = dict(attr)  # graph-level attributes, like nx.Graph.graph
        self._node_order = []  # string keys, insertion order
        self._node_pos = {}  # key -> index (O(1) lookup)
        self._node_orig = {}  # key -> original object
        self._node_cols = {}  # attr name -> list aligned with _node_order
        self._edge_order = []  # (skey, tkey) tuples
        self._edge_pos = {}  # edge key -> index
        self._edge_cols = {}
        self._nodes_table = pa.table({"node": pa.array([], type=pa.string())})
        self._edges_table = pa.table(
            {"source": pa.array([], type=pa.string()),
             "target": pa.array([], type=pa.string())}
        )
        self._nodes_dirty = False
        self._edges_dirty = False
        if isinstance(incoming_graph_data, ArrowDiGraph):
            other = incoming_graph_data
            self._node_order = list(other._node_order)
            self._node_pos = dict(other._node_pos)
            self._node_orig = dict(other._node_orig)
            self._node_cols = {k: list(v) for k, v in other._node_cols.items()}
            self._edge_order = list(other._edge_order)
            self._edge_pos = dict(other._edge_pos)
            self._edge_cols = {k: list(v) for k, v in other._edge_cols.items()}
            self.graph.update(other.graph)
            self._nodes_dirty = self._edges_dirty = True
        elif incoming_graph_data is not None:
            if not hasattr(incoming_graph_data, "nodes"):
                # edgelists, dicts, generators...: use the standard conversion
                import networkx as nx

                incoming_graph_data = nx.convert.to_networkx_graph(
                    incoming_graph_data, create_using=nx.DiGraph
                )
            for n, d in incoming_graph_data.nodes(data=True):
                self.add_node(n, **d)
            for u, v, d in incoming_graph_data.edges(data=True):
                self.add_edge(u, v, **d)
            if hasattr(incoming_graph_data, "graph"):
                self.graph.update(incoming_graph_data.graph)

    # -- nodes --
    def add_node(self, node, **attrs):
        key = str(node)
        if key not in self._node_orig:
            self._node_orig[key] = node
            self._node_pos[key] = len(self._node_order)
            self._node_order.append(key)
            for name in self._node_cols:
                self._node_cols[name].append(None)
        idx = self._node_pos[key]
        for name, val in attrs.items():
            self._node_cols.setdefault(name, [None] * len(self._node_order))
            self._node_cols[name][idx] = val
        self._nodes_dirty = True

    def add_nodes_from(self, nodes):
        for n in nodes:
            if isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], dict):
                self.add_node(n[0], **n[1])
            else:
                self.add_node(n)

    def has_node(self, node):
        return str(node) in self._node_orig

    def remove_node(self, node):
        key = str(node)
        if key not in self._node_orig:
            raise KeyError(node)
        idx = self._node_pos.pop(key)
        del self._node_order[idx]
        del self._node_orig[key]
        for name in self._node_cols:
            del self._node_cols[name][idx]
        for i in range(idx, len(self._node_order)):
            self._node_pos[self._node_order[i]] = i
        keep = [(s, t) for (s, t) in self._edge_order if s != key and t != key]
        drop = set(self._edge_order) - set(keep)
        if drop:
            idxs = [i for i, e in enumerate(self._edge_order) if e in drop]
            for i in sorted(idxs, reverse=True):
                del self._edge_order[i]
                for name in self._edge_cols:
                    del self._edge_cols[name][i]
            self._edge_pos = {e: i for i, e in enumerate(self._edge_order)}
        self._nodes_dirty = True
        self._edges_dirty = True

    def node_attrs(self, node):
        key = str(node)
        idx = self._node_pos[key]
        return {n: c[idx] for n, c in self._node_cols.items() if c[idx] is not None}

    @property
    def nodes(self):
        return [self._node_orig[k] for k in self._node_order]

    def number_of_nodes(self):
        return len(self._node_order)

    # -- edges --
    def add_edge(self, u, v, **attrs):
        self.add_node(u)
        self.add_node(v)
        key = (str(u), str(v))
        if key not in self._edge_pos:
            self._edge_pos[key] = len(self._edge_order)
            self._edge_order.append(key)
            for name in self._edge_cols:
                self._edge_cols[name].append(None)
        idx = self._edge_pos[key]
        for name, val in attrs.items():
            self._edge_cols.setdefault(name, [None] * len(self._edge_order))
            self._edge_cols[name][idx] = val
        self._edges_dirty = True

    def add_edges_from(self, ebunch):
        for e in ebunch:
            if len(e) == 2:
                self.add_edge(*e)
            else:
                self.add_edge(e[0], e[1], **e[2])

    def has_edge(self, u, v):
        return (str(u), str(v)) in self._edge_pos

    def remove_edge(self, u, v):
        key = (str(u), str(v))
        if key not in self._edge_pos:
            raise KeyError((u, v))
        idx = self._edge_pos.pop(key)
        del self._edge_order[idx]
        for name in self._edge_cols:
            del self._edge_cols[name][idx]
        for i in range(idx, len(self._edge_order)):
            self._edge_pos[self._edge_order[i]] = i
        self._edges_dirty = True

    def edge_attrs(self, u, v):
        idx = self._edge_pos[(str(u), str(v))]
        return {n: c[idx] for n, c in self._edge_cols.items() if c[idx] is not None}

    def successors(self, node):
        key = str(node)
        return [self._node_orig[t] for (s, t) in self._edge_order if s == key]

    def predecessors(self, node):
        key = str(node)
        return [self._node_orig[s] for (s, t) in self._edge_order if t == key]

    @property
    def edges(self):
        return [(self._node_orig[s], self._node_orig[t]) for (s, t) in self._edge_order]

    def number_of_edges(self):
        return len(self._edge_order)

    # -- arrow-native --
    def _sync(self):
        if self._nodes_dirty:
            self._rebuild_nodes_table()
            self._nodes_dirty = False
        if self._edges_dirty:
            self._rebuild_edges_table()
            self._edges_dirty = False

    def nodes_table(self):
        self._sync()
        return self._nodes_table

    def edges_table(self):
        self._sync()
        return self._edges_table

    def as_writeable(self):
        """Fall back to a dict-backed ``DiGraph`` for write-heavy work.

        Use when doing many incremental adds/removes (O(1) per edit vs
        O(n) table rebuilds here). Convert back with ``from_networkx``.
        """
        return self.to_networkx()

    @classmethod
    def from_networkx(cls, g):
        """Bulk-load from a dict-backed graph (the write-heavy fallback)."""
        return cls(g)

    @classmethod
    def from_arrow(cls, nodes=None, edges=None):
        g = cls()
        if nodes is not None:
            pylist = nodes.to_pylist()
            for row in pylist:
                g.add_node(row.pop("node"), **{k: v for k, v in row.items() if v is not None})
        if edges is not None:
            for row in edges.to_pylist():
                s, t = row.pop("source"), row.pop("target")
                g.add_edge(s, t, **{k: v for k, v in row.items() if v is not None})
        return g

    def out_degree_table(self):
        self._sync()
        if self._edges_table.num_rows == 0:
            return pa.table(
                {"node": pa.array([], type=pa.string()),
                 "out_degree": pa.array([], type=pa.int64())}
            )
        grouped = self._edges_table.group_by("source").aggregate([("source", "count")])
        return pa.table(
            {
                "node": grouped.column("source"),
                "out_degree": pc.fill_null(grouped.column("source_count"), 0),
            }
        )

    def to_networkx(self):
        import networkx as nx

        g = nx.DiGraph()
        g.graph.update(self.graph)
        for i, k in enumerate(self._node_order):
            g.add_node(
                self._node_orig[k],
                **{n: c[i] for n, c in self._node_cols.items() if c[i] is not None},
            )
        for i, (s, t) in enumerate(self._edge_order):
            g.add_edge(
                self._node_orig[s], self._node_orig[t],
                **{n: c[i] for n, c in self._edge_cols.items() if c[i] is not None},
            )
        return g

    def _rebuild_nodes_table(self):
        cols = {"node": pa.array(self._node_order, type=pa.string())}
        for name, vals in self._node_cols.items():
            cols[name] = _typed_column(list(vals))
        self._nodes_table = pa.table(cols)

    def _rebuild_edges_table(self):
        cols = {
            "source": pa.array([s for s, _ in self._edge_order], type=pa.string()),
            "target": pa.array([t for _, t in self._edge_order], type=pa.string()),
        }
        for name, vals in self._edge_cols.items():
            cols[name] = _typed_column(list(vals))
        self._edges_table = pa.table(cols)

    def is_directed(self):
        return True

    def is_multigraph(self):
        return False

    def __len__(self):
        return self.number_of_nodes()

    def __contains__(self, node):
        return self.has_node(node)
