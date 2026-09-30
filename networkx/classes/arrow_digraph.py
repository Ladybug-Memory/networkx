"""Directed graph backed by ``pyarrow.Table`` for node and edge storage.

Nodes live in a single-column table (``node``) and edges in a
``source``/``target`` table. Attribute dicts are kept alongside for O(1)
Python-level lookup; the tables are the source of truth for membership
and give columnar memory layout plus zero-copy analytics (filter,
group-by, join) via the Arrow compute API.
"""

import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["ArrowDiGraph"]


class ArrowDiGraph:
    """Minimal directed graph with edges in a ``pyarrow.Table``."""

    def __init__(self, incoming_graph_data=None):
        self._node_attrs = {}
        self._edge_attrs = {}  # (u, v) -> dict
        self._nodes_table = pa.table(
            {"node": pa.array([], type=pa.string())}
        )
        self._edges_table = pa.table(
            {"source": pa.array([], type=pa.string()),
             "target": pa.array([], type=pa.string())}
        )
        if incoming_graph_data is not None:
            self.add_edges_from(incoming_graph_data.edges())
            for n, d in incoming_graph_data.nodes(data=True):
                self.add_node(n, **d)

    # -- nodes --
    def add_node(self, node, **attrs):
        key = str(node)
        is_new = key not in self._node_attrs
        if is_new:
            self._node_attrs[key] = {"_orig": node}
            self._nodes_table = pa.concat_tables(
                [self._nodes_table, pa.table({"node": [key]})]
            )
        self._node_attrs[key].update(attrs)

    def add_nodes_from(self, nodes):
        for n in nodes:
            if isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], dict):
                self.add_node(n[0], **n[1])
            else:
                self.add_node(n)

    def has_node(self, node):
        return str(node) in self._node_attrs

    def remove_node(self, node):
        key = str(node)
        if key not in self._node_attrs:
            raise KeyError(node)
        del self._node_attrs[key]
        kill = [(u, v) for (u, v) in self._edge_attrs if u == key or v == key]
        for e in kill:
            del self._edge_attrs[e]
        self._rebuild_nodes_table()
        self._rebuild_table()

    @property
    def nodes(self):
        return [d["_orig"] for d in self._node_attrs.values()]

    def number_of_nodes(self):
        return len(self._node_attrs)

    # -- edges --
    def add_edge(self, u, v, **attrs):
        self.add_node(u)
        self.add_node(v)
        key = (str(u), str(v))
        if key not in self._edge_attrs:
            self._edge_attrs[key] = {}
            self._edges_table = pa.concat_tables(
                [
                    self._edges_table,
                    pa.table({"source": [key[0]], "target": [key[1]]}),
                ]
            )
        self._edge_attrs[key].update(attrs)

    def add_edges_from(self, ebunch):
        for e in ebunch:
            if len(e) == 2:
                self.add_edge(*e)
            else:
                u, v, d = e[0], e[1], e[2]
                self.add_edge(u, v, **d)

    def has_edge(self, u, v):
        return (str(u), str(v)) in self._edge_attrs

    def remove_edge(self, u, v):
        key = (str(u), str(v))
        if key not in self._edge_attrs:
            raise KeyError((u, v))
        del self._edge_attrs[key]
        self._rebuild_table()

    def successors(self, node):
        key = str(node)
        return [self._node_attrs[t]["_orig"] for (s, t) in self._edge_attrs if s == key]

    def predecessors(self, node):
        key = str(node)
        return [self._node_attrs[s]["_orig"] for (s, t) in self._edge_attrs if t == key]

    @property
    def edges(self):
        return [
            (
                self._node_attrs[s]["_orig"],
                self._node_attrs[t]["_orig"],
            )
            for (s, t) in self._edge_attrs
        ]

    def number_of_edges(self):
        return len(self._edge_attrs)

    # -- arrow-native --
    def nodes_table(self):
        """Return the underlying ``pyarrow.Table`` of nodes."""
        return self._nodes_table

    def edges_table(self):
        """Return the underlying ``pyarrow.Table`` of edges."""
        return self._edges_table

    @classmethod
    def from_arrow(cls, table):
        g = cls()
        for s, t in zip(
            table.column("source").to_pylist(), table.column("target").to_pylist()
        ):
            g.add_edge(s, t)
        return g

    def out_degree_table(self):
        """Arrow table of (node, out_degree) via ``pyarrow.compute``."""
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
        for key, attrs in self._node_attrs.items():
            g.add_node(attrs["_orig"], **{k: v for k, v in attrs.items() if k != "_orig"})
        for (s, t), attrs in self._edge_attrs.items():
            g.add_edge(self._node_attrs[s]["_orig"], self._node_attrs[t]["_orig"], **attrs)
        return g

    def _rebuild_nodes_table(self):
        keys = list(self._node_attrs)
        self._nodes_table = pa.table({"node": pa.array(keys, type=pa.string())})

    def _rebuild_table(self):
        if not self._edge_attrs:
            self._edges_table = pa.table(
                {"source": pa.array([], type=pa.string()),
                 "target": pa.array([], type=pa.string())}
            )
        else:
            keys = list(self._edge_attrs)
            self._edges_table = pa.table(
                {
                    "source": [k[0] for k in keys],
                    "target": [k[1] for k in keys],
                }
            )

    def __len__(self):
        return self.number_of_nodes()

    def __contains__(self, node):
        return self.has_node(node)
