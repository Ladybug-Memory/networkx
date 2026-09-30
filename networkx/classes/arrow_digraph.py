"""Directed graph backed by ``pyarrow.Table`` for nodes and edges.

Nodes: ``node`` key column + one typed column per homogeneous attribute.
Edges: ``source``/``target`` key columns + one typed column per attribute.
Reserved ``node_type``/``edge_type`` columns are materialized only when at
least one node/edge actually has a type, so single-type (or untyped)
graphs pay nothing for the feature. Missing values are null. Columns are (dictionary-)encoded by Arrow, so
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
        self._node_types = []  # reserved node_type column, aligned with _node_order
        self._edge_order = []  # (skey, tkey) tuples
        self._edge_pos = {}  # edge key -> index
        self._edge_cols = {}
        self._edge_types = []  # reserved edge_type column, aligned with _edge_order
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
            self._node_types = list(other._node_types)
            self._edge_order = list(other._edge_order)
            self._edge_pos = dict(other._edge_pos)
            self._edge_cols = {k: list(v) for k, v in other._edge_cols.items()}
            self._edge_types = list(other._edge_types)
            self.graph.update(other.graph)
            self._nodes_dirty = self._edges_dirty = True
        elif incoming_graph_data is not None:
            if not hasattr(incoming_graph_data, "nodes"):
                # edgelists, dicts, generators...: use the standard conversion
                import networkx as nx

                incoming_graph_data = nx.convert.to_networkx_graph(
                    incoming_graph_data, create_using=nx.DiGraph
                )
            self._add_nodes_batch(incoming_graph_data.nodes(data=True))
            self._add_edges_batch(incoming_graph_data.edges(data=True))
            if hasattr(incoming_graph_data, "graph"):
                self.graph.update(incoming_graph_data.graph)

    # -- nodes --
    def add_node(self, node, node_type=None, **attrs):
        key = str(node)
        if key not in self._node_orig:
            self._node_orig[key] = node
            self._node_pos[key] = len(self._node_order)
            self._node_order.append(key)
            self._node_types.append(None)
            for name in self._node_cols:
                self._node_cols[name].append(None)
        idx = self._node_pos[key]
        if node_type is not None:
            self._node_types[idx] = node_type
        for name, val in attrs.items():
            self._node_cols.setdefault(name, [None] * len(self._node_order))
            self._node_cols[name][idx] = val
        self._nodes_dirty = True

    def node_type(self, node):
        return self._node_types[self._node_pos[str(node)]]

    @property
    def node_types(self):
        return sorted({t for t in self._node_types if t is not None})

    def nodes_of_type(self, node_type):
        """Nodes of a given type, filtered in Arrow (no Python scan)."""
        self._sync()
        if "node_type" not in self._nodes_table.column_names:
            return []
        mask = pc.equal(self._nodes_table.column("node_type"), node_type)
        keys = pc.take(
            self._nodes_table.column("node"), pc.indices_nonzero(mask)
        ).to_pylist()
        return [self._node_orig[k] for k in keys]

    def add_nodes_from(self, nodes):
        self._add_nodes_batch(nodes)

    def _add_nodes_batch(self, nodes):
        """Bulk node ingest: a few list-level passes, one table build."""
        norm = []  # (key, orig, type, attrs)
        for n in nodes:
            if isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], dict):
                d = n[1]
                norm.append((str(n[0]), n[0], d.get("node_type"), d))
            else:
                norm.append((str(n), n, None, None))
        if not norm:
            return
        for key, orig, _, _ in norm:
            if key not in self._node_orig:
                self._node_orig[key] = orig
                self._node_pos[key] = len(self._node_order)
                self._node_order.append(key)
        new_count = len(self._node_order) - len(self._node_types)
        if new_count:
            self._node_types.extend([None] * new_count)
            for col in self._node_cols.values():
                col.extend([None] * new_count)
        for _, _, _, d in norm:
            if d:
                for name in d:
                    if name != "node_type" and name not in self._node_cols:
                        self._node_cols[name] = [None] * len(self._node_order)
        for key, _, t, d in norm:
            idx = self._node_pos[key]
            if t is not None:
                self._node_types[idx] = t
            if d:
                for k, v in d.items():
                    if k != "node_type":
                        self._node_cols[k][idx] = v
        self._nodes_dirty = True

    def has_node(self, node):
        return str(node) in self._node_orig

    def remove_node(self, node):
        key = str(node)
        if key not in self._node_orig:
            raise KeyError(node)
        idx = self._node_pos.pop(key)
        del self._node_order[idx]
        del self._node_orig[key]
        del self._node_types[idx]
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
                del self._edge_types[i]
                for name in self._edge_cols:
                    del self._edge_cols[name][i]
            self._edge_pos = {e: i for i, e in enumerate(self._edge_order)}
        self._nodes_dirty = True
        self._edges_dirty = True

    def node_attrs(self, node):
        key = str(node)
        idx = self._node_pos[key]
        d = {n: c[idx] for n, c in self._node_cols.items() if c[idx] is not None}
        if self._node_types[idx] is not None:
            d["node_type"] = self._node_types[idx]
        return d

    @property
    def nodes(self):
        return [self._node_orig[k] for k in self._node_order]

    def number_of_nodes(self):
        return len(self._node_order)

    # -- edges --
    def add_edge(self, u, v, edge_type=None, **attrs):
        self.add_node(u)
        self.add_node(v)
        key = (str(u), str(v))
        if key not in self._edge_pos:
            self._edge_pos[key] = len(self._edge_order)
            self._edge_order.append(key)
            self._edge_types.append(None)
            for name in self._edge_cols:
                self._edge_cols[name].append(None)
        idx = self._edge_pos[key]
        if edge_type is not None:
            self._edge_types[idx] = edge_type
        for name, val in attrs.items():
            self._edge_cols.setdefault(name, [None] * len(self._edge_order))
            self._edge_cols[name][idx] = val
        self._edges_dirty = True

    def edge_type(self, u, v):
        return self._edge_types[self._edge_pos[(str(u), str(v))]]

    @property
    def edge_types(self):
        return sorted({t for t in self._edge_types if t is not None})

    def edges_of_type(self, edge_type):
        """Edges of a given type, filtered in Arrow (no Python scan)."""
        self._sync()
        if "edge_type" not in self._edges_table.column_names:
            return []
        mask = pc.equal(self._edges_table.column("edge_type"), edge_type)
        rows = pc.take(self._edges_table, pc.indices_nonzero(mask)).to_pylist()
        return [(self._node_orig[r["source"]], self._node_orig[r["target"]])
                for r in rows]

    def add_edges_from(self, ebunch):
        self._add_edges_batch(ebunch)

    def _add_edges_batch(self, ebunch):
        """Bulk edge ingest: endpoints ensured in one node batch, then
        edge keys/attrs merged in list-level passes, one table build."""
        norm = []  # (u_orig, v_orig, type, attrs)
        for e in ebunch:
            if len(e) == 2:
                norm.append((e[0], e[1], None, None))
            else:
                norm.append((e[0], e[1], e[2].get("edge_type"), e[2]))
        if not norm:
            return
        self._add_nodes_batch([u for u, _, _, _ in norm] +
                              [v for _, v, _, _ in norm])
        ekeys = [(str(u), str(v)) for u, v, _, _ in norm]
        for key in ekeys:
            if key not in self._edge_pos:
                self._edge_pos[key] = len(self._edge_order)
                self._edge_order.append(key)
        new_count = len(self._edge_order) - len(self._edge_types)
        if new_count:
            self._edge_types.extend([None] * new_count)
            for col in self._edge_cols.values():
                col.extend([None] * new_count)
        for _, _, _, d in norm:
            if d:
                for name in d:
                    if name != "edge_type" and name not in self._edge_cols:
                        self._edge_cols[name] = [None] * len(self._edge_order)
        for (_, _, t, d), key in zip(norm, ekeys):
            idx = self._edge_pos[key]
            if t is not None:
                self._edge_types[idx] = t
            if d:
                for k, v in d.items():
                    if k != "edge_type":
                        self._edge_cols[k][idx] = v
        self._edges_dirty = True

    def has_edge(self, u, v):
        return (str(u), str(v)) in self._edge_pos

    def remove_edge(self, u, v):
        key = (str(u), str(v))
        if key not in self._edge_pos:
            raise KeyError((u, v))
        idx = self._edge_pos.pop(key)
        del self._edge_order[idx]
        del self._edge_types[idx]
        for name in self._edge_cols:
            del self._edge_cols[name][idx]
        for i in range(idx, len(self._edge_order)):
            self._edge_pos[self._edge_order[i]] = i
        self._edges_dirty = True

    def edge_attrs(self, u, v):
        idx = self._edge_pos[(str(u), str(v))]
        d = {n: c[idx] for n, c in self._edge_cols.items() if c[idx] is not None}
        if self._edge_types[idx] is not None:
            d["edge_type"] = self._edge_types[idx]
        return d

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
        """Bulk-adopt Arrow tables: no per-row Python loop.

        The input tables become the backing tables directly; Python lookup
        indexes are derived with vectorized ``to_pylist()`` calls. Node
        objects are the string keys (originals are not recoverable from
        a bare table)."""
        g = cls()
        if nodes is not None:
            order = nodes.column("node").to_pylist()
            g._node_order = order
            g._node_pos = {k: i for i, k in enumerate(order)}
            g._node_orig = dict(zip(order, order))
            names = nodes.column_names
            g._node_types = (
                nodes.column("node_type").to_pylist()
                if "node_type" in names else [None] * len(order)
            )
            g._node_cols = {
                name: nodes.column(name).to_pylist()
                for name in names if name not in ("node", "node_type")
            }
            g._nodes_table = nodes
        if edges is not None:
            src = edges.column("source").to_pylist()
            tgt = edges.column("target").to_pylist()
            order = list(zip(src, tgt))
            g._edge_order = order
            g._edge_pos = {k: i for i, k in enumerate(order)}
            names = edges.column_names
            g._edge_types = (
                edges.column("edge_type").to_pylist()
                if "edge_type" in names else [None] * len(order)
            )
            g._edge_cols = {
                name: edges.column(name).to_pylist()
                for name in names if name not in ("source", "target", "edge_type")
            }
            g._edges_table = edges
        return g
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
        for k in self._node_order:
            g.add_node(self._node_orig[k], **self.node_attrs(self._node_orig[k]))
        for s, t in self._edge_order:
            g.add_edge(
                self._node_orig[s],
                self._node_orig[t],
                **self.edge_attrs(self._node_orig[s], self._node_orig[t]),
            )
        return g

    def _rebuild_nodes_table(self):
        cols = {"node": pa.array(self._node_order, type=pa.string())}
        if any(t is not None for t in self._node_types):
            cols["node_type"] = pa.array(self._node_types, type=pa.string())
        for name, vals in self._node_cols.items():
            cols[name] = _typed_column(list(vals))
        self._nodes_table = pa.table(cols)

    def _rebuild_edges_table(self):
        cols = {
            "source": pa.array([s for s, _ in self._edge_order], type=pa.string()),
            "target": pa.array([t for _, t in self._edge_order], type=pa.string()),
        }
        if any(t is not None for t in self._edge_types):
            cols["edge_type"] = pa.array(self._edge_types, type=pa.string())
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
