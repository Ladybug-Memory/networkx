import pytest

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

import networkx as nx
from networkx.classes.arrow_backend import backend_interface
from networkx.classes.arrow_digraph import ArrowDiGraph


@pytest.fixture
def typed_graph():
    g = ArrowDiGraph()
    g.add_node("alice", node_type="user", age=30)
    g.add_node("bob", node_type="user", age=25)
    g.add_node("post1", node_type="post", text="hello")
    g.add_edge("alice", "post1", edge_type="authored")
    g.add_edge("bob", "post1", edge_type="liked", weight=0.9)
    g.add_edge("alice", "bob", edge_type="follows")
    return g


def test_type_columns_and_filters(typed_graph):
    assert typed_graph.node_types == ["post", "user"]
    assert typed_graph.edge_types == ["authored", "follows", "liked"]
    assert typed_graph.nodes_of_type("user") == ["alice", "bob"]
    assert typed_graph.edges_of_type("liked") == [("bob", "post1")]
    assert typed_graph.node_type("alice") == "user"
    assert typed_graph.edge_type("bob", "post1") == "liked"
    assert typed_graph.nodes_table().schema.field("node_type").type == pa.string()
    assert typed_graph.edges_table().schema.field("edge_type").type == pa.string()


def test_heterogeneous_attrs_keep_types(typed_graph):
    schema = typed_graph.nodes_table().schema
    assert schema.field("age").type == pa.int64()
    assert schema.field("text").type == pa.string()


def test_removal_keeps_type_columns_aligned(typed_graph):
    typed_graph.remove_edge("alice", "bob")
    typed_graph.remove_node("bob")
    assert typed_graph.node_types == ["post", "user"]
    assert typed_graph.edge_types == ["authored"]
    assert typed_graph.nodes_table().num_rows == 2
    assert typed_graph.edges_table().num_rows == 1


def test_networkx_roundtrip_preserves_types(typed_graph):
    rt = ArrowDiGraph.from_networkx(typed_graph.to_networkx())
    assert rt.node_types == ["post", "user"]
    assert rt.edge_types == ["authored", "follows", "liked"]
    assert rt.nodes_of_type("post") == ["post1"]
    assert rt.node_attrs("alice")["node_type"] == "user"


def test_arrow_table_roundtrip_preserves_types(typed_graph):
    rt = ArrowDiGraph.from_arrow(typed_graph.nodes_table(), typed_graph.edges_table())
    assert rt.node_types == ["post", "user"]
    assert rt.edges_of_type("follows") == [("alice", "bob")]


def test_bulk_ingest_matches_incremental():
    edges = [(0, 1), (1, 2), (0, 2), (2, 0)]
    attrs = {0: {"color": "red"}, 1: {"color": "blue"}, 2: {}}

    incr = ArrowDiGraph()
    for n, d in attrs.items():
        incr.add_node(n, **d)
    for u, v in edges:
        incr.add_edge(u, v, weight=1.0)

    bulk = ArrowDiGraph()
    bulk.add_nodes_from(list(attrs.items()))
    bulk.add_edges_from([(u, v, {"weight": 1.0}) for u, v in edges])
    assert bulk.nodes_table().equals(incr.nodes_table())
    assert bulk.edges_table().equals(incr.edges_table())

    from_nx = ArrowDiGraph(nx.DiGraph(incr.to_networkx()))
    assert {r["node"] for r in from_nx.nodes_table().to_pylist()} == {
        r["node"] for r in incr.nodes_table().to_pylist()
    }
    assert {(r["source"], r["target"]) for r in from_nx.edges_table().to_pylist()} == {
        (r["source"], r["target"]) for r in incr.edges_table().to_pylist()
    }
    assert from_nx.node_attrs(0) == incr.node_attrs(0)


def test_type_columns_omitted_when_unused():
    g = ArrowDiGraph()
    g.add_node("a", color="red")
    g.add_edge("a", "b", weight=1.0)
    assert "node_type" not in g.nodes_table().column_names
    assert "edge_type" not in g.edges_table().column_names
    assert g.nodes_of_type("user") == []
    assert g.edges_of_type("likes") == []
    assert g.node_types == [] and g.edge_types == []
    # setting a type later materializes the column ...
    g.add_node("a", node_type="user")
    g.add_edge("a", "b", edge_type="likes")
    assert g.nodes_table().column("node_type").to_pylist() == ["user", None]
    assert g.edges_table().column("edge_type").to_pylist() == ["likes"]
    # ... and removing every typed entry drops it again
    g.remove_edge("a", "b")
    g.remove_node("a")
    assert "node_type" not in g.nodes_table().column_names
    assert "edge_type" not in g.edges_table().column_names


def test_backend_constructors():
    g = nx.Graph(backend="arrow")
    d = nx.DiGraph(backend="arrow")
    assert isinstance(g, ArrowDiGraph)
    assert isinstance(d, ArrowDiGraph)
    assert g.__networkx_backend__ == "arrow"

    e = nx.DiGraph([(1, 2), (2, 3)], backend="arrow")
    assert e.number_of_nodes() == 3
    assert e.number_of_edges() == 2

    f = nx.DiGraph(backend="arrow", name="foo")
    assert f.graph == {"name": "foo"}


def test_native_dispatch_matches_networkx(typed_graph):
    expected = typed_graph.to_networkx()
    assert nx.out_degree_centrality(typed_graph, backend="arrow") == (
        nx.out_degree_centrality(expected)
    )
    assert nx.in_degree_centrality(typed_graph, backend="arrow") == (
        nx.in_degree_centrality(expected)
    )
    assert list(nx.topological_sort(typed_graph, backend="arrow")) == list(
        nx.topological_sort(expected)
    )


def test_backend_interface_conversions(typed_graph):
    expected = typed_graph.to_networkx()
    converted = backend_interface.convert_from_nx(expected)
    assert isinstance(converted, ArrowDiGraph)
    assert converted.node_types == ["post", "user"]
    assert converted.to_networkx().nodes["alice"]["node_type"] == "user"
    assert backend_interface.convert_to_nx(converted).nodes["post1"]["node_type"] == (
        "post"
    )
