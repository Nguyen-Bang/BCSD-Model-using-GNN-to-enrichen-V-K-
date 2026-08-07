"""Tests for reverse edges and typed self-loops during collation."""

from dataset.collate import (
    REVERSE_TYPE_OFFSET,
    SELF_LOOP_TYPE,
    _collate_graphs,
    collate_gnn,
)


def _mock_function(num_nodes, edges_with_types):
    """edges_with_types: list of (src, tgt, type_id_0_to_4)."""
    return {
        "node_features": [[0.0] * 20 for _ in range(num_nodes)],
        "edges": [[s, t] for s, t, _ in edges_with_types],
        "edge_types": [tp for _, _, tp in edges_with_types],
        "node_count": num_nodes,
        "edge_count": len(edges_with_types),
        "token_ids": [0],
        "attention_mask": [1],
        "token_type_ids": [0],
        "function_features": [0.0] * 132,
    }


def test_legacy_keys_unchanged():
    """Original edge_index/edge_types keys must be byte-identical to today."""
    binaries = [_mock_function(3, [(0, 1, 0), (1, 2, 2)])]
    out_legacy = _collate_graphs(binaries, use_edge_features=False, use_reverse_edges=False)
    assert out_legacy["edge_index"].shape == (2, 2)
    assert out_legacy["edge_types"].tolist() == [0, 2]
    assert "gnn_edge_index" not in out_legacy
    assert "gnn_edge_types" not in out_legacy


def test_augmented_with_reverse_and_self_loops():
    """With both flags on: 2 fwd + 2 rev + 3 self-loop = 7 edges."""
    binaries = [_mock_function(3, [(0, 1, 0), (1, 2, 2)])]
    out = _collate_graphs(binaries, use_edge_features=True, use_reverse_edges=True)

    assert out["edge_index"].shape == (2, 2)
    assert out["edge_types"].tolist() == [0, 2]

    assert out["gnn_edge_index"].shape == (2, 7), out["gnn_edge_index"].shape
    assert out["gnn_edge_types"].shape == (7,), out["gnn_edge_types"].shape

    src = out["gnn_edge_index"][0].tolist()
    tgt = out["gnn_edge_index"][1].tolist()
    types = out["gnn_edge_types"].tolist()

    assert (src[0], tgt[0], types[0]) == (0, 1, 0)
    assert (src[1], tgt[1], types[1]) == (1, 2, 2)
    assert (src[2], tgt[2], types[2]) == (1, 0, 0 + REVERSE_TYPE_OFFSET)
    assert (src[3], tgt[3], types[3]) == (2, 1, 2 + REVERSE_TYPE_OFFSET)
    assert (src[4], tgt[4], types[4]) == (0, 0, SELF_LOOP_TYPE)
    assert (src[5], tgt[5], types[5]) == (1, 1, SELF_LOOP_TYPE)
    assert (src[6], tgt[6], types[6]) == (2, 2, SELF_LOOP_TYPE)

    assert out["gnn_edge_index"].size(1) == out["gnn_edge_types"].size(0)


def test_no_reverse_only_self_loops():
    """use_edge_features=True but use_reverse_edges=False: fwd + self-loops only."""
    binaries = [_mock_function(3, [(0, 1, 0), (1, 2, 2)])]
    out = _collate_graphs(binaries, use_edge_features=True, use_reverse_edges=False)
    # 2 fwd + 3 self = 5
    assert out["gnn_edge_index"].shape == (2, 5)
    assert out["gnn_edge_types"].tolist() == [
        0,
        2,
        SELF_LOOP_TYPE,
        SELF_LOOP_TYPE,
        SELF_LOOP_TYPE,
    ]


def test_offset_across_graphs():
    """Per-graph node-id offsets propagate into both legacy and augmented."""
    binaries = [
        _mock_function(2, [(0, 1, 0)]),
        _mock_function(3, [(0, 2, 1)]),
    ]
    out = _collate_graphs(binaries, use_edge_features=True, use_reverse_edges=True)
    assert out["edge_index"][0].tolist() == [0, 2]
    assert out["edge_index"][1].tolist() == [1, 4]
    # g0: fwd(0->1,t=0), rev(1->0,t=REVERSE_TYPE_OFFSET+0), self(0,1)
    # g1: fwd(2->4,t=1), rev(4->2,t=REVERSE_TYPE_OFFSET+1), self(2,3,4)
    R = REVERSE_TYPE_OFFSET
    S = SELF_LOOP_TYPE
    expected_src = [0, 1, 0, 1, 2, 4, 2, 3, 4]
    expected_tgt = [1, 0, 0, 1, 4, 2, 2, 3, 4]
    expected_typ = [0, R, S, S, 1, R + 1, S, S, S]
    assert out["gnn_edge_index"][0].tolist() == expected_src
    assert out["gnn_edge_index"][1].tolist() == expected_tgt
    assert out["gnn_edge_types"].tolist() == expected_typ


def test_collate_gnn_exposes_new_keys():
    """collate_gnn end-to-end produces gnn_edge_index_a/b under flag."""
    sample = {
        "binary_1": _mock_function(2, [(0, 1, 0)]),
        "binary_2": _mock_function(2, [(0, 1, 2)]),
        "label": 1,
        "metadata": {
            "project": "x",
            "function_name": "f",
            "binary_1_name": "a",
            "binary_2_name": "b",
        },
    }
    batch = collate_gnn([sample], use_edge_features=True, use_reverse_edges=True)
    assert "gnn_edge_index_a" in batch
    assert "gnn_edge_index_b" in batch
    assert "gnn_edge_types_a" in batch
    assert "gnn_edge_types_b" in batch
    # Originals also still present.
    assert "edge_index_a" in batch
    assert "edge_types_a" in batch
    # Defaults-off keeps gnn_* out.
    batch2 = collate_gnn([sample])
    assert "gnn_edge_index_a" not in batch2


def test_original_self_loop_dedup():
    """Original CFG self-loops are neither reversed nor duplicated."""
    # 3 nodes, one original self-loop on node 1 (type=1, uncond_jump),
    # and one normal edge 0->2 (type=0, conditional_jump).
    binaries = [_mock_function(3, [(0, 2, 0), (1, 1, 1)])]
    out = _collate_graphs(binaries, use_edge_features=True, use_reverse_edges=True)

    # Forward: 2 edges (both kept).
    # Reverse: only (2->0, type=5). The (1->1) self-loop is NOT reversed.
    # Self-loops: nodes 0 and 2 get type=10. Node 1 already has its own
    # self-loop, so it is NOT re-emitted as SELF_LOOP_TYPE.
    # Total: 2 fwd + 1 rev + 2 self = 5
    assert out["gnn_edge_index"].shape == (2, 5), out["gnn_edge_index"].shape

    src = out["gnn_edge_index"][0].tolist()
    tgt = out["gnn_edge_index"][1].tolist()
    typ = out["gnn_edge_types"].tolist()
    assert (src[0], tgt[0], typ[0]) == (0, 2, 0)
    assert (src[1], tgt[1], typ[1]) == (1, 1, 1)
    assert (src[2], tgt[2], typ[2]) == (2, 0, 5)
    # Type-10 self-loops emitted only for nodes 0 and 2 (not node 1).
    self_loops = [(s, t, ty) for s, t, ty in zip(src, tgt, typ, strict=False) if ty == 10]
    assert sorted(self_loops) == [(0, 0, 10), (2, 2, 10)], self_loops
    # Node 1's only self-edge is the original type=1, not type=10.
    node1_self = [(s, t, ty) for s, t, ty in zip(src, tgt, typ, strict=False) if s == 1 and t == 1]
    assert node1_self == [(1, 1, 1)], node1_self
