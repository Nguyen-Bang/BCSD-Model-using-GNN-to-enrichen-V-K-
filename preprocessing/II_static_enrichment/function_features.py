"""
Function-level statistical feature extraction.

Pure computation — no file I/O. Computes structural function-level
features and opcode hash fingerprints from already-extracted data.

Features:
- node_count, edge_count, function_size, cyclomatic_complexity (structural)
- opcode_sha256: exact fingerprint of the opcode sequence
- opcode_minhash: locality-sensitive hash signature (128 values) for fuzzy matching

Module: preprocessing.II_static_enrichment.function_features
"""

import hashlib
from typing import Any

from datasketch import MinHash

MINHASH_NUM_PERM = 128


def _extract_opcodes(rebased_instructions: dict[str, str]) -> list[str]:
    """
    Extract the mnemonic (opcode) from each rebased instruction.

    Args:
        rebased_instructions: Dict mapping index string to instruction string,
            e.g. {"1": "mov rax, rdi", "2": "add rsp, 8"}.

    Returns:
        Ordered list of opcodes, e.g. ["mov", "add"].
    """
    sorted_keys = sorted(rebased_instructions.keys(), key=lambda k: int(k))
    opcodes = []
    for k in sorted_keys:
        instr = rebased_instructions[k].strip()
        if instr:
            opcodes.append(instr.split()[0].lower())
    return opcodes


def compute_opcode_sha256(opcodes: list[str]) -> str:
    """
    Compute SHA-256 hash of the opcode sequence.

    Produces an exact fingerprint: two functions with identical opcode
    sequences produce the same hash. One opcode difference = completely
    different hash.

    Args:
        opcodes: Ordered list of opcode mnemonics.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    canonical = " ".join(opcodes)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_opcode_minhash(opcodes: list[str], num_perm: int = MINHASH_NUM_PERM) -> list[int]:
    """
    Compute MinHash signature of the opcode bigram set.

    Uses bigrams (consecutive opcode pairs) as the shingle set so that
    ordering information is partially preserved. Two functions with similar
    opcode sequences will share many MinHash values.

    Args:
        opcodes: Ordered list of opcode mnemonics.
        num_perm: Number of hash permutations (signature length).

    Returns:
        List of `num_perm` integer hash values (the MinHash signature).
    """
    m = MinHash(num_perm=num_perm)
    if len(opcodes) < 2:
        for op in opcodes:
            m.update(op.encode("utf-8"))
    else:
        for i in range(len(opcodes) - 1):
            bigram = f"{opcodes[i]} {opcodes[i + 1]}"
            m.update(bigram.encode("utf-8"))
    return m.hashvalues.tolist()


def compute_function_statistics(
    nodes: list[dict], edges: list, rebased_instructions: dict[str, str] = None
) -> dict[str, Any]:
    """
    Compute function-level features from extracted nodes, edges, and instructions.

    Args:
        nodes: List of node dicts (each must have a "size" key).
        edges: List of edges (each is [src, dst, type] or similar).
        rebased_instructions: Dict mapping instruction index to instruction string.
            If provided, opcode hashes are computed. If None, hash fields are omitted.

    Returns:
        Dictionary with structural stats and opcode hash features.
    """
    num_nodes = len(nodes)
    num_edges = len(edges)
    function_size = sum(node.get("size", 0) for node in nodes)
    cyclomatic_complexity = num_edges - num_nodes + 2

    result = {
        "node_count": num_nodes,
        "edge_count": num_edges,
        "function_size": function_size,
        "function_cyclomatic_complexity": cyclomatic_complexity,
    }

    if rebased_instructions:
        opcodes = _extract_opcodes(rebased_instructions)
        if opcodes:
            result["opcode_sha256"] = compute_opcode_sha256(opcodes)
            result["opcode_minhash"] = compute_opcode_minhash(opcodes)
        else:
            result["opcode_sha256"] = None
            result["opcode_minhash"] = None

    return result
