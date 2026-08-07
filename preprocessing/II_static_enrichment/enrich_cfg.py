"""
Single-file CFG enrichment orchestrator.

Loads a raw CFG JSON (from IDA extraction), adds node-level and
function-level statistical features, and writes the enriched output.

Usage:
    python -m preprocessing.II_static_enrichment.enrich_cfg <input_file> [output_file]

If output_file is omitted, appends '_enriched' to the input filename:
    input.json -> input_enriched.json

"""

import json
import sys
from pathlib import Path
from typing import Any

from .function_features import compute_function_statistics
from .node_features import compute_node_statistics


def enrich_cfg_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Enrich CFG data in-memory with node-level and function-level features.

    Args:
        data: Raw CFG dict (as loaded from IDA extraction JSON).

    Returns:
        The same dict, mutated in-place with added features.
    """
    for function in data.get("functions", []):
        for node in function.get("nodes", []):
            instructions = node.get("instructions", {})
            stats = compute_node_statistics(instructions)
            node.update(stats)

        nodes = function.get("nodes", [])
        edges = function.get("edges", [])
        rebased = function.get("rebased_instructions", {})
        func_stats = compute_function_statistics(nodes, edges, rebased)
        function.update(func_stats)

    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["enriched"] = True
    data["metadata"]["enrichment_version"] = "2.0"

    return data


def default_output_path(input_path: str) -> str:
    """Derive default output path by appending '_enriched' before extension."""
    p = Path(input_path)
    return str(p.with_stem(p.stem + "_enriched"))


def enrich_cfg_file(input_path: str, output_path: str | None = None) -> str:
    """
    Enrich a single CFG JSON file and write the result.

    Args:
        input_path: Path to raw CFG JSON.
        output_path: Path to write enriched JSON. If None, appends '_enriched'.

    Returns:
        The output file path that was written.
    """
    if output_path is None:
        output_path = default_output_path(input_path)

    with open(input_path) as f:
        data = json.load(f)

    enrich_cfg_data(data)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    func_count = len(data.get("functions", []))
    node_count = sum(len(fn.get("nodes", [])) for fn in data.get("functions", []))
    print(f"  Enriched {node_count} nodes in {func_count} functions -> {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m preprocessing.II_static_enrichment.enrich_cfg "
            "<input_file> [output_file]"
        )
        print("  If output_file is omitted, appends '_enriched' to input filename.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    enrich_cfg_file(input_path, output_path)


if __name__ == "__main__":
    main()
