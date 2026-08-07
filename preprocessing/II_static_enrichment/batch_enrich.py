#!/usr/bin/env python3
"""
Batch static enrichment of CFG JSON files.

Recursively processes all JSON files in input_dir, adds node-level and
function-level features, and writes enriched output to output_dir
preserving subdirectory structure.

Usage:
    python -m preprocessing.II_static_enrichment.batch_enrich <input_dir> <output_dir>

Example:
    python -m preprocessing.II_static_enrichment.batch_enrich data/I_ida_processed data/II_static_enriched
"""

import argparse
import sys
from pathlib import Path

from .enrich_cfg import enrich_cfg_file


def batch_enrich(input_dir: str, output_dir: str) -> int:
    """
    Enrich all JSON files in input_dir, writing to output_dir.

    Recurses into subdirectories and preserves folder structure.
    Output filenames get '_enriched' appended to the stem.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    json_files = sorted(input_dir.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0

    print(f"Enriching {len(json_files)} files from {input_dir} -> {output_dir}\n")

    successful = 0
    failed = 0

    for i, json_path in enumerate(json_files, 1):
        rel_path = json_path.relative_to(input_dir)
        out_path = output_dir / rel_path.parent / f"{json_path.stem}_enriched.json"

        print(f"[{i}/{len(json_files)}] {rel_path}")
        try:
            enrich_cfg_file(str(json_path), str(out_path))
            successful += 1
        except Exception as e:
            print(f"  Error: {e}")
            failed += 1

    print(f"\nDone. {successful} succeeded, {failed} failed.")
    return failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch enrich CFG JSON files with statistical features"
    )
    parser.add_argument(
        "input_dir",
        help="Input directory containing raw CFG JSONs (searched recursively)",
    )
    parser.add_argument("output_dir", help="Output directory for enriched JSON files")
    args = parser.parse_args(argv)

    return min(batch_enrich(args.input_dir, args.output_dir), 1)


if __name__ == "__main__":
    sys.exit(main())
