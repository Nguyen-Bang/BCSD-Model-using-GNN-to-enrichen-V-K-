"""
Static feature enrichment subpackage.

Adds node-level and function-level features to extracted CFG JSON files.
All computation is regex-based and does not require IDA Pro.

Files:
    node_features.py      -- Pure logic: compute_node_statistics()
    function_features.py  -- Pure logic: compute_function_statistics()
    enrich_cfg.py         -- Single-file CLI: enrich_cfg.py <input_file> [output_file]
    batch_enrich.py       -- Batch CLI: batch_enrich.py <input_dir> <output_dir>
"""

from .enrich_cfg import enrich_cfg_data as enrich_cfg_data
from .enrich_cfg import enrich_cfg_file as enrich_cfg_file
from .function_features import (
    compute_function_statistics as compute_function_statistics,
)
from .node_features import compute_node_statistics as compute_node_statistics

__all__ = [
    "compute_function_statistics",
    "compute_node_statistics",
    "enrich_cfg_data",
    "enrich_cfg_file",
]
