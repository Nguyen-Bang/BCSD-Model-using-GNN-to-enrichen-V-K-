"""
BCSD Pipeline Package

Binary Code Similarity Detection using:
- Angr for disassembly
- BERT for encoding
- GNN for graph-based enrichment
"""

from .angr_disassembly import AngrDisassembler, disassemble_binary, disassemble_binaries_folder
from .bert_encoder import BertEncoder, encode_disassembled_binary
from .gnn_model import GNNModel, GNNProcessor, process_encoded_binary
from .pipeline import BCSDPipeline

__version__ = '0.1.0'
__all__ = [
    'AngrDisassembler',
    'disassemble_binary',
    'disassemble_binaries_folder',
    'BertEncoder',
    'encode_disassembled_binary',
    'GNNModel',
    'GNNProcessor',
    'process_encoded_binary',
    'BCSDPipeline'
]
