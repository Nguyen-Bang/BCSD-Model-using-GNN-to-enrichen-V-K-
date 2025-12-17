"""
CFG Feature Extraction using angr.

This module extracts Control Flow Graphs from binary executables using angr's
CFGFast analysis. Per research.md, we use CFGFast for accurate CFG extraction
without full symbolic execution overhead.

Module: preprocessing.extract_features
Owner: User Story 1 (US1) - Binary Preprocessing
Constitution: Principle II (Reproducible Pipeline) - Independent execution
"""

import angr
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Setup logging
logger = logging.getLogger("bcsd.preprocessing")


def compute_hash(binary_path: str) -> str:
    """
    Compute SHA256 hash of binary file for unique identification.
    
    Args:
        binary_path: Path to binary file
        
    Returns:
        str: SHA256 hash in hexadecimal
    """
    sha256 = hashlib.sha256()
    with open(binary_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_single_cfg(
    binary_path: str,
    output_dir: Optional[str] = None,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Extract Control Flow Graph from a single binary using angr CFGFast.
    
    This is the basic CFG extraction function for Phase 2 validation.
    For full preprocessing pipeline, use extract_cfg() which outputs
    the unified JSON format (tokens + edges).
    
    Args:
        binary_path: Absolute path to binary executable
        output_dir: Directory to save output JSON (optional for Phase 2 testing)
        timeout: Maximum analysis time in seconds (default: 600)
        
    Returns:
        dict: CFG extraction results with structure:
            {
                "status": "success" | "failed",
                "binary_hash": "sha256_hash",
                "binary_path": str,
                "cfg_data": {
                    "nodes": List[dict],  # Basic blocks
                    "edges": List[List[Any]],  # Edge list [src, dst, jumpkind]
                    "node_count": int,
                    "edge_count": int,
                    "entry_points": List[int]
                },
                "functions": List[dict],  # Function metadata
                "error": str,  # Only if failed
                "processing_time_seconds": float
            }
            
    Example:
        >>> result = extract_single_cfg("test_binaries/test_gnn_gcc_O0")
        >>> print(f"Extracted {result['cfg_data']['node_count']} nodes")
        >>> print(f"Functions: {[f['name'] for f in result['functions']]}")
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Loading binary: {binary_path}")
        
        # Compute binary hash
        binary_hash = compute_hash(binary_path)
        
        # Load binary with angr
        project = angr.Project(
            binary_path,
            auto_load_libs=False,  # Don't analyze library code
            load_debug_info=True  # Load symbols if available
        )
        
        logger.info("Running CFGFast analysis...")
        
        # Extract CFG using CFGFast
        cfg = project.analyses.CFGFast(
            normalize=True,
            force_complete_scan=False
        )
        
        logger.info(f"CFG extracted: {len(cfg.graph.nodes())} nodes, {len(cfg.graph.edges())} edges")
        
        # Extract nodes (basic blocks)
        nodes = []
        node_id_map = {}  # Map angr node addresses to sequential IDs
        
        for idx, node in enumerate(cfg.graph.nodes()):
            if hasattr(node, 'addr') and hasattr(node, 'size'):
                node_info = {
                    "id": idx,
                    "addr": hex(node.addr),
                    "size": node.size,
                    "instructions": []
                }
                
                # Extract instructions from the block
                try:
                    block = project.factory.block(node.addr, size=node.size)
                    for insn in block.capstone.insns:
                        node_info["instructions"].append({
                            "addr": hex(insn.address),
                            "mnemonic": insn.mnemonic,
                            "op_str": insn.op_str,
                            "full": f"{insn.mnemonic} {insn.op_str}".strip()
                        })
                except Exception as e:
                    logger.warning(f"Could not disassemble block at {hex(node.addr)}: {e}")
                
                nodes.append(node_info)
                node_id_map[node.addr] = idx
        
        # Extract edges
        edges = []
        for src, dst, data in cfg.graph.edges(data=True):
            if hasattr(src, 'addr') and hasattr(dst, 'addr'):
                src_id = node_id_map.get(src.addr)
                dst_id = node_id_map.get(dst.addr)
                if src_id is not None and dst_id is not None:
                    edges.append([src_id, dst_id, data.get('jumpkind', 'Ijk_Boring')])
        
        # Find entry points
        entry_points = []
        if project.entry in node_id_map:
            entry_points.append(node_id_map[project.entry])
        
        # Extract function information
        functions = []
        for func_addr, func in cfg.functions.items():
            func_info = {
                "name": func.name,
                "addr": hex(func_addr),
                "size": func.size,
                "is_plt": func.is_plt,
                "is_simprocedure": func.is_simprocedure
            }
            functions.append(func_info)
        
        logger.info(f"Found {len(functions)} functions")
        
        # Build result
        result = {
            "status": "success",
            "binary_hash": binary_hash,
            "binary_path": binary_path,
            "cfg_data": {
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "entry_points": entry_points
            },
            "functions": functions,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
        }
        
        # Optionally save to file
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            output_file = output_path / f"{binary_hash}_cfg.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Saved CFG to: {output_file}")
            result["output_file"] = str(output_file)
        
        return result
        
    except FileNotFoundError:
        error_msg = f"Binary not found: {binary_path}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "binary_path": binary_path,
            "error": error_msg,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
        }
        
    except Exception as e:
        error_msg = f"CFG extraction failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "failed",
            "binary_path": binary_path,
            "error": error_msg,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
        }


def extract_cfg(
    binary_path: str,
    output_dir: str,
    tokenizer=None,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Extract CFG and output unified JSON format for full preprocessing pipeline (T015).
    
    This function is used in Phase 3+ for complete preprocessing.
    Outputs single {hash}.json file with tokenized instructions + edge list.
    
    Format of {hash}.json:
    {
        "binary_hash": str,
        "binary_path": str,
        "binary_name": str,
        "functions": [
            {
                "function_name": str,
                "function_addr": str,
                "nodes": [
                    {
                        "id": int,
                        "addr": str,
                        "instructions": [str],  # Raw instruction strings
                        "token_ids": [int],     # Tokenized if tokenizer provided
                        "attention_mask": [int]
                    }
                ],
                "edges": [[src_id, dst_id, jumpkind]]  # Edge list for this function's CFG
            }
        ],
        "metadata": {
            "total_nodes": int,
            "total_edges": int,
            "total_functions": int,
            "processing_time": float
        }
    }
    
    Args:
        binary_path: Path to binary
        output_dir: Output directory for preprocessed/{hash}.json
        tokenizer: AssemblyTokenizer instance (optional, for tokenization)
        timeout: Analysis timeout in seconds
        
    Returns:
        dict: Processing result with status and file paths
            {
                "status": "success" | "failed",
                "binary_hash": str,
                "output_file": str,  # Only if successful
                "error": str,        # Only if failed
                "processing_time_seconds": float
            }
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing binary: {binary_path}")
        
        # Compute binary hash
        binary_hash = compute_hash(binary_path)
        binary_name = Path(binary_path).name
        
        # Load binary with angr
        project = angr.Project(
            binary_path,
            auto_load_libs=False,
            load_debug_info=True
        )
        
        logger.info("Running CFGFast analysis...")
        
        # Extract CFG using CFGFast (with timeout handling)
        cfg = project.analyses.CFGFast(
            normalize=True,
            force_complete_scan=False
        )
        
        logger.info(f"CFG extracted: {len(cfg.graph.nodes())} nodes, {len(cfg.graph.edges())} edges")
        
        # Build node ID mapping
        node_id_map = {}
        for idx, node in enumerate(cfg.graph.nodes()):
            if hasattr(node, 'addr'):
                node_id_map[node.addr] = idx
        
        # Process each function
        functions_data = []
        
        for func_addr, func in cfg.functions.items():
            # Skip PLT stubs and SimProcedures
            if func.is_plt or func.is_simprocedure:
                continue
            
            function_nodes = []
            function_edges = []
            
            # Build function-local node ID mapping
            func_node_addrs = set()
            for block in func.blocks:
                if hasattr(block, 'addr'):
                    func_node_addrs.add(block.addr)
            
            if not func_node_addrs:
                continue  # Skip functions with no blocks
            
            # Create local ID mapping for this function
            local_id_map = {addr: idx for idx, addr in enumerate(sorted(func_node_addrs))}
            
            # Extract nodes for this function
            for block in func.blocks:
                if not hasattr(block, 'addr') or not hasattr(block, 'size'):
                    continue
                
                try:
                    # Disassemble block
                    instructions = []
                    block_obj = project.factory.block(block.addr, size=block.size)
                    
                    for insn in block_obj.capstone.insns:
                        inst_str = f"{insn.mnemonic} {insn.op_str}".strip()
                        instructions.append(inst_str)
                    
                    node_data = {
                        "id": local_id_map[block.addr],
                        "addr": hex(block.addr),
                        "instructions": instructions
                    }
                    
                    # Tokenize if tokenizer provided
                    if tokenizer and instructions:
                        tokens = tokenizer.tokenize(instructions, add_special_tokens=False)
                        node_data["token_ids"] = tokens["token_ids"]
                        node_data["attention_mask"] = tokens["attention_mask"]
                    
                    function_nodes.append(node_data)
                    
                except Exception as e:
                    logger.warning(f"Could not disassemble block at {hex(block.addr)}: {e}")
                    continue
            
            # Extract edges for this function
            for src, dst, data in cfg.graph.edges(data=True):
                if hasattr(src, 'addr') and hasattr(dst, 'addr'):
                    if src.addr in local_id_map and dst.addr in local_id_map:
                        function_edges.append([
                            local_id_map[src.addr],
                            local_id_map[dst.addr],
                            data.get('jumpkind', 'Ijk_Boring')
                        ])
            
            # Add function data
            functions_data.append({
                "function_name": func.name,
                "function_addr": hex(func_addr),
                "nodes": function_nodes,
                "edges": function_edges
            })
        
        # Build unified output
        total_nodes = sum(len(f["nodes"]) for f in functions_data)
        total_edges = sum(len(f["edges"]) for f in functions_data)
        
        output_data = {
            "binary_hash": binary_hash,
            "binary_path": binary_path,
            "binary_name": binary_name,
            "functions": functions_data,
            "metadata": {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "total_functions": len(functions_data),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        }
        
        # Save to output file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"{binary_hash}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved preprocessed data to: {output_file}")
        logger.info(f"  Functions: {len(functions_data)}, Nodes: {total_nodes}, Edges: {total_edges}")
        
        return {
            "status": "success",
            "binary_hash": binary_hash,
            "binary_name": binary_name,
            "output_file": str(output_file),
            "total_functions": len(functions_data),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
        }
        
    except FileNotFoundError:
        error_msg = f"Binary not found: {binary_path}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "binary_path": binary_path,
            "error": error_msg,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
        }
        
    except Exception as e:
        error_msg = f"CFG extraction failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "failed",
            "binary_path": binary_path,
            "error": error_msg,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
        }



if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    parser = argparse.ArgumentParser(description="Extract CFG features from binary")
    parser.add_argument("binary_path", help="Path to binary file")
    parser.add_argument("output_dir", nargs="?", help="Directory to save JSON output")
    parser.add_argument("--visualize", action="store_true", help="Generate CFG visualization (requires angr-utils)")
    
    args = parser.parse_args()
    
    # Setup simple logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Standard JSON extraction
    result = extract_single_cfg(args.binary_path, args.output_dir)
    
    if result["status"] == "success":
        print(f"\n✓ CFG extraction successful!")
        print(f"  Nodes: {result['cfg_data']['node_count']}")
        print(f"  Edges: {result['cfg_data']['edge_count']}")
        print(f"  Functions: {len(result['functions'])}")
        print(f"  Processing time: {result['processing_time_seconds']:.2f}s")
        if "output_file" in result:
            print(f"  Output: {result['output_file']}")
    else:
        print(f"\n✗ CFG extraction failed: {result['error']}")
        sys.exit(1)

    # Visualization (Debugging only)
    if args.visualize:
        print("\n--- Generating Visualization ---")
        try:
            from angrutils import plot_cfg
            
            # We re-run analysis here to keep extract_single_cfg pure
            print(f"Reloading {args.binary_path} for visualization...")
            project = angr.Project(
                args.binary_path,
                auto_load_libs=False,
                load_debug_info=True
            )
            
            print("Re-running CFGFast...")
            cfg = project.analyses.CFGFast(normalize=True, force_complete_scan=False)
            
            # Determine output name
            bin_name = os.path.basename(args.binary_path)
            output_name = f"{bin_name}_cfg"
            if args.output_dir:
                output_name = os.path.join(args.output_dir, output_name)
            
            print(f"Plotting to {output_name}...")
            plot_cfg(
                cfg, 
                output_name, 
                asminst=True, 
                remove_imports=True, 
                remove_path_terminator=True
            )
            print(f"✓ Visualization saved to {output_name}.png (and .dot)")
            
        except ImportError:
            print("✗ Error: 'angr-utils' or 'bingraphvis' not installed.")
            print("  Install with: pip install angr-utils bingraphvis")
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
