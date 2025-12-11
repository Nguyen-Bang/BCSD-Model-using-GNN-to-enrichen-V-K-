"""
Angr Disassembly Module for BCSD Model

Extracts linearized instruction sequences and CFG structure from binary functions.
Designed for Graph-Aware Language Model training.
"""

import angr
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: Import visualization libraries (only needed for testing)
try:
    from angrutils import plot_cfg
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("angrutils not available. Install with: pip install angrutils")


def extract_function_data(binary_path: str) -> List[Dict[str, Any]]:
    """
    Extract linearized instructions and CFG structure for all functions in a binary.
    
    Args:
        binary_path: Path to the X64 ELF binary
        
    Returns:
        List of dictionaries, one per function, containing:
        - function_name: Name of the function
        - function_address: Starting address of the function
        - blocks: List of basic blocks with id, address, and instructions
        - edges: List of directed edges [source_id, target_id] between blocks
    """
    try:
        # Load binary (crucial: no system libraries)
        logger.info(f"Loading binary: {binary_path}")
        proj = angr.Project(binary_path, auto_load_libs=False)
        
        # Generate CFG
        logger.info("Generating Control Flow Graph...")
        cfg = proj.analyses.CFGFast(normalize=True)
        
        functions_data = []
        
        # Iterate through all functions
        for func_addr, func in cfg.kb.functions.items():
            # Skip system placeholders (SimProcedures, PLT stubs, external functions)
            if func.is_simprocedure or func.is_plt:
                continue
            
            # Skip unresolvable targets and unnamed functions starting with 'sub_'
            # (optional - you can include 'sub_' functions if needed for analysis)
            if func.name in ['UnresolvableCallTarget', 'UnresolvableJumpTarget']:
                continue
            
            try:
                # Extract blocks and sort by address for linearization
                blocks_list = sorted(func.blocks, key=lambda b: b.addr)
                
                # Build address-to-id mapping for edge creation
                addr_to_id = {block.addr: idx for idx, block in enumerate(blocks_list)}
                
                # Extract block data with instructions
                blocks_data = []
                for block_id, block in enumerate(blocks_list):
                    try:
                        # Get Capstone disassembly
                        instructions = []
                        for insn in block.capstone.insns:
                            # Concatenate mnemonic and operands
                            token = f"{insn.mnemonic} {insn.op_str}".strip()
                            instructions.append(token)
                        
                        blocks_data.append({
                            "id": block_id,
                            "address": hex(block.addr),
                            "instructions": instructions
                        })
                    except Exception as e:
                        logger.warning(f"Failed to disassemble block at {hex(block.addr)}: {e}")
                        continue
                
                # Extract CFG edges between blocks
                edges = []
                for block in blocks_list:
                    source_id = addr_to_id.get(block.addr)
                    if source_id is None:
                        continue
                    
                    # Get successors from CFG graph
                    for successor in cfg.graph.successors(block):
                        target_addr = successor.addr
                        target_id = addr_to_id.get(target_addr)
                        
                        # Only include edges within this function's blocks
                        if target_id is not None:
                            edges.append([source_id, target_id])
                
                # Build function data structure
                function_data = {
                    "function_name": func.name,
                    "function_address": hex(func_addr),
                    "blocks": blocks_data,
                    "edges": edges
                }
                
                functions_data.append(function_data)
                
            except Exception as e:
                logger.warning(f"Error processing function {func.name} at {hex(func_addr)}: {e}")
                continue
        
        logger.info(f"Successfully extracted {len(functions_data)} functions")
        return functions_data
        
    except Exception as e:
        logger.error(f"Error loading or analyzing binary: {e}")
        return []


def visualize_function_cfg(binary_path: str, function_name: Optional[str] = None, output_path: str = "cfg_visualization.pdf"):
    """
    Visualize the CFG for a specific function or the entire binary (for testing purposes).
    
    Args:
        binary_path: Path to the binary
        function_name: Name of specific function to visualize (None for full CFG)
        output_path: Where to save the visualization (PDF format)
    
    Returns:
        True if successful, False otherwise
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Visualization not available. Install angr-utils: pip install angr-utils")
        return False
    
    try:
        logger.info(f"Loading binary for visualization: {binary_path}")
        proj = angr.Project(binary_path, auto_load_libs=False)
        cfg = proj.analyses.CFGFast(normalize=True)
        
        if function_name:
            # Visualize specific function
            logger.info(f"Visualizing function: {function_name}")
            func = cfg.kb.functions.get_by_name(function_name)
            if not func:
                logger.error(f"Function '{function_name}' not found")
                return False
            func = list(func)[0]  # Get first match
            plot_cfg(cfg, output_path, func_addr=func.addr, format="pdf")
        else:
            # Visualize entire CFG
            logger.info("Visualizing entire CFG")
            plot_cfg(cfg, output_path, format="pdf")
        
        logger.info(f"Visualization saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        return False


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python angr_disassembly.py <binary_path> [--save] [--visualize [function_name]]")
        print("\nOptions:")
        print("  --save              Save function data to JSON")
        print("  --visualize [func]  Visualize CFG (optional: specify function name)")
        sys.exit(1)
    
    binary_path = sys.argv[1]
    
    # Check for visualization flag
    if "--visualize" in sys.argv:
        viz_idx = sys.argv.index("--visualize")
        func_name = sys.argv[viz_idx + 1] if len(sys.argv) > viz_idx + 1 and not sys.argv[viz_idx + 1].startswith("--") else None
        output = f"{func_name}_cfg.pdf" if func_name else "full_cfg.pdf"
        visualize_function_cfg(binary_path, func_name, output)
        sys.exit(0)
    
    # Extract function data
    functions_data = extract_function_data(binary_path)
    
    # Print summary
    print(f"\nExtracted {len(functions_data)} functions")
    for func in functions_data[:3]:  # Show first 3 as examples
        print(f"\nFunction: {func['function_name']} @ {func['function_address']}")
        print(f"  Blocks: {len(func['blocks'])}")
        print(f"  Edges: {len(func['edges'])}")
        if func['blocks']:
            print(f"  Sample instructions: {func['blocks'][0]['instructions'][:3]}")
    
    # Optionally save to JSON
    if "--save" in sys.argv:
        output_file = "function_data.json"
        with open(output_file, 'w') as f:
            json.dump(functions_data, f, indent=2)
        print(f"\nSaved to {output_file}")
