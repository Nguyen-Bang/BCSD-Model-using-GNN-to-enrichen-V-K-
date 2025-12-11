"""
Angr Disassembly Module for BCSD Model

This module handles the disassembly of binary files using angr.
It extracts control flow graphs, basic blocks, and instructions from binary executables.
"""

import angr
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AngrDisassembler:
    """
    Handles binary disassembly using the angr framework.
    """
    
    def __init__(self, binary_path: str):
        """
        Initialize the disassembler with a binary file path.
        
        Args:
            binary_path: Path to the binary file to disassemble
        """
        self.binary_path = Path(binary_path)
        self.project = None
        self.cfg = None
        
    def load_binary(self) -> bool:
        """
        Load the binary file into angr project.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading binary: {self.binary_path}")
            self.project = angr.Project(str(self.binary_path), auto_load_libs=False)
            return True
        except Exception as e:
            logger.error(f"Error loading binary: {e}")
            return False
    
    def generate_cfg(self) -> bool:
        """
        Generate Control Flow Graph for the binary.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Generating Control Flow Graph...")
            self.cfg = self.project.analyses.CFGFast()
            return True
        except Exception as e:
            logger.error(f"Error generating CFG: {e}")
            return False
    
    def extract_functions(self) -> Dict[str, Any]:
        """
        Extract function information from the binary.
        
        Returns:
            Dictionary containing function data
        """
        if not self.cfg:
            logger.error("CFG not generated. Call generate_cfg() first.")
            return {}
        
        functions_data = {}
        for func_addr, func in self.cfg.kb.functions.items():
            functions_data[func_addr] = {
                'name': func.name,
                'address': func_addr,
                'size': func.size,
                'blocks': [block.addr for block in func.blocks],
                'is_plt': func.is_plt,
                'is_simprocedure': func.is_simprocedure
            }
        
        logger.info(f"Extracted {len(functions_data)} functions")
        return functions_data
    
    def extract_basic_blocks(self) -> List[Dict[str, Any]]:
        """
        Extract basic block information from the binary.
        
        Returns:
            List of basic block data
        """
        if not self.cfg:
            logger.error("CFG not generated. Call generate_cfg() first.")
            return []
        
        blocks_data = []
        for node in self.cfg.graph.nodes():
            if hasattr(node, 'block') and node.block:
                block_data = {
                    'address': node.addr,
                    'size': node.size,
                    'instructions': []
                }
                
                # Extract instructions from the block
                try:
                    block = self.project.factory.block(node.addr, size=node.size)
                    for insn in block.capstone.insns:
                        block_data['instructions'].append({
                            'address': insn.address,
                            'mnemonic': insn.mnemonic,
                            'op_str': insn.op_str
                        })
                except Exception as e:
                    logger.warning(f"Error extracting instructions from block at {hex(node.addr)}: {e}")
                
                blocks_data.append(block_data)
        
        logger.info(f"Extracted {len(blocks_data)} basic blocks")
        return blocks_data
    
    def disassemble(self) -> Dict[str, Any]:
        """
        Main method to perform complete disassembly.
        
        Returns:
            Dictionary containing all disassembly data
        """
        if not self.load_binary():
            return {}
        
        if not self.generate_cfg():
            return {}
        
        result = {
            'binary_path': str(self.binary_path),
            'functions': self.extract_functions(),
            'basic_blocks': self.extract_basic_blocks()
        }
        
        return result


def disassemble_binary(binary_path: str) -> Dict[str, Any]:
    """
    Convenience function to disassemble a binary file.
    
    Args:
        binary_path: Path to the binary file
        
    Returns:
        Dictionary containing disassembly data
    """
    disassembler = AngrDisassembler(binary_path)
    return disassembler.disassemble()


def disassemble_binaries_folder(folder_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Disassemble all binaries in a folder.
    
    Args:
        folder_path: Path to folder containing binary files
        
    Returns:
        Dictionary mapping binary names to their disassembly data
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return {}
    
    results = {}
    binary_files = [f for f in folder.iterdir() if f.is_file()]
    
    logger.info(f"Found {len(binary_files)} files in {folder_path}")
    
    for binary_file in binary_files:
        logger.info(f"Processing: {binary_file.name}")
        try:
            result = disassemble_binary(str(binary_file))
            if result:
                results[binary_file.name] = result
        except Exception as e:
            logger.error(f"Error processing {binary_file.name}: {e}")
    
    logger.info(f"Successfully disassembled {len(results)} binaries")
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python angr_disassembly.py <binary_path_or_folder>")
        sys.exit(1)
    
    path = sys.argv[1]
    path_obj = Path(path)
    
    if path_obj.is_file():
        result = disassemble_binary(path)
        print(f"Disassembled {path}")
        print(f"Functions: {len(result.get('functions', {}))}")
        print(f"Basic Blocks: {len(result.get('basic_blocks', []))}")
    elif path_obj.is_dir():
        results = disassemble_binaries_folder(path)
        print(f"Disassembled {len(results)} binaries from {path}")
    else:
        print(f"Invalid path: {path}")
        sys.exit(1)
