"""
BERT Encoder Module for BCSD Model

This module handles encoding of binary code features using BERT (Bidirectional Encoder 
Representations from Transformers). It converts assembly instructions and code features 
into dense vector representations.
"""

import torch
import logging
from typing import List, Dict, Any, Optional
from transformers import BertModel, BertTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertEncoder:
    """
    Encodes binary code features using BERT model.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', device: Optional[str] = None):
        """
        Initialize the BERT encoder.
        
        Args:
            model_name: Name of the pretrained BERT model to use
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing BERT encoder with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the BERT model and tokenizer."""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            raise
    
    def encode_instruction(self, instruction: str) -> np.ndarray:
        """
        Encode a single assembly instruction.
        
        Args:
            instruction: Assembly instruction string
            
        Returns:
            Numpy array of encoded features
        """
        return self.encode_text(instruction)
    
    def encode_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Encode text using BERT.
        
        Args:
            text: Input text to encode
            max_length: Maximum sequence length
            
        Returns:
            Numpy array of encoded features (768-dimensional for bert-base)
        """
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token representation as sentence embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding.squeeze()
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return np.zeros(768)  # Return zero vector on error
    
    def encode_instructions_batch(self, instructions: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple instructions in batches.
        
        Args:
            instructions: List of assembly instructions
            batch_size: Number of instructions to process at once
            
        Returns:
            Numpy array of shape (num_instructions, embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(instructions), batch_size):
            batch = instructions[i:i + batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_embeddings.append(embeddings)
                
            except Exception as e:
                logger.error(f"Error encoding batch {i//batch_size}: {e}")
                # Add zero vectors for failed batch
                all_embeddings.append(np.zeros((len(batch), 768)))
        
        return np.vstack(all_embeddings)
    
    def encode_basic_block(self, basic_block: Dict[str, Any]) -> np.ndarray:
        """
        Encode a basic block by encoding its instructions.
        
        Args:
            basic_block: Dictionary containing basic block data with 'instructions' key
            
        Returns:
            Numpy array representing the basic block (average of instruction embeddings)
        """
        instructions = basic_block.get('instructions', [])
        
        if not instructions:
            logger.warning("Basic block has no instructions")
            return np.zeros(768)
        
        # Create instruction strings (mnemonic + operands)
        instruction_texts = []
        for insn in instructions:
            mnemonic = insn.get('mnemonic', '')
            op_str = insn.get('op_str', '')
            instruction_texts.append(f"{mnemonic} {op_str}".strip())
        
        # Encode all instructions
        embeddings = self.encode_instructions_batch(instruction_texts)
        
        # Return average embedding for the basic block
        return np.mean(embeddings, axis=0)
    
    def encode_function(self, function_data: Dict[str, Any], 
                       basic_blocks_data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Encode a function and its basic blocks.
        
        Args:
            function_data: Dictionary containing function information
            basic_blocks_data: List of basic block dictionaries
            
        Returns:
            Dictionary with 'function_embedding' and 'block_embeddings'
        """
        # Get basic blocks belonging to this function
        block_addrs = set(function_data.get('blocks', []))
        relevant_blocks = [b for b in basic_blocks_data if b['address'] in block_addrs]
        
        if not relevant_blocks:
            logger.warning(f"No basic blocks found for function {function_data.get('name')}")
            return {
                'function_embedding': np.zeros(768),
                'block_embeddings': []
            }
        
        # Encode each basic block
        block_embeddings = []
        for block in relevant_blocks:
            block_emb = self.encode_basic_block(block)
            block_embeddings.append(block_emb)
        
        # Function embedding is average of its basic blocks
        function_embedding = np.mean(block_embeddings, axis=0) if block_embeddings else np.zeros(768)
        
        return {
            'function_embedding': function_embedding,
            'block_embeddings': block_embeddings
        }
    
    def encode_binary(self, disassembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode all features from a disassembled binary.
        
        Args:
            disassembly_data: Output from angr_disassembly module
            
        Returns:
            Dictionary containing encoded features for the binary
        """
        logger.info(f"Encoding binary: {disassembly_data.get('binary_path')}")
        
        functions = disassembly_data.get('functions', {})
        basic_blocks = disassembly_data.get('basic_blocks', [])
        
        # Encode all basic blocks
        logger.info(f"Encoding {len(basic_blocks)} basic blocks...")
        block_embeddings = {}
        for block in basic_blocks:
            block_addr = block['address']
            block_embeddings[block_addr] = self.encode_basic_block(block)
        
        # Encode all functions
        logger.info(f"Encoding {len(functions)} functions...")
        function_embeddings = {}
        for func_addr, func_data in functions.items():
            func_encoding = self.encode_function(func_data, basic_blocks)
            # Store only the function embedding, not the entire dict
            function_embeddings[func_addr] = func_encoding['function_embedding']
        
        return {
            'binary_path': disassembly_data.get('binary_path'),
            'block_embeddings': block_embeddings,
            'function_embeddings': function_embeddings
        }


def encode_disassembled_binary(disassembly_data: Dict[str, Any], 
                               model_name: str = 'bert-base-uncased') -> Dict[str, Any]:
    """
    Convenience function to encode a disassembled binary.
    
    Args:
        disassembly_data: Output from angr_disassembly module
        model_name: Name of BERT model to use
        
    Returns:
        Dictionary containing encoded features
    """
    encoder = BertEncoder(model_name=model_name)
    return encoder.encode_binary(disassembly_data)


if __name__ == "__main__":
    import sys
    import json
    
    # Test the encoder with sample data
    logger.info("Testing BERT encoder...")
    
    encoder = BertEncoder()
    
    # Test single instruction encoding
    test_instruction = "mov eax, ebx"
    embedding = encoder.encode_instruction(test_instruction)
    print(f"Encoded instruction: {test_instruction}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding preview: {embedding[:5]}")
    
    # Test batch encoding
    test_instructions = [
        "mov eax, ebx",
        "add eax, 0x10",
        "jmp 0x401000",
        "call 0x402000"
    ]
    batch_embeddings = encoder.encode_instructions_batch(test_instructions)
    print(f"\nEncoded {len(test_instructions)} instructions")
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
