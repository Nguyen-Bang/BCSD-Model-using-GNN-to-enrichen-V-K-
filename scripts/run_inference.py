"""
Inference Entry Point for BCSD Model

Command-line script to vectorize binaries and perform similarity search.

Usage:
    # Vectorize a directory of binaries
    python scripts/run_inference.py vectorize --input Dataset-1/clamav/ --output embeddings/clamav/
    
    # Search for similar binaries
    python scripts/run_inference.py search --query test_binaries/test_gnn_gcc_O0 --database embeddings/ --top-k 10

Module: scripts.run_inference
Owner: User Story 8 (US8) - Vectorization & Similarity Search
Tasks: T097
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.vectorizer import Vectorizer
from inference.similarity import (
    build_embedding_database,
    search_similar_functions
)

logger = logging.getLogger("bcsd.scripts")


def vectorize_command(args):
    """Vectorize binaries in a directory."""
    logger.info("="*60)
    logger.info("BCSD Vectorization")
    logger.info("="*60)
    
    # Create vectorizer
    vectorizer = Vectorizer(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device
    )
    
    # Vectorize directory
    results = vectorizer.vectorize_directory(
        directory_path=args.input,
        output_dir=args.output,
        pattern=args.pattern
    )
    
    logger.info(f"\n✓ Vectorization complete!")
    logger.info(f"Generated {len(results)} embeddings")
    logger.info(f"Saved to: {args.output}")


def search_command(args):
    """Search for similar binaries."""
    logger.info("="*60)
    logger.info("BCSD Similarity Search")
    logger.info("="*60)
    
    # Build database
    logger.info(f"\nLoading embedding database from {args.database}...")
    database = build_embedding_database(args.database)
    
    if not database:
        logger.error("No embeddings found in database!")
        return
    
    # Vectorize query binary
    logger.info(f"\nVectorizing query: {args.query}...")
    vectorizer = Vectorizer(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device
    )
    
    query_embedding = vectorizer.vectorize_binary(args.query)
    
    if query_embedding is None:
        logger.error("Failed to vectorize query binary!")
        return
    
    # Search for similar functions
    logger.info(f"\nSearching for top-{args.top_k} similar functions...")
    query_name = Path(args.query).name
    
    results = search_similar_functions(
        query_binary=query_name,
        database=database,
        query_embedding=query_embedding,
        k=args.top_k
    )
    
    # Display results
    print("\n" + "="*80)
    print(f"Query: {query_name}")
    print("="*80)
    print(f"{'Rank':<6} {'Binary':<50} {'Similarity':<12} {'Notes'}")
    print("-"*80)
    
    for result in results:
        rank = result["rank"]
        binary = result["binary"]
        similarity = result["similarity"]
        notes = "← QUERY" if result["is_query"] else ""
        
        print(f"{rank:<6} {binary:<50} {similarity:>10.4f}  {notes}")
    
    print("="*80)
    logger.info(f"\n✓ Search complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BCSD Inference: Vectorization and Similarity Search"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Vectorize command
    vectorize_parser = subparsers.add_parser(
        "vectorize",
        help="Vectorize binaries in a directory"
    )
    vectorize_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing binaries"
    )
    vectorize_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for embeddings"
    )
    vectorize_parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_best.pt",
        help="Path to model checkpoint"
    )
    vectorize_parser.add_argument(
        "--vocab",
        type=str,
        default="data/vocab.json",
        help="Path to tokenizer vocabulary"
    )
    vectorize_parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern for binary files"
    )
    vectorize_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search for similar binaries"
    )
    search_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query binary to search for"
    )
    search_parser.add_argument(
        "--database",
        type=str,
        required=True,
        help="Directory containing embedding database (.npy files)"
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to return"
    )
    search_parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_best.pt",
        help="Path to model checkpoint"
    )
    search_parser.add_argument(
        "--vocab",
        type=str,
        default="data/vocab.json",
        help="Path to tokenizer vocabulary"
    )
    search_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run command
    if args.command == "vectorize":
        vectorize_command(args)
    elif args.command == "search":
        search_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
