#!/usr/bin/env python3
"""
Test script for validating angr disassembly on test_gnn binaries.

This script:
1. Compiles test_gnn.c if binaries don't exist
2. Extracts CFG from all variants
3. Validates expected patterns (loops, branches)
4. Generates comparison report

Usage:
    python test_binaries/test_disassembly.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.angr_disassembly import extract_function_data


def compile_test_binaries():
    """Compile test_gnn.c if binaries don't exist."""
    test_dir = Path(__file__).parent
    compile_script = test_dir / "compile.sh"
    
    # Check if any binary exists
    variants = ["test_gnn_gcc_O0", "test_gnn_gcc_O3", "test_gnn_clang_O0", "test_gnn_clang_O3"]
    binaries_exist = any((test_dir / v).exists() for v in variants)
    
    if not binaries_exist:
        print("Compiling test binaries...")
        try:
            subprocess.run(["bash", str(compile_script)], cwd=test_dir, check=True)
            print("✓ Compilation successful\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ Compilation failed: {e}")
            return False
    else:
        print("Test binaries already exist\n")
    
    return True


def validate_function_patterns(func_data, func_name):
    """Validate expected CFG patterns for each function."""
    errors = []
    
    if func_name == "calculate_sum":
        # Should have backward edge (loop)
        has_backward_edge = False
        for src, dst in func_data["edges"]:
            if dst < src:  # Backward edge (cycle)
                has_backward_edge = True
                break
        
        if not has_backward_edge:
            errors.append(f"  ✗ {func_name}: No backward edge found (loop missing)")
        else:
            print(f"  ✓ {func_name}: Backward edge detected (loop present)")
    
    elif func_name == "check_value":
        # Should have branches (multiple outgoing edges from some node)
        has_branches = False
        node_outgoing = {}
        for src, dst in func_data["edges"]:
            node_outgoing[src] = node_outgoing.get(src, 0) + 1
        
        max_outgoing = max(node_outgoing.values()) if node_outgoing else 0
        if max_outgoing >= 2:
            has_branches = True
        
        if not has_branches:
            errors.append(f"  ✗ {func_name}: No branches found (if/else missing)")
        else:
            print(f"  ✓ {func_name}: Branches detected ({max_outgoing} outgoing edges from conditional)")
    
    elif func_name == "main":
        # Should have multiple blocks
        if len(func_data["blocks"]) < 3:
            errors.append(f"  ✗ {func_name}: Too few basic blocks ({len(func_data['blocks'])})")
        else:
            print(f"  ✓ {func_name}: {len(func_data['blocks'])} basic blocks")
    
    return errors


def test_single_binary(binary_path):
    """Test disassembly on a single binary."""
    print(f"\n{'='*60}")
    print(f"Testing: {binary_path.name}")
    print(f"{'='*60}")
    
    # Extract functions
    try:
        functions = extract_function_data(str(binary_path))
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False
    
    if not functions:
        print("✗ No functions extracted")
        return False
    
    print(f"✓ Extracted {len(functions)} functions")
    
    # Save output
    output_path = binary_path.parent / f"{binary_path.name}_output.json"
    with open(output_path, 'w') as f:
        json.dump(functions, f, indent=2)
    print(f"✓ Saved output to: {output_path}")
    
    # Validate each function
    print("\nValidating function patterns:")
    errors = []
    expected_functions = {"calculate_sum", "check_value", "main"}
    found_functions = {f["function_name"] for f in functions}
    
    # Check all expected functions exist
    missing = expected_functions - found_functions
    if missing:
        print(f"✗ Missing functions: {missing}")
        errors.append(f"Missing functions: {missing}")
    
    # Validate patterns
    for func in functions:
        func_name = func["function_name"]
        if func_name in expected_functions:
            func_errors = validate_function_patterns(func, func_name)
            errors.extend(func_errors)
    
    # Print summary
    print(f"\nSummary for {binary_path.name}:")
    print(f"  Functions: {len(functions)}")
    print(f"  Total blocks: {sum(len(f['blocks']) for f in functions)}")
    print(f"  Total edges: {sum(len(f['edges']) for f in functions)}")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(error)
        return False
    else:
        print("\n✓ All validations passed")
        return True


def compare_variants():
    """Compare CFG complexity across compilation variants."""
    test_dir = Path(__file__).parent
    variants = ["test_gnn_gcc_O0", "test_gnn_gcc_O3", "test_gnn_clang_O0", "test_gnn_clang_O3"]
    
    print(f"\n{'='*60}")
    print("Compilation Variant Comparison")
    print(f"{'='*60}")
    
    comparison_data = []
    
    for variant_name in variants:
        variant_path = test_dir / variant_name
        output_path = test_dir / f"{variant_name}_output.json"
        
        if not output_path.exists():
            print(f"⚠ Skipping {variant_name}: output not found")
            continue
        
        with open(output_path) as f:
            functions = json.load(f)
        
        for func in functions:
            comparison_data.append({
                "variant": variant_name,
                "function": func["function_name"],
                "blocks": len(func["blocks"]),
                "edges": len(func["edges"])
            })
    
    # Print comparison table
    print(f"\n{'Variant':<25} {'Function':<20} {'Blocks':<10} {'Edges':<10}")
    print("-" * 65)
    
    for data in comparison_data:
        print(f"{data['variant']:<25} {data['function']:<20} {data['blocks']:<10} {data['edges']:<10}")
    
    print("\nObservations:")
    print("  - O3 variants typically have fewer blocks (optimization)")
    print("  - gcc and clang may produce different CFG structures")
    print("  - Loop backward edges should be present in all variants")


def main():
    """Main test execution."""
    print("="*60)
    print("Test Binary Disassembly Validation")
    print("="*60)
    
    # Compile binaries if needed
    if not compile_test_binaries():
        print("\n✗ Test failed: Could not compile binaries")
        return 1
    
    # Test each binary
    test_dir = Path(__file__).parent
    variants = ["test_gnn_gcc_O0", "test_gnn_gcc_O3", "test_gnn_clang_O0", "test_gnn_clang_O3"]
    
    results = {}
    for variant_name in variants:
        variant_path = test_dir / variant_name
        if variant_path.exists():
            results[variant_name] = test_single_binary(variant_path)
        else:
            print(f"\n⚠ Skipping {variant_name}: binary not found")
            results[variant_name] = None
    
    # Compare variants
    compare_variants()
    
    # Final summary
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed > 0:
        print("\n✗ Some tests failed")
        return 1
    elif passed == 0:
        print("\n⚠ No tests were run")
        return 1
    else:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("  1. Review generated *_output.json files")
        print("  2. Use outputs as examples in thesis")
        print("  3. Process full dataset binaries")
        return 0


if __name__ == "__main__":
    sys.exit(main())
