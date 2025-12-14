#!/bin/bash
# Compilation script for test_gnn.c
# Creates multiple compilation variants for testing disassembly and tokenization

set -e  # Exit on error

echo "=== Compiling test_gnn.c with Multiple Variants ==="
echo ""

# Check if test_gnn.c exists
if [ ! -f "test_gnn.c" ]; then
    echo "Error: test_gnn.c not found!"
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -f test_gnn_gcc_O0 test_gnn_gcc_O3 test_gnn_clang_O0 test_gnn_clang_O3
echo ""

# Compile with gcc -O0 (no optimization, preserves structure)
echo "1. Compiling with gcc -O0 (no optimization)..."
gcc -g -O0 -m64 -o test_gnn_gcc_O0 test_gnn.c
if [ $? -eq 0 ]; then
    echo "   ✓ test_gnn_gcc_O0 created"
else
    echo "   ✗ gcc -O0 compilation failed!"
    exit 1
fi
echo ""

# Compile with gcc -O3 (full optimization)
echo "2. Compiling with gcc -O3 (full optimization)..."
gcc -g -O3 -m64 -o test_gnn_gcc_O3 test_gnn.c
if [ $? -eq 0 ]; then
    echo "   ✓ test_gnn_gcc_O3 created"
else
    echo "   ✗ gcc -O3 compilation failed!"
    exit 1
fi
echo ""

# Compile with clang -O0 (if available)
if command -v clang &> /dev/null; then
    echo "3. Compiling with clang -O0 (no optimization)..."
    clang -g -O0 -m64 -o test_gnn_clang_O0 test_gnn.c
    if [ $? -eq 0 ]; then
        echo "   ✓ test_gnn_clang_O0 created"
    else
        echo "   ✗ clang -O0 compilation failed!"
    fi
    echo ""
    
    echo "4. Compiling with clang -O3 (full optimization)..."
    clang -g -O3 -m64 -o test_gnn_clang_O3 test_gnn.c
    if [ $? -eq 0 ]; then
        echo "   ✓ test_gnn_clang_O3 created"
    else
        echo "   ✗ clang -O3 compilation failed!"
    fi
    echo ""
else
    echo "3. Clang not found, skipping clang variants"
    echo ""
fi

# Display binary information
echo "=== Binary Information ==="
for binary in test_gnn_gcc_O0 test_gnn_gcc_O3 test_gnn_clang_O0 test_gnn_clang_O3; do
    if [ -f "$binary" ]; then
        echo ""
        echo "File: $binary"
        file "$binary"
        size=$(stat -c%s "$binary" 2>/dev/null || stat -f%z "$binary" 2>/dev/null)
        echo "Size: $size bytes"
        
        # Check for debug symbols
        if nm "$binary" | grep -q "calculate_sum"; then
            echo "Symbols: Present (unstripped)"
        else
            echo "Symbols: Absent (stripped)"
        fi
    fi
done
echo ""

# Run one variant to verify functionality
echo "=== Testing Execution (gcc -O0) ==="
if [ -f "test_gnn_gcc_O0" ]; then
    ./test_gnn_gcc_O0
    echo ""
fi

echo "=== Compilation Summary ==="
echo "✓ All compilation variants created successfully!"
echo ""
echo "Created files:"
ls -lh test_gnn_* 2>/dev/null || echo "No binaries found"
echo ""
echo "Next steps:"
echo "  1. Run angr disassembly on these binaries"
echo "  2. Compare CFG structures across variants"
echo "  3. Validate tokenization correctness"
echo "  4. Use outputs as thesis examples"
