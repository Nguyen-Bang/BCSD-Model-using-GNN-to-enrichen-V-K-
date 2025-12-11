#!/bin/bash
# Compilation script for test binaries

echo "=== Compiling test_gnn.c ==="

# Compile with debug symbols and no optimization
gcc -g -O0 -m64 -o test_binary test_gnn.c

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo ""
    
    # Verify the binary
    echo "=== Binary Information ==="
    file test_binary
    echo ""
    
    # Check if stripped
    echo "=== Symbol Information ==="
    nm test_binary | grep -E "(calculate_sum|check_value|main)" || echo "No symbols found (binary is stripped!)"
    echo ""
    
    # Run the binary
    echo "=== Running Binary ==="
    ./test_binary
    echo ""
    echo "✓ Test completed successfully!"
else
    echo "✗ Compilation failed!"
    exit 1
fi
