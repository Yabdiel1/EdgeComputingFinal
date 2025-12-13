#!/bin/bash

echo "Parallel FFT Test Runner"
echo "========================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo

# Check if src/tests/test_parallel.py exists
if [ -f "src/test_parallel.py" ]; then
    echo "Found test_parallel.py"
else
    echo "ERROR: test_parallel.py not found!"
    echo "Looking in: $(pwd)/src/"
    ls -la src/tests/ 2>/dev/null || echo "src/ directory doesn't exist"
    exit 1
fi

# Run the tests
echo
echo "Running tests with 4 processes..."
echo "================================="
mpiexec -n 4 python -m src.test_parallel

# Check exit status
if [ $? -eq 0 ]; then
    echo
    echo "Tests completed successfully!"
else
    echo
    echo "Tests failed!"
fi