#!/bin/bash

# Script to run PAC tests 1000 times and count failures
# Usage: ./run_tests_1000.sh

TEST_DIR="$(pwd)"
TEST_BINARY="$TEST_DIR/build/release/test/unittest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_RUNS=1000
FAILURES=0
SUCCESSES=0

# Create a log directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$TEST_DIR/test_logs_$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "PAC Test Runner - 1000 Iterations"
echo "=========================================="
echo "Test binary: $TEST_BINARY"
echo "Log directory: $LOG_DIR"
echo "=========================================="
echo ""

# Check if test binary exists
if [ ! -f "$TEST_BINARY" ]; then
    echo -e "${RED}ERROR: Test binary not found at $TEST_BINARY${NC}"
    echo "Please build the project first."
    exit 1
fi

# Run tests 1000 times
for i in $(seq 1 $TOTAL_RUNS); do
    # Progress indicator
    if [ $((i % 10)) -eq 0 ]; then
        echo -ne "\rProgress: $i/$TOTAL_RUNS (Failures: $FAILURES, Successes: $SUCCESSES)"
    fi

    # Run the test and capture output
    OUTPUT_FILE="$LOG_DIR/run_$i.log"
    cd "$TEST_DIR/build/release" || exit 1

    if "$TEST_BINARY" > "$OUTPUT_FILE" 2>&1; then
        SUCCESSES=$((SUCCESSES + 1))
        # Remove successful run logs to save space (optional)
        rm "$OUTPUT_FILE"
    else
        FAILURES=$((FAILURES + 1))
        # Keep the failure log
        echo "Run $i: FAILED" >> "$LOG_DIR/failure_summary.txt"

        # Extract the failure reason
        grep -A 10 "FAIL\|Error\|Mismatch\|Assertion" "$OUTPUT_FILE" | head -20 >> "$LOG_DIR/failure_summary.txt"
        echo "----------------------------------------" >> "$LOG_DIR/failure_summary.txt"
    fi
done

echo -ne "\r"
echo "=========================================="
echo "Test Run Complete!"
echo "=========================================="
echo -e "Total runs:    ${YELLOW}$TOTAL_RUNS${NC}"
echo -e "Successes:     ${GREEN}$SUCCESSES${NC}"
echo -e "Failures:      ${RED}$FAILURES${NC}"

if [ $FAILURES -gt 0 ]; then
    FAILURE_RATE=$(awk "BEGIN {printf \"%.2f\", ($FAILURES / $TOTAL_RUNS) * 100}")
    echo -e "Failure rate:  ${RED}${FAILURE_RATE}%${NC}"
    echo ""
    echo "Failure logs saved in: $LOG_DIR"
    echo "Summary: $LOG_DIR/failure_summary.txt"

    # Show a sample of failures
    echo ""
    echo "=========================================="
    echo "Sample Failures (first 50 lines):"
    echo "=========================================="
    head -50 "$LOG_DIR/failure_summary.txt"
else
    echo -e "${GREEN}All tests passed!${NC}"
    # Clean up log directory if all tests passed
    rmdir "$LOG_DIR" 2>/dev/null
fi

echo "=========================================="

# Exit with failure code if any tests failed
if [ $FAILURES -gt 0 ]; then
    exit 1
else
    exit 0
fi
