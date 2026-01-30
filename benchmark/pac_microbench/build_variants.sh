#!/bin/bash
# Build PAC DuckDB variants with different optimization flags
# Each variant is built to a separate binary for benchmarking
#
# Consolidated binary names (aggregate-agnostic):
#   default       - default cascading/banking/buffering (all optimizations on, approx sum)
#   nobuffering   - disable input buffering (lazy allocation)
#   noboundopt    - disable bound optimization (only affects min/max)
#   signedsum     - disable handling negative values in signed sums using separate (negated) counters
#   exactsum      - disable approximate sum optimization, use exact cascading (implies signedsum)
#   nocascading   - disable cascading (count, sum/avg -- implies exactsum)
#   nosimd        - nocascading with simd-unfriendly update kernels and auto-vectorization disabled

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PAC_ROOT/build/bench"
BINARIES_DIR="$SCRIPT_DIR/binaries"

mkdir -p "$BUILD_DIR"
mkdir -p "$BINARIES_DIR"

# Build configurations (bash 3 compatible - no associative arrays)
# Format: "name:flags" (empty flags = default build)
CONFIGS=(
    "default:"
    "noboundopt:-DPAC_NOBOUNDOPT"
    "nobuffering:-DPAC_NOBUFFERING"
    "nocascading:-DPAC_NOCASCADING"
    "nosimd:-DPAC_NOCASCADING -DPAC_NOSIMD -fno-vectorize -fno-slp-vectorize"
    "signedsum:-DPAC_SIGNEDSUM"
    "exactsum:-DPAC_EXACTSUM"
)

# Get config name from "name:flags" string
get_name() { echo "${1%%:*}"; }
# Get config flags from "name:flags" string
get_flags() { echo "${1#*:}"; }

# Find config by name, returns the full "name:flags" string or empty
find_config() {
    local search=$1
    for cfg in "${CONFIGS[@]}"; do
        if [[ "$(get_name "$cfg")" == "$search" ]]; then
            echo "$cfg"
            return
        fi
    done
}

# List all config names
list_names() {
    for cfg in "${CONFIGS[@]}"; do
        get_name "$cfg"
    done
}

# Track which flag combinations we've already built (to avoid redundant builds)
BUILT_FLAGS=""

build_variant() {
    local name=$1
    local flags=$2
    local binary="$BINARIES_DIR/duckdb_$name"

    if [[ -f "$binary" && "$FORCE_REBUILD" != "1" ]]; then
        echo "Skipping $name (already exists, use FORCE_REBUILD=1 to rebuild)"
        return
    fi

    # Check if we already built a binary with these exact flags
    # If so, just copy/link instead of rebuilding
    for built in $BUILT_FLAGS; do
        local built_name="${built%%=*}"
        local built_flags="${built#*=}"
        if [[ "$built_flags" == "$flags" ]]; then
            local source_binary="$BINARIES_DIR/duckdb_$built_name"
            if [[ -f "$source_binary" ]]; then
                echo "Copying $built_name -> $name (same flags: '$flags')"
                cp "$source_binary" "$binary"
                return
            fi
        fi
    done

    echo "========================================"
    echo "Building variant: $name"
    echo "CXXFLAGS: $flags"
    echo "========================================"

    cd "$PAC_ROOT"

    # Clean and build
    rm -rf build/release
    CXX=../llvm-project/build/bin/clang++ CXXFLAGS="$flags" GEN=ninja make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

    # Copy binary
    cp build/release/duckdb "$binary"
    echo "Built: $binary"

    # Track this build
    BUILT_FLAGS="$BUILT_FLAGS $name=$flags"
}

# Parse arguments
VARIANTS_TO_BUILD=()
if [[ $# -eq 0 ]]; then
    # Build all variants
    for cfg in "${CONFIGS[@]}"; do
        VARIANTS_TO_BUILD+=("$cfg")
    done
else
    # Build specified variants
    for arg in "$@"; do
        cfg=$(find_config "$arg")
        if [[ -n "$cfg" ]]; then
            VARIANTS_TO_BUILD+=("$cfg")
        else
            echo "Unknown variant: $arg"
            echo "Available variants: $(list_names | tr '\n' ' ')"
            exit 1
        fi
    done
fi

echo "Building variants: $(for v in "${VARIANTS_TO_BUILD[@]}"; do get_name "$v"; done | tr '\n' ' ')"
echo ""

for cfg in "${VARIANTS_TO_BUILD[@]}"; do
    build_variant "$(get_name "$cfg")" "$(get_flags "$cfg")"
done

echo ""
echo "========================================"
echo "Build complete. Binaries in: $BINARIES_DIR"
echo "========================================"
ls -la "$BINARIES_DIR"
