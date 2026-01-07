#!/bin/bash
# Create test database for PAC microbenchmarks
# Generates views data1, data10, data100 with 10M, 100M, 1B rows respectively.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARIES_DIR="$SCRIPT_DIR/binaries"
DB_FILE="$BINARIES_DIR/test_data.duckdb"

# Use any available binary to create the database
get_duckdb_binary() {
    for b in "$BINARIES_DIR"/duckdb_*; do
        if [[ -f "$b" ]]; then
            echo "$b"
            return
        fi
    done
    echo "../../build/release/duckdb"
}

DUCKDB=$(get_duckdb_binary)

echo "========================================"
echo "Creating PAC Microbenchmark Test Database"
echo "========================================"
echo "Binary: $DUCKDB"
echo "Output: $DB_FILE"
echo ""

# Remove existing database
rm -f "$DB_FILE"

# Create base view and scaled views
echo "Creating views: data (base), data1 (10M), data10 (100M), data100 (1B)..."

"$DUCKDB" "$DB_FILE" -c "
    -- Base view with 10M rows
    CREATE VIEW data AS
    SELECT
        -- Row identifier
        i,

        -- Data columns with various value ranges (typed) - use hash for non-monotonic
        ((hash(i) % 256)::INTEGER - 128)::TINYINT as col_8,
        ((hash(i + 1) % 65536)::INTEGER - 32768)::SMALLINT as col_16,
        ((hash(i + 2) % 4294967296)::BIGINT - 2147483648)::INTEGER as col_32,
        (hash(i + 3) & 9223372036854775807)::BIGINT as col_64,
        (i % 10000000 * 1000000::HUGEINT) as col_128,
        ((i % 1000) + 0.5)::FLOAT as col_flt,
        ((i % 1000000) + 0.123456789)::DOUBLE as col_dbl,

        -- Random value columns with different domains (all BIGINT)
        (hash(i + 1) % 2)::BIGINT as tiny_64,
        (abs(hash(i + 2)) % 1000)::BIGINT as small_64,
        (abs(hash(i + 3)) % 1000000)::BIGINT as medium_64,
        (abs(hash(i + 4)) % 1000000000)::BIGINT as large_64,

        -- Distribution test columns (UBIGINT, domain 0-255)
        (hash(i) & 255)::UBIGINT as rand_8,
        ((i >> 16) & 255)::UBIGINT as inc_8,
        (255 - ((i >> 16) & 255))::UBIGINT as dec_8

    FROM range(10000000) t(i);

    -- data1: 10M rows (scale=1)
    -- grp_X: sequential groups (i // divisor, integer division)
    -- prg_X: scattered groups (i % X)
    CREATE VIEW data1 AS
    SELECT
        *,
        (i // 1000000)::INTEGER as grp_10,
        (i // 10000)::INTEGER as grp_1000,
        (i // 100)::INTEGER as grp_100000,
        i::INTEGER as grp_10000000,
        (i % 10)::INTEGER as prg_10,
        (i % 1000)::INTEGER as prg_1000,
        (i % 100000)::INTEGER as prg_100000,
        (i % 10000000)::INTEGER as prg_10000000
    FROM data;

    -- data10: 100M rows (scale=10)
    -- Division factors increased by 10x to maintain same group counts
    CREATE VIEW data10 AS
    SELECT
        i,
        (i % 4)::TINYINT as col_8,
        (i % 1024)::SMALLINT as col_16,
        (i % 262144)::INTEGER as col_32,
        (i % 1099511627776)::BIGINT as col_64,
        (i % 10000000 * 1000000::HUGEINT) as col_128,
        ((i % 1000) + 0.5)::FLOAT as col_flt,
        ((i % 1000000) + 0.123456789)::DOUBLE as col_dbl,
        (hash(i + 1) % 2)::BIGINT as tiny_64,
        (abs(hash(i + 2)) % 1000)::BIGINT as small_64,
        (abs(hash(i + 3)) % 1000000)::BIGINT as medium_64,
        (abs(hash(i + 4)) % 1000000000)::BIGINT as large_64,
        (hash(i) & 255)::UBIGINT as rand_8,
        ((i >> 19) & 255)::UBIGINT as inc_8,
        (255 - ((i >> 19) & 255))::UBIGINT as dec_8,
        (i // 10000000)::INTEGER as grp_10,
        (i // 100000)::INTEGER as grp_1000,
        (i // 1000)::INTEGER as grp_100000,
        (i // 10)::INTEGER as grp_10000000,
        (i % 10)::INTEGER as prg_10,
        (i % 1000)::INTEGER as prg_1000,
        (i % 100000)::INTEGER as prg_100000,
        (i % 10000000)::INTEGER as prg_10000000
    FROM range(100000000) t(i);

    -- data100: 1B rows (scale=100)
    -- Division factors increased by 100x to maintain same group counts
    CREATE VIEW data100 AS
    SELECT
        i,
        (i % 4)::TINYINT as col_8,
        (i % 1024)::SMALLINT as col_16,
        (i % 262144)::INTEGER as col_32,
        (i % 1099511627776)::BIGINT as col_64,
        (i % 10000000 * 1000000::HUGEINT) as col_128,
        ((i % 1000) + 0.5)::FLOAT as col_flt,
        ((i % 1000000) + 0.123456789)::DOUBLE as col_dbl,
        (hash(i + 1) % 2)::BIGINT as tiny_64,
        (abs(hash(i + 2)) % 1000)::BIGINT as small_64,
        (abs(hash(i + 3)) % 1000000)::BIGINT as medium_64,
        (abs(hash(i + 4)) % 1000000000)::BIGINT as large_64,
        (hash(i) & 255)::UBIGINT as rand_8,
        ((i >> 22) & 255)::UBIGINT as inc_8,
        (255 - ((i >> 22) & 255))::UBIGINT as dec_8,
        (i // 100000000)::INTEGER as grp_10,
        (i // 1000000)::INTEGER as grp_1000,
        (i // 10000)::INTEGER as grp_100000,
        (i // 100)::INTEGER as grp_10000000,
        (i % 10)::INTEGER as prg_10,
        (i % 1000)::INTEGER as prg_1000,
        (i % 100000)::INTEGER as prg_100000,
        (i % 10000000)::INTEGER as prg_10000000
    FROM range(1000000000) t(i);
"

echo ""
echo "Verifying views..."

"$DUCKDB" "$DB_FILE" -c "
    SELECT 'data1' as view, COUNT(*) as rows FROM data1
    UNION ALL
    SELECT 'data10', COUNT(*) FROM data10
    UNION ALL
    SELECT 'data100', COUNT(*) FROM data100;
"

echo ""
echo "Column summary (data1):"
"$DUCKDB" "$DB_FILE" -c "DESCRIBE data1;"

echo ""
echo "Sample values (showing grp vs prg difference in data1):"
"$DUCKDB" "$DB_FILE" -c "SELECT i, grp_10, prg_10, grp_1000, prg_1000 FROM data1 WHERE i < 15;"

echo ""
echo "========================================"
echo "Database created successfully: $DB_FILE"
echo "========================================"
echo ""
echo "Views:"
echo "  data1:   10M rows  (scale 1x)"
echo "  data10:  100M rows (scale 10x)"
echo "  data100: 1B rows   (scale 100x)"
echo ""
echo "Grouping columns (same group counts across all scales):"
echo "  grp_X: sequential groups - consecutive rows in same group"
echo "  prg_X: scattered groups - rows cycle through all groups"
echo ""
echo "Usage in benchmarks:"
echo "  SELECT pac_sum(hash(i), col_64, 0.0) FROM data10;"
echo ""

ls -lh "$DB_FILE"
