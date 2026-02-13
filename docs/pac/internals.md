# PAC Internals

This document describes the internal implementation of the PAC (Privacy-Aware-Computed) aggregation algorithm.

## Overview

PAC transforms standard SQL aggregates (SUM, COUNT, AVG, MIN, MAX) into privacy-preserving versions that use **bit-level probabilistic counting**. Instead of traditional sampling where rows are included or excluded, PAC maintains 64 parallel counters and uses hash bits to determine which counters receive each value.

## Core Algorithm

### Hash-Based Counter Selection

For each row in the query:

1. **Compute hash**: The privacy unit's key (PAC_KEY, PRIMARY KEY, or rowid) is hashed to produce a 64-bit value
2. **Distribute to counters**: For each of the 64 counters, if bit `j` of the hash is 1, the value is added to counter `j`

```cpp
for (int j = 0; j < 64; j++) {
    if ((key_hash >> j) & 1) {
        counter[j] += value;
    }
}
```

On average, each row contributes to ~32 counters (half the bits are 1).

### Counter Selection at Finalize

The `pac_mi` setting (0-63) determines which counter is used for the final result:

```cpp
result = counter[pac_mi];
```

### 2x Multiplier Compensation

Since each counter only receives ~50% of values on average, the result is multiplied by 2:

```cpp
result *= 2.0;
```

This compensates for the probabilistic sampling.

## PAC Aggregate Functions

### pac_sum

Transforms `SUM(value)` into `pac_sum(hash_expr, value)`.

**Implementation details:**

- **Cascaded accumulators**: Uses multiple precision levels (8-bit, 16-bit, 32-bit, 64-bit) for efficiency
- **SWAR optimization**: Uses SIMD-Within-A-Register to pack multiple counters per 64-bit word
- **Signed value handling**: Can use double-sided mode (separate positive/negative states) or signed mode

```sql
-- Original
SELECT SUM(amount) FROM orders;

-- Transformed
SELECT pac_sum(hash(customer_id), amount) FROM orders;
```

### pac_count

Transforms `COUNT(*)` or `COUNT(expr)` into `pac_count(hash_expr, 1)`.

**Implementation details:**

- Uses the same cascaded accumulator structure as pac_sum
- COUNT(*) passes 1 as the value argument
- COUNT(expr) passes 1 for non-NULL values

```sql
-- Original
SELECT COUNT(*) FROM orders;

-- Transformed  
SELECT pac_count(hash(customer_id), 1) FROM orders;
```

### pac_avg

Transforms `AVG(value)` into `pac_avg(hash_expr, value)`.

**Implementation details:**

- Internally computes `pac_sum(hash, value) / count`
- Tracks exact count alongside probabilistic sum
- Applies 2x compensation after division

```sql
-- Original
SELECT AVG(amount) FROM orders;

-- Transformed
SELECT pac_avg(hash(customer_id), amount) FROM orders;
```

### pac_min / pac_max

Transforms `MIN(value)` and `MAX(value)` into their PAC equivalents.

**Implementation details:**

- **Bound optimization**: Tracks current min/max bounds to skip unnecessary comparisons
- **Banked storage**: Uses efficient memory layout for 64 parallel min/max values
- **Lazy allocation**: Only allocates memory for counters that receive values

```sql
-- Original
SELECT MIN(price), MAX(price) FROM products;

-- Transformed
SELECT pac_min(hash(product_id), price), pac_max(hash(product_id), price) FROM products;
```

## Hash Input Generation

The hash input is generated based on the privacy unit's identifier:

### Priority Order

1. **PAC_KEY**: If defined via `CREATE PAC TABLE` or `ALTER PAC TABLE ADD PAC_KEY`
2. **PRIMARY KEY**: Database-level primary key constraint
3. **rowid**: DuckDB's internal row identifier (fallback)

### Composite Keys

For composite keys (multiple columns), values are XOR'd before hashing:

```cpp
// Single column
hash_expr = hash(pk_column);

// Composite key
hash_expr = hash(xor(pk_col1, xor(pk_col2, pk_col3)));
```

## Query Transformation

### Direct Privacy Unit Scan

When the query directly scans a privacy unit table:

```sql
-- Original
SELECT SUM(balance) FROM customers;

-- Transformation
-- 1. Build hash from customers.id (PAC_KEY)
-- 2. Replace SUM with pac_sum
SELECT pac_sum(hash(id), balance) FROM customers;
```

### Foreign Key Path

When the query scans a table linked to a privacy unit via PAC_LINK:

```sql
-- Original  
SELECT SUM(amount) FROM orders;
-- orders has PAC_LINK(customer_id) REFERENCES customers(id)

-- Transformation
-- 1. Follow FK path: orders -> customers
-- 2. Build hash from orders.customer_id (the FK column)
SELECT pac_sum(hash(customer_id), amount) FROM orders;
```

### Join Without PAC_LINK

When a privacy unit is joined with a table that has no PAC_LINK:

```sql
-- Original
SELECT SUM(s.amount) FROM customers c JOIN sales s ON c.id = s.customer_id;

-- Transformation
-- Uses the PU's PAC_KEY for hashing
SELECT pac_sum(hash(c.id), s.amount) FROM customers c JOIN sales s ON c.id = s.customer_id;
```

## Configuration Parameters

| Parameter | Effect on Algorithm |
|-----------|---------------------|
| `pac_mi` | Selects which counter (0-63) to use for result |
| `pac_seed` | Seeds the RNG for deterministic noise |
| `pac_noise` | Enables/disables noise addition |
| `pac_deterministic_noise` | Uses deterministic (reproducible) noise |

## Build-Time Optimization Flags

These compile-time flags control algorithm variants:

| Flag | Effect |
|------|--------|
| `PAC_COUNT_NONBANKED` | Disable banked counting optimization |
| `PAC_SUMAVG_NONCASCADING` | Disable cascaded accumulators |
| `PAC_SUMAVG_NONLAZY` | Pre-allocate all accumulator levels |
| `PAC_MINMAX_NONBANKED` | Disable banked min/max |
| `PAC_MINMAX_NOBOUNDOPT` | Disable bound optimization |
| `PAC_MINMAX_NONLAZY` | Pre-allocate all min/max levels |

## Source Code Organization

| File | Description |
|------|-------------|
| `src/aggregates/pac_sum.cpp` | pac_sum and pac_avg implementation |
| `src/aggregates/pac_count.cpp` | pac_count implementation |
| `src/aggregates/pac_min_max.cpp` | pac_min and pac_max implementation |
| `src/aggregates/pac_aggregate_hash.cpp` | Hash expression building |
| `src/compiler/pac_bitslice_compiler.cpp` | Query transformation logic |
| `src/query_processing/pac_expression_builder.cpp` | Expression transformation |

