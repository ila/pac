# PAC — DuckDB Privacy-Aware Aggregation Extension

PAC (Privacy-Aware-Computed) is a DuckDB extension that enforces privacy rules on aggregation queries over designated tables ("PAC tables"). It provides differential privacy guarantees by transforming standard SQL aggregates into privacy-preserving versions.

## What PAC Does

PAC enables privacy-preserving analytics on sensitive data by automatically transforming SQL queries. When you designate a table as a **privacy unit** (using `CREATE PAC TABLE` or `ALTER TABLE SET PAC`), PAC ensures that individual records cannot be directly accessed or leaked through query results.

Instead of returning exact values, PAC transforms aggregates like `SUM`, `COUNT`, `AVG`, `MIN`, and `MAX` into probabilistic versions that provide strong privacy guarantees while maintaining statistical accuracy. The extension uses a bit-level probabilistic counting algorithm that distributes each record's contribution across 64 parallel counters based on a hash of the privacy unit's key, then samples from these counters with appropriate noise to produce privacy-preserving results.

PAC enforces access controls based on the `PROTECTED` clause:
- **Privacy unit with PROTECTED columns**: Only the protected columns are restricted to aggregate access; non-protected columns can be projected normally
- **Privacy unit without PROTECTED columns**: ALL columns are treated as protected and can only be accessed inside aggregate functions

For tables that are not privacy units, you can still mark specific columns as `PROTECTED` to restrict their access to aggregate functions only.

## Documentation

Additional documentation is available in the `docs/` folder:

| Document | Description |
|----------|-------------|
| [docs/build/README.md](docs/build/README.md) | Build instructions (Make, CMake, Ninja) |
| [docs/build/clion.md](docs/build/clion.md) | CLion IDE setup and debugging |
| [docs/build/updating.md](docs/build/updating.md) | How to update DuckDB version |
| [docs/pac/internals.md](docs/pac/internals.md) | PAC algorithm implementation details |
| [docs/test/README.md](docs/test/README.md) | Running SQL and C++ tests |
| [docs/benchmark/README.md](docs/benchmark/README.md) | Benchmark overview |
| [docs/benchmark/tpch.md](docs/benchmark/tpch.md) | TPC-H benchmark instructions |
| [docs/benchmark/tpch_compiler.md](docs/benchmark/tpch_compiler.md) | TPC-H compiler benchmark |
| [docs/benchmark/microbenchmarks.md](docs/benchmark/microbenchmarks.md) | Microbenchmark suite |

## PAC SQL Syntax

### CREATE PAC TABLE

Creates a table marked as a privacy unit. **Must include either `PAC_KEY` or `PRIMARY KEY`**.

```sql
CREATE PAC TABLE users (
    user_id INTEGER,
    name VARCHAR,
    email VARCHAR,
    PAC_KEY (user_id),
    PROTECTED (email)
);

-- Or with PRIMARY KEY
CREATE PAC TABLE users (
    user_id INTEGER PRIMARY KEY,
    name VARCHAR,
    email VARCHAR,
    PROTECTED (email)
);
```

**Important:** 
- If a privacy unit has **no PROTECTED columns defined**, then **all columns are treated as protected** — they can only be accessed inside aggregate functions.
- If a privacy unit has **PROTECTED columns defined**, then only those specific columns are restricted; other columns can be projected normally.

### PAC Clauses

| Clause | Description |
|--------|-------------|
| `PAC_KEY (col1, col2, ...)` | Defines the privacy unit identifier (composite keys supported) |
| `PAC_LINK (col) REFERENCES table(col)` | Links this table to a privacy unit via foreign key relationship |
| `PROTECTED (col1, col2, ...)` | Marks specific columns as sensitive (restricted to aggregate access) |

### PAC_LINK Explained

`PAC_LINK` establishes a relationship between a regular table and a privacy unit, similar to a foreign key constraint but for privacy purposes. When a table has a `PAC_LINK`:

1. **Privacy propagation**: The linked table inherits privacy constraints from the privacy unit
2. **Hash derivation**: PAC uses the link columns to derive the privacy hash (instead of requiring direct access to the PU table)
3. **Query optimization**: PAC can compile queries on the linked table without needing to join with the PU table

```sql
-- Privacy unit table
CREATE PAC TABLE customers (
    id INTEGER,
    name VARCHAR,
    PAC_KEY (id),
    PROTECTED (name)
);

-- Regular table with PAC_LINK to the privacy unit
CREATE TABLE orders (
    order_id INTEGER,
    customer_id INTEGER,
    amount DECIMAL(10,2),
    PAC_LINK (customer_id) REFERENCES customers(id)
);

-- Now queries on 'orders' automatically use customer_id for privacy hashing
SELECT SUM(amount) FROM orders;  -- Uses hash(customer_id) internally
```

Without a `PAC_LINK`, you can still join tables with privacy units, but the PU table must be present in the query for PAC to derive the privacy hash.

### ALTER PAC TABLE

Add PAC metadata to existing tables:

```sql
-- Add PAC_KEY
ALTER PAC TABLE orders ADD PAC_KEY (order_id);

-- Add PAC_LINK (foreign key relationship)
ALTER PAC TABLE orders ADD PAC_LINK (customer_id) REFERENCES customers(id);

-- Add protected columns
ALTER PAC TABLE orders ADD PROTECTED (total_amount);

-- Drop PAC metadata
ALTER PAC TABLE orders DROP PAC_LINK (customer_id);
ALTER PAC TABLE orders DROP PROTECTED (total_amount);
```

### SET PAC / UNSET PAC

Mark or unmark a table as a privacy unit:

```sql
-- Mark as privacy unit (table must have PRIMARY KEY)
ALTER TABLE customers SET PAC;

-- Remove privacy unit status
ALTER TABLE customers UNSET PAC;
```

### PAC Metadata Management

```sql
-- Save PAC metadata to file
PRAGMA save_pac_metadata('pac_metadata.json');

-- Load PAC metadata from file
PRAGMA load_pac_metadata('pac_metadata.json');

-- Clear all PAC metadata
PRAGMA clear_pac_metadata;
```

## Configuration Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `pac_enabled` | BOOLEAN | true | Enable/disable PAC query rewriting |
| `pac_seed` | BIGINT | (random) | Deterministic RNG seed for reproducible results |
| `pac_mi` | DOUBLE | 0.0 | Privacy parameter (counter index, 0-63) |
| `pac_noise` | BOOLEAN | true | Whether to add PAC noise |
| `pac_deterministic_noise` | BOOLEAN | false | Use deterministic noise (for testing) |
| `pac_conservative_mode` | BOOLEAN | true | Throw errors on unsupported queries |
| `pac_compiled_path` | VARCHAR | "." | Output path for compiled PAC SQL |
| `pac_privacy_file` | VARCHAR | "" | Path to CSV file listing PAC tables |

### Example Configuration

```sql
SET pac_seed = 42;
SET pac_mi = 0;
SET pac_deterministic_noise = true;
SET threads = 1;
```

## PAC Limitations

The PAC compiler has been tested with the TPC-H queries (SQL-92 constructs). In general, when querying PAC tables (privacy units), the following restrictions apply:

| Feature | Status | Notes                                    |
|---------|--------|------------------------------------------|
| SUM, COUNT, AVG, MIN, MAX | ✅ Supported | Transformed to pac_sum, pac_count, etc.  |
| COUNT(DISTINCT col) | ✅ Supported | Distinct counting within aggregates      |
| Other aggregates | ❌ Disallowed | Custom aggregates not supported          |
| Window functions | ❌ Disallowed | OVER clauses not supported               |
| SELECT DISTINCT | ❌ Disallowed | Use aggregates instead                   |
| INNER JOIN | ✅ Supported | Standard joins work                      |
| LEFT/RIGHT OUTER JOIN | ✅ Supported | Outer joins are supported                |
| CROSS JOIN | ✅ Supported | Cartesian products allowed               |
| Subqueries | ✅ Supported | Both correlated and uncorrelated         |
| UNION / UNION ALL | ✅ Supported | Set union operations work                |
| EXCEPT | ❌ Disallowed | Not implemented yet                      |
| INTERSECT | ❌ Disallowed | Not implemented yet            |
| ORDER BY, LIMIT | ✅ Supported | On final query only                      |
| GROUP BY | ✅ Supported | Cannot group by protected columns        |
| Projecting PROTECTED columns | ❌ Disallowed | Protected columns require aggregation    |
| Projecting non-protected columns | ✅ Allowed | If PROTECTED is defined on other columns |

## Examples

### Privacy Unit Without PROTECTED (All Columns Restricted)

```sql
-- Create a privacy unit with NO protected columns specified
-- This means ALL columns are treated as protected
CREATE PAC TABLE customers (
    id INTEGER,
    name VARCHAR,
    balance DECIMAL(10,2),
    PAC_KEY (id)
);

INSERT INTO customers VALUES (1, 'Alice', 1000), (2, 'Bob', 2000);

-- This fails: ALL columns require aggregation when no PROTECTED is defined
SELECT name FROM customers;  -- Error: requires aggregation
SELECT * FROM customers;     -- Error: requires aggregation

-- This works: aggregate query
SELECT SUM(balance) FROM customers;
```

### Privacy Unit With PROTECTED (Only Specified Columns Restricted)

```sql
-- Create a privacy unit WITH protected columns specified
-- Only the protected columns are restricted
CREATE PAC TABLE employees (
    id INTEGER,
    department VARCHAR,
    salary DECIMAL(10,2),
    PAC_KEY (id),
    PROTECTED (salary)
);

INSERT INTO employees VALUES (1, 'Engineering', 100000), (2, 'Sales', 80000);

-- This works: non-protected columns can be projected
SELECT id, department FROM employees;

-- This fails: salary is protected
SELECT salary FROM employees;  -- Error: protected column

-- This works: aggregate the protected column
SELECT AVG(salary) FROM employees;
SELECT department, AVG(salary) FROM employees GROUP BY department;
```

### Using PROTECTED on Non-PU Tables

```sql
-- Regular table (not a privacy unit) with a protected column
CREATE TABLE survey_data (
    response_id INTEGER,
    category VARCHAR,
    sensitive_answer INTEGER,
    PROTECTED (sensitive_answer)
);

-- This works: project non-protected columns
SELECT response_id, category FROM survey_data;

-- This fails: sensitive_answer is protected
SELECT sensitive_answer FROM survey_data;  -- Error: protected column

-- This works: aggregate the protected column
SELECT AVG(sensitive_answer) FROM survey_data;
```

### Join with PAC_LINK Table

```sql
-- Create related table with PAC_LINK
CREATE TABLE orders (
    order_id INTEGER,
    customer_id INTEGER,
    amount DECIMAL(10,2),
    PAC_LINK (customer_id) REFERENCES customers(id)
);

INSERT INTO orders VALUES (1, 1, 100), (2, 1, 200), (3, 2, 150);

-- Query orders directly - PAC uses customer_id for privacy hash
SELECT SUM(amount) FROM orders;

-- Join and aggregate
SELECT SUM(amount) FROM customers 
JOIN orders ON customers.id = orders.customer_id;

-- Group by non-protected column
SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id;
```

### Join Without PAC_LINK

Privacy units can be joined with tables that don't have an explicit PAC_LINK. The system uses the privacy unit's PAC_KEY/PRIMARY KEY for hashing:

```sql
-- Regular table without PAC_LINK
CREATE TABLE sales (
    sale_id INTEGER,
    customer_id INTEGER,
    amount DECIMAL(10,2)
);

-- This works: PU's PAC_KEY is used for privacy
SELECT SUM(amount) FROM customers 
JOIN sales ON customers.id = sales.customer_id;
```

## License

See `LICENSE` in the repository root.

## Maintainer

This extension is maintained by **@ila** (ilaria@cwi.nl).

## Literature
WIP: We will add references to relevant papers and resources on PAC privacy and privacy-preserving query processing here.
