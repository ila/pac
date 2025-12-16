# PAC — DuckDB Privacy-Aware Aggregation Extension

PAC (Privacy-Aware-Computed) is a small DuckDB optimizer extension that enforces a set of privacy rules on aggregation queries over designated tables ("PAC tables"). If a query against a PAC table does not satisfy the rules the extension rejects it with a clear ParserException message describing the reason.

This README focuses on practical developer workflows: building, running, configuring, and testing the extension.

What PAC enforces
- The query must scan at least one table listed in the PAC tables file.
- Allowed aggregates: SUM, COUNT, AVG. Other aggregates (MIN, MAX, custom aggregates) are disallowed.
- Nested aggregates (an aggregate inside another aggregate) are disallowed.
- Window functions are disallowed.
- DISTINCT is disallowed.
- Only INNER JOIN is allowed. Outer joins, cross products, and set operations (UNION/EXCEPT/INTERSECT) are disallowed.

When a query is rejected the optimizer throws a ParserException with one of the explanatory messages, for example:
- "Query does not scan any PAC table!"
- "Query does not contain any allowed aggregation (sum, count, avg)!"
- "Query contains disallowed aggregates (only sum, count, avg allowed; no nested aggregates)!"
- "Query contains window functions, which are not allowed in PAC-compatible queries!"
- "Query contains DISTINCT, which is not allowed in PAC-compatible queries!"
- "Query contains disallowed joins (only INNER JOIN allowed in PAC-compatible queries)!"

Repository layout (important parts)
- `src/`, `include/` — extension source code and headers
- `test/sql/` — SQL tests exercised by the DuckDB test runner
- `duckdb/` — vendored DuckDB source used to build the extension
- `build/` — out-of-tree build output (debug/release)

Quick build (Makefile shortcut)
- The repository ships a Makefile target that builds DuckDB + extension. From the repo root:

```sh
make
```

This produces artifacts under `build/release/` (or `build/debug` if you configure a debug build): the DuckDB binary, test runner, and the extension library.

Manual build (CMake + Ninja)

```sh
cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j$(nproc)
```

To build Release, switch to `-DCMAKE_BUILD_TYPE=Release` and a different build directory.

Configuration
- `pac_privacy_file` is a DuckDB setting that points to a newline-separated CSV file listing PAC table names (one name per line). Example:

```sql
SET pac_privacy_file = 'test/sql/test_pac_tables.csv';
```

- Runtime PRAGMAs exposed for tests and convenience:
  - `PRAGMA add_pac_privacy_unit('table_name');`
  - `PRAGMA remove_pac_privacy_unit('table_name');`

These PRAGMAs are test helpers that modify the CSV file used by the tests.

PAC SQL functions
-----------------

The extension provides a PAC-aware aggregation helper that operates on per-sample materialized
arrays produced by a single privacy unit (table). The function enforces per-sample semantics
and (optionally) checks the configured `pac_m` sample count.

pac_aggregate(signature)
- signature: `pac_aggregate(list<double> values, list<int> counts, double mi, int k) -> double`
  - `values`: list of per-sample aggregated values (one element per sample position)
  - `counts`: list of per-sample counts (one integer per sample position)
  - `mi`: the privacy parameter used in the PAC mechanism (double)
  - `k`: minimum required count in a sample position (integer)

Behavior and rules
- The `values` and `counts` arguments must be arrays with the same length (one count per value). If they
  differ the function throws an error.
- When the session setting `enforce_m_values` is true (the default) the function also enforces that each
  per-sample array has length equal to the configured `pac_m` (default 128). If `enforce_m_values` is false
  the `pac_aggregate` will accept arrays of any length as long as `values` and `counts` match in length.
- Any NULL in the per-sample `values` causes the function to return NULL for that group (strict null refusal).
- The function uses the configured `pac_seed` (if set) to initialize a deterministic RNG for the sampled
  noise; `pac_noise` controls whether PAC noise is applied at all (useful for deterministic tests).

Examples

Disable strict m-enforcement for testing and run a simple call:

```sql
SET pac_seed = 12345;
SET pac_noise = false;
SET enforce_m_values = false; -- allow non-m-length arrays for tests

SELECT pac_aggregate(ARRAY[42.0, 42.0, 42.0, 42.0], ARRAY[3,3,3,3], 1.0/128, 3);
```

Typical per-sample usage pattern (materialize per-sample rows into a table with `sample_id` and then aggregate):

```sql
CREATE TABLE per_sample AS
SELECT src.group_key, s.sample_id, /* per-sample value and cnt */
  FROM src_table src
  CROSS JOIN generate_series(1,128) AS s(sample_id);

SELECT group_key, pac_aggregate(array_agg(val ORDER BY sample_id), array_agg(cnt ORDER BY sample_id), 1.0/128, 3)
FROM per_sample
GROUP BY group_key;
```

New configuration settings
- `pac_noise` (BOOLEAN): whether PAC noise is added by the mechanism. Useful for deterministic tests.
- `pac_seed` (BIGINT): deterministic RNG seed for PAC-related functions (useful for tests).
- `pac_m` (INTEGER): configured number of per-sample subsets (m). Default: 128.
- `enforce_m_values` (BOOLEAN): when true (default) PAC functions enforce that per-sample arrays have length `pac_m`.
  Tests may set this to `false` to avoid strict length checks.
- `pac_compiled_path` (VARCHAR): filesystem path where compiled PAC artifacts (generated CTE SQL) are written. Default: `.` (current working directory).

When PAC compilation runs (optimizer detects a PAC-compatible query against a single privacy unit) the
compiler writes a sample CTE SQL file named `<privacy_unit>_<query_hash>.sql` to `pac_compiled_path`.
The `query_hash` is derived from a normalized form of the original query (newlines collapsed, lowercased, extra
whitespace removed) so compiled artifacts are stable for the same query text.

Notes
- The optimizer enforces that a PAC query references at most one privacy unit (a ParserException is thrown if
  a query references multiple PAC tables). The compiler emits a template CTE file; schema-aware generation
  and automatic materialization are future enhancements.

Running the extension (DuckDB shell)

Start the built DuckDB binary and use SQL normally. The extension validates queries during optimization and will reject incompatible queries with a ParserException printed by the shell.

```sh
./build/release/duckdb
```

Running tests
- The SQL tests in `test/sql/` are executed by the DuckDB `unittest` binary.

Convenience (Makefile):
```sh
make test
```

Run the test runner directly (example for debug build):

```sh
build/debug/test/unittest --test-dir ../../.. [sql] -R pac -V
```

Adjust paths and filters to your build layout and desired test selection.

Building and running tests with Ninja
- If you prefer Ninja as the generator you can drive the Makefile with the `GEN=ninja` variable; this repository exposes a convenience Make target for common CMake workflows. Example (Debug build):

```sh
GEN=ninja make debug
```

This creates a `build/debug/` directory and compiles the DuckDB binary and test binaries using Ninja. After the build, test binaries live under `build/debug/test/`. To run the updated test executable added in this patch (all tests are contained in `src/test_update_parent_aggregate.cpp`), run:

```sh
build/debug/test/test_update_parent_aggregate
```

CLion / Debugging tips
- Open `duckdb/CMakeLists.txt` or the project root in CLion and point the CMake profile to an out-of-tree build directory (for example `build/debug`).
- Add the extension CMake config if needed: pass `-DDUCKDB_EXTENSION_CONFIGS=<path-to-pac/CMakeLists.txt>` to the DuckDB CMake invocation so the main build knows about the extension.
- Create a Run/Debug configuration that runs the `unittest` binary and pass program arguments to filter SQL tests (e.g. `--test-dir ../../.. [sql] -R pac`).
- If CLion's Run/Debug console disappears, check Run -> Edit Configurations -> ensure the correct binary is selected and that the "Show command line afterwards" or similar console options are enabled.

Contributing
- Add SQL tests under `test/sql/` for any behavior changes.
- Run the tests locally before submitting a patch.

License
- See `LICENSE` in the repository root.
