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
