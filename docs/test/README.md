# Testing PAC Extension

This document covers how to run SQL and C++ tests for the PAC extension.

## Quick Start

From the repository root:

```bash
# Run all tests (release build)
make test

# Run all tests (debug build)
make test_debug
```

## SQL Tests

SQL tests are located in `test/sql/` and use DuckDB's [SQLLogicTest](https://duckdb.org/dev/sqllogictest/intro.html) format.

### Running SQL Tests

```bash
# From repository root
cd /path/to/pac

# Run all SQL tests
./build/release/test/unittest --test-dir . [sql]

# Run tests matching "pac"
./build/release/test/unittest --test-dir . [sql] -R pac

# Run a specific test file
./build/release/test/unittest --test-dir . [sql] test/sql/pac_parser.test

# Verbose output
./build/release/test/unittest --test-dir . [sql] -R pac -V
```

### Test Runner Arguments

| Argument | Description |
|----------|-------------|
| `--test-dir .` | Set test directory to repo root |
| `[sql]` | Run only SQL tests |
| `-R <pattern>` | Filter tests by regex pattern |
| `-V` | Verbose output |
| `-e <pattern>` | Exclude tests matching pattern |

### Writing SQL Tests

SQL tests use a simple format:

```sql
# name: test/sql/my_test.test
# description: Test description
# group: [sql]

require pac

statement ok
CREATE PAC TABLE test (id INTEGER, PAC_KEY (id));

query I
SELECT COUNT(*) FROM test;
----
0

statement error
SELECT * FROM test;
----
Query does not contain any allowed aggregation
```

### Test Directives

| Directive | Description |
|-----------|-------------|
| `statement ok` | Statement should succeed |
| `statement error` | Statement should fail (can specify expected error) |
| `query <types>` | Query returning results (I=integer, R=real, T=text) |
| `----` | Separator between query and expected result |
| `require pac` | Load the PAC extension |

## C++ Tests

C++ tests are located in `src/test/` and use a custom test framework.

### Running C++ Tests

C++ tests are compiled into a standalone `pac_test_runner` executable:

```bash
# From repository root
cd /path/to/pac

# Build the test runner
make debug

# Run C++ tests
./build/debug/pac_test_runner
```

### Test Files

| File | Description |
|------|-------------|
| `src/test/test_runner.cpp` | Main test runner |
| `src/test/test_pac_parser.cpp` | Parser tests |
| `src/test/test_schema_metadata.cpp` | Schema metadata tests |
| `src/test/test_plan_traversal.cpp` | Plan traversal tests |
| `src/test/test_compiler_functions.cpp` | Compiler function tests |

## Debugging Failed Tests

### Get More Information

```bash
# Verbose mode shows full query output
./build/debug/test/unittest --test-dir . [sql] -R pac -V

# Show actual vs expected
./build/debug/test/unittest --test-dir . [sql] test/sql/failing_test.test -V
```

### Run Single Test Interactively

```bash
# Start DuckDB shell
./build/debug/duckdb

# Manually run the failing SQL
D CREATE PAC TABLE test (id INTEGER, PAC_KEY (id));
D SELECT COUNT(*) FROM test;
```

### Debug Build

For debugging with GDB/LLDB:

```bash
# Build debug version
make debug

# Run with debugger
gdb --args ./build/debug/test/unittest --test-dir . [sql] test/sql/failing_test.test
```

## Test Organization

```
test/
└── sql/
    ├── pac_parser.test              # Parser and syntax tests
    ├── pac_bitslice_compiler.test   # Compiler tests
    ├── pac_sum.test                 # pac_sum aggregate tests
    ├── pac_count.test               # pac_count aggregate tests
    ├── pac_min_max.test             # pac_min/pac_max tests
    └── ...
```

## Continuous Integration

Tests run automatically on push via GitHub Actions. See `.github/workflows/` for CI configuration.
