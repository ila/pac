# PAC Benchmarks

This document provides an overview of the benchmark suite for the PAC extension. To clone the repository, do the following:
```bash
git clone --recurse-submodules https://github.com/cwida/pac.git
```

## Why Benchmarks?

PAC provides privacy guarantees by transforming SQL aggregates into privacy-preserving versions. This transformation introduces computational overhead that we need to measure and optimize. The benchmarks help us answer key questions:

1. **PAC Overhead**: How much slower are PAC queries compared to standard DuckDB queries? The bitslice compilation algorithm adds per-row hashing and probabilistic counting overhead.

2. **Join Overhead**: PAC requires joining through the privacy unit chain (e.g., `lineitem → orders → customer`) to obtain the privacy key hash. This adds join costs even when the original query doesn't need those joins.

## Benchmark Executables

PAC benchmarks are compiled as **separate standalone executables** (not run through DuckDB's extension loading). They are built using CMake targets:

| Executable | CMake Target | Description |
|------------|--------------|-------------|
| `pac_tpch_benchmark` | `pac_tpch_benchmark` | TPC-H benchmark comparing PAC vs baseline |
| `pac_tpch_compiler_benchmark` | `pac_tpch_compiler_benchmark` | Compiler benchmark comparing auto-compiled vs manual PAC queries |
| `pac_clickhouse_benchmark` | `pac_clickhouse_benchmark` | ClickBench benchmark for web analytics workloads |

### Reproducibility
All the benchmarks have been executed on Ubuntu 24.04. Further, the latest version of `clang` should be compiled and installed, to then compile our extension with it.
```bash
sudo apt update && sudo apt upgrade
sudo apt install -y \
  build-essential \
  cmake ninja-build git python3 \
  libffi-dev zlib1g-dev \
  libncurses-dev libxml2-dev \
  libedit-dev libssl-dev ccache
git clone https://github.com/llvm/llvm-project.git
cd llvm-project 
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm-main \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RTTI=ON
ninja -j$(nproc)
sudo ninja install  
echo 'export PATH=/opt/llvm-main/bin:$PATH' | sudo tee /etc/profile.d/llvm-main.sh
source /etc/profile.d/llvm-main.sh
sudo update-alternatives --install /usr/bin/cc cc /opt/llvm-main/bin/clang 100
sudo update-alternatives --install /usr/bin/c++ c++ /opt/llvm-main/bin/clang++ 100
```
After these steps, `which c++` should point to `/opt/llvm-main/bin/clang++`, and `clang++ --version` should show the latest Clang version. This ensures that the benchmarks are compiled with the latest optimizations and features from Clang.

Then, build DuckDB and the PAC extension in Release mode (assuming the `duckdb` submodule is present):
```bash
cd /path/to/pac
GEN=ninja make
```
For instructions on how to run the individual benchmarks, please refer to the respective sections below.

### Folder Structure
```
benchmark/
├── include/                        # Benchmark headers
│   ├── pac_tpch_benchmark.hpp
│   └── pac_tpch_compiler_benchmark.hpp
├── pac_microbench/                 # Microbenchmark suite
├── pac_tpch_benchmark.cpp          # TPC-H benchmark runner
├── pac_tpch_compiler_benchmark.cpp # Compiler benchmark runner
│   ├── pac_clickhouse_benchmark.hpp
├── plot_tpch_results.R             # R script for plotting results
├── tpch_pac_queries/               # Hand-written optimized PAC queries
├── clickbench_queries/             # ClickBench query suite
└── tpch_pac_simple_hash_queries/   # Simple hash PAC query variants
├── pac_clickhouse_benchmark.cpp    # ClickBench benchmark runner

## Output

├── plot_clickbench_results.R       # R script for ClickBench plots
├── plot_tpch_results.R             # R script for TPC-H plots
- PNG plots are generated automatically if R and required packages are installed
