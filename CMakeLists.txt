cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME pac)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
#find_package(OpensSL REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)
include_directories(benchmark/include)

# Option to disable approximate sum algorithm (use exact cascading instead)
# By default, pac_sum/pac_avg use the approx algorithm (16-bit cascading counters)
# Set PAC_NOAPPROX=ON to use exact cascading instead
option(PAC_NOAPPROX "Disable approximate sum algorithm, use exact cascading" OFF)
if(PAC_NOAPPROX)
    add_definitions(-DPAC_NOAPPROX)
    message(STATUS "PAC: Using exact cascading sum algorithm (PAC_NOAPPROX)")
else()
    message(STATUS "PAC: Using approximate sum algorithm (default)")
endif()

# Extension sources
set(EXTENSION_SOURCES src/pac_extension.cpp src/pac_optimizer.cpp src/pac_privacy_unit.cpp
        src/pac_compatibility_check.cpp src/pac_aggregate.cpp src/pac_helpers.cpp
        src/pac_count.cpp src/pac_min_max.cpp src/pac_bitslice_compiler.cpp src/pac_compiler_helpers.cpp
        src/pac_expression_builder.cpp src/pac_plan_traversal.cpp src/pac_projection_propagation.cpp
        src/pac_parser.cpp src/pac_metadata_manager.cpp src/pac_metadata_serialization.cpp src/pac_parser_helpers.cpp
        src/pac_subquery_handler.cpp src/pac_categorical.cpp src/pac_categorical_rewriter.cpp src/pac_sum.cpp src/pac_avg.cpp)
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

# Test runners and benchmarks - only build for native platforms, not WASM or Windows
# (Windows lacks unistd.h and other POSIX features used by the benchmark code)
if(NOT EMSCRIPTEN AND NOT WIN32)
# Build a single unified test executable that invokes all tests via src/test/test_runner.cpp
add_executable(pac_test_runner
    src/test/test_runner.cpp
    src/test/test_plan_traversal.cpp
    src/test/test_schema_metadata.cpp
    src/test/test_compiler_functions.cpp
    src/test/test_pac_parser.cpp)
target_link_libraries(pac_test_runner ${EXTENSION_NAME})

# TPCH + PAC benchmark runner
add_executable(pac_tpch_benchmark benchmark/pac_tpch_benchmark.cpp)
set_target_properties(pac_tpch_benchmark PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_link_libraries(pac_tpch_benchmark ${EXTENSION_NAME})

# TPCH Compiler benchmark runner (compares PAC vs regular TPC-H)
add_executable(pac_tpch_compiler_benchmark benchmark/pac_tpch_compiler_benchmark.cpp)
set_target_properties(pac_tpch_compiler_benchmark PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_link_libraries(pac_tpch_compiler_benchmark ${EXTENSION_NAME})
endif()
