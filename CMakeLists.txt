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
include_directories(src/include third_party/lpts)

# Extension sources
set(EXTENSION_SOURCES src/pac_extension.cpp src/pac_optimizer.cpp src/pac_compiler.cpp src/pac_privacy_unit.cpp
        src/pac_compatibility_check.cpp src/pac_aggregate.cpp src/pac_helpers.cpp third_party/lpts/logical_plan_to_sql.cpp
        src/pac_count.cpp src/pac_sum.cpp src/pac_bitslice_compiler.cpp src/pac_compiler_helpers.cpp)
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

# Add a small test runner that lives inside the pac project (does not modify duckdb sources)
# Temporarily disabled due to ReplaceNode API changes
#add_executable(test_update_parent_runner src/test_update_parent_runner.cpp)
#target_link_libraries(test_update_parent_runner ${EXTENSION_NAME})

#add_executable(test_pac_privacy_columns src/test_pac_privacy_columns.cpp)
#target_link_libraries(test_pac_privacy_columns ${EXTENSION_NAME})
