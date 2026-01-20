//
// Created by ila on 1/19/26.
//

#include "include/pac_tpch_compiler_benchmark.hpp"

#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/common/printer.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <stdexcept>
#include <ctime>
#include <iomanip>
#include <unistd.h>
#include <limits.h>

namespace duckdb {

static string Timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return string(buf);
}

static bool FileExists(const string &path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

static string ReadFileToString(const string &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// Format a double without trailing zeros (e.g. 0.100000 -> 0.1, 2.5000 -> 2.5)
static string FormatNumber(double v) {
	std::ostringstream oss;
	// Use high precision to avoid losing significant digits, but we'll trim trailing zeros ourselves
	oss << std::setprecision(15) << std::defaultfloat << v;
	string s = oss.str();
	// If there's a decimal point, trim trailing zeros
	auto pos = s.find('.');
	if (pos != string::npos) {
		// remove trailing zeros
		while (!s.empty() && s.back() == '0') { s.pop_back(); }
		// if decimal point is now last, remove it too
		if (!s.empty() && s.back() == '.') { s.pop_back(); }
	}
	return s;
}

static string FindQueriesDirectory(const string &folder_name) {
    // Try common relative locations
    vector<string> candidates = {
        folder_name,
        "./" + folder_name,
        "../" + folder_name,
        "../../" + folder_name,
        "benchmark/" + folder_name,
        "../benchmark/" + folder_name,
        "../../benchmark/" + folder_name
    };

    for (auto &cand : candidates) {
        struct stat st;
        if (stat(cand.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            return cand;
        }
    }

    // Try relative to executable
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
    if (len != -1) {
        exe_path[len] = '\0';
        string dir = string(exe_path);
        auto pos = dir.find_last_of('/');
        if (pos != string::npos) {
            dir = dir.substr(0, pos);
        }

        for (int i = 0; i < 6; ++i) {
            string cand = dir + "/benchmark/" + folder_name;
            struct stat st;
            if (stat(cand.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                return cand;
            }
            auto p2 = dir.find_last_of('/');
            if (p2 == string::npos) break;
            dir = dir.substr(0, p2);
        }
    }

    return "";
}

static string FindSchemaFile(const string &filename) {
    // Try common relative locations
    vector<string> candidates = {
        filename,
        "./" + filename,
        "../" + filename,
        "../../" + filename,
        "benchmark/" + filename,
        "../benchmark/" + filename,
        "../../benchmark/" + filename
    };

    for (auto &cand : candidates) {
        if (FileExists(cand)) {
            return cand;
        }
    }

    // Try relative to executable
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
    if (len != -1) {
        exe_path[len] = '\0';
        string dir = string(exe_path);
        auto pos = dir.find_last_of('/');
        if (pos != string::npos) {
            dir = dir.substr(0, pos);
        }

        for (int i = 0; i < 6; ++i) {
            string cand = dir + "/benchmark/" + filename;
            if (FileExists(cand)) {
                return cand;
            }
            auto p2 = dir.find_last_of('/');
            if (p2 == string::npos) break;
            dir = dir.substr(0, p2);
        }
    }

    return "";
}

static void CreateTPCHDatabase(Connection &con, double scale_factor) {
    std::cout << "[" << Timestamp() << "] Creating TPC-H database with SF=" << scale_factor << std::endl;

    // Execute schema file to create tables with PK/FK constraints
    std::cout << "[" << Timestamp() << "] Creating TPC-H schema with constraints..." << std::endl;
	string schema_file = FindSchemaFile("pac_tpch_schema.sql");
	if (schema_file.empty()) {
		throw std::runtime_error("Cannot find pac_tpch_schema.sql");
	}

    string schema_sql = ReadFileToString(schema_file);
    auto schema_result = con.Query(schema_sql);
    if (schema_result->HasError()) {
        throw std::runtime_error("Failed to create schema: " + schema_result->GetError());
    }

    // Generate TPC-H data
    std::cout << "[" << Timestamp() << "] Generating TPC-H data..." << std::endl;
    con.Query("CALL dbgen(sf=" + FormatNumber(scale_factor) + ")");

    std::cout << "[" << Timestamp() << "] TPC-H database created successfully" << std::endl;
}

static unique_ptr<MaterializedQueryResult> ExecuteQueryWithTiming(
    Connection &con,
    const string &query,
    double &exec_time_ms) {

    auto start = std::chrono::high_resolution_clock::now();
    unique_ptr<MaterializedQueryResult> result;

    try {
        result = con.Query(query);
        auto end = std::chrono::high_resolution_clock::now();
        exec_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (result->HasError()) {
            throw std::runtime_error(result->GetError());
        }
    } catch (const std::exception &e) {
        auto end = std::chrono::high_resolution_clock::now();
        exec_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        throw;
    }

    return result;
}

static bool CompareResults(MaterializedQueryResult &r1, MaterializedQueryResult &r2) {
    // Compare row counts
    if (r1.RowCount() != r2.RowCount()) {
        return false;
    }

    // Compare column counts
    if (r1.ColumnCount() != r2.ColumnCount()) {
        return false;
    }

    // Compare actual data
    for (idx_t row_idx = 0; row_idx < r1.RowCount(); row_idx++) {
        for (idx_t col_idx = 0; col_idx < r1.ColumnCount(); col_idx++) {
            auto v1 = r1.GetValue(col_idx, row_idx);
            auto v2 = r2.GetValue(col_idx, row_idx);

            // Handle NULL values separately
            bool v1_is_null = v1.IsNull();
            bool v2_is_null = v2.IsNull();

            // If one is NULL and the other isn't, they don't match
            if (v1_is_null != v2_is_null) {
                return false;
            }

            // If both are NULL, they match - continue to next value
            if (v1_is_null && v2_is_null) {
                continue;
            }

            // Compare non-NULL values - they must be equal
            if (v1 != v2) {
                return false;
            }
        }
    }

    return true;
}

void RunTPCHCompilerBenchmark(double scale_factor, const string &scale_factor_str) {
    std::cout << "========================================" << std::endl;
    std::cout << "TPC-H Compiler Benchmark" << std::endl;
    std::cout << "Scale Factor: " << scale_factor << std::endl;
    std::cout << "========================================" << std::endl;

	// Scale factor must be at most 1
	if (scale_factor > 1.0) {
		throw std::runtime_error("Scale factor must be lower or equal than 1.0");
	}

    // Database setup
    string db_path = "tpch_sf" + scale_factor_str + ".db";
    bool db_exists = FileExists(db_path);

    std::cout << "[" << Timestamp() << "] Database: " << db_path;
    if (db_exists) {
        std::cout << " (exists)" << std::endl;
    } else {
        std::cout << " (will create)" << std::endl;
    }

    // Open database
    DuckDB db(db_path);
    Connection con(db);

	// Load TPC-H extension
	con.Query("INSTALL tpch");
	con.Query("LOAD tpch");

    // Create database if needed
    if (!db_exists) {
        CreateTPCHDatabase(con, scale_factor);
    }

    // Find query directories
    string pac_queries_dir = FindQueriesDirectory("tpch_pac_queries");

    if (pac_queries_dir.empty()) {
        std::cerr << "ERROR: Cannot find tpch_pac_queries directory" << std::endl;
        return;
    }

    std::cout << "[" << Timestamp() << "] PAC queries directory: " << pac_queries_dir << std::endl;

    // Set seed once for deterministic noise generation
    int seed = 42;

    // Statistics tracking
    int total_queries = 0;
    int skipped_queries = 0;
    int matched_queries = 0;
    int mismatched_queries = 0;
    int automatic_exceptions = 0;
    int manual_exceptions = 0;
    int both_failed_queries = 0;
    vector<pair<int, string>> automatic_exception_list;  // query number, exception message
    vector<pair<int, string>> manual_exception_list;     // query number, exception message
    vector<int> mismatched_query_list;

    // Run benchmarks for all 22 TPC-H queries
    std::cout << "\n[" << Timestamp() << "] Starting query benchmarks..." << std::endl;
    std::cout << "========================================" << std::endl;

    for (int q = 1; q <= 22; q++) {
        string q_num = (q < 10) ? "0" + std::to_string(q) : std::to_string(q);
        string manual_query_file = pac_queries_dir + "/q" + q_num + ".sql";

        std::cout << "\n--- Query " << q << " ---" << std::endl;

        // Check if manual query file exists
        if (!FileExists(manual_query_file)) {
            std::cout << "[SKIP] Manual PAC query file not found: " << manual_query_file << std::endl;
            skipped_queries++;
            continue;
        }

        total_queries++;

        try {
            // Read manual query file
            string manual_query = ReadFileToString(manual_query_file);

            // Build PRAGMA query for automatic compilation
            string pragma_query = "PRAGMA tpch(" + std::to_string(q) + ");";

            // ===== PHASE 1: Run automatically compiled query (PRAGMA) =====

            // Cold run of automatically compiled query (not timed)
            std::cout << "[" << Timestamp() << "] Running cold TPCH query (no PAC)..." << std::endl;
            try {
                auto cold_result = con.Query(pragma_query);
                if (cold_result->HasError()) {
                    std::cout << "[WARNING] Cold run error: " << cold_result->GetError() << std::endl;
                }
            } catch (const std::exception &e) {
                std::cout << "[WARNING] Cold run exception: " << e.what() << std::endl;
            }

        	// ===== PHASE 2: Add privacy unit =====

        	std::cout << "[" << Timestamp() << "] Adding customer as privacy unit" << std::endl;
        	auto add_result = con.Query("pragma add_pac_privacy_unit('customer')");
        	if (add_result->HasError()) {
        		std::cerr << "[ERROR] Adding privacy unit: " << add_result->GetError() << std::endl;
        		continue;
        	}

        	// ===== PHASE 3: Run automatically compiled query (PRAGMA) =====
            std::cout << "[" << Timestamp() << "] Running timed PAC automatic query..." << std::endl;

            // Reset seed for deterministic noise
            con.Query("SET pac_seed = " + std::to_string(seed));

            double automatic_time_ms = 0;
            unique_ptr<MaterializedQueryResult> automatic_result;
            bool automatic_success = false;
            string automatic_error;

            try {
                automatic_result = ExecuteQueryWithTiming(con, pragma_query, automatic_time_ms);
                automatic_success = true;
                std::cout << "[SUCCESS] PAC automatic query completed in " << std::fixed << std::setprecision(2)
                         << automatic_time_ms << " ms" << std::endl;
                std::cout << "          Result: " << automatic_result->RowCount() << " rows, "
                         << automatic_result->ColumnCount() << " columns" << std::endl;
            } catch (const std::exception &e) {
                automatic_error = e.what();
                automatic_exceptions++;
                automatic_exception_list.emplace_back(q, automatic_error);
                std::cout << "[ERROR] PAC automatic query failed in " << std::fixed << std::setprecision(2)
                         << automatic_time_ms << " ms" << std::endl;
                std::cout << "        Error: " << automatic_error << std::endl;
            }

            // ===== PHASE 4: Remove privacy unit =====

            std::cout << "[" << Timestamp() << "] Removing customer privacy unit" << std::endl;
            auto remove_result = con.Query("pragma remove_pac_privacy_unit('customer')");
            if (remove_result->HasError()) {
                std::cerr << "[ERROR] Removing privacy unit: " << remove_result->GetError() << std::endl;
            }

            // ===== PHASE 5: Run manually compiled query (from file) =====

            std::cout << "[" << Timestamp() << "] Running timed PAC manual query..." << std::endl;

            // Reset seed to ensure same noise as automatic query
            con.Query("SET pac_seed = " + std::to_string(seed));

            double manual_time_ms = 0;
            unique_ptr<MaterializedQueryResult> manual_result;
            bool manual_success = false;
            string manual_error;

            try {
                manual_result = ExecuteQueryWithTiming(con, manual_query, manual_time_ms);
                manual_success = true;
                std::cout << "[SUCCESS] PAC manual query completed in " << std::fixed << std::setprecision(2)
                         << manual_time_ms << " ms" << std::endl;
                std::cout << "          Result: " << manual_result->RowCount() << " rows, "
                         << manual_result->ColumnCount() << " columns" << std::endl;
            } catch (const std::exception &e) {
                manual_error = e.what();
                manual_exceptions++;
                manual_exception_list.emplace_back(q, manual_error);
                std::cout << "[ERROR] PAC manual query failed in " << std::fixed << std::setprecision(2)
                         << manual_time_ms << " ms" << std::endl;
                std::cout << "        Error: " << manual_error << std::endl;
            }

            // ===== COMPARE RESULTS =====

            // Compare automatic vs manual (both are PAC queries)
            if (automatic_success && manual_success) {
                bool match = CompareResults(*automatic_result, *manual_result);
                if (match) {
                    std::cout << "[MATCH] Automatic and manual PAC results are identical" << std::endl;
                    matched_queries++;
                } else {
                    std::cout << "[MISMATCH] Automatic and manual PAC results differ!" << std::endl;
                    mismatched_queries++;
                    mismatched_query_list.push_back(q);
                }

                // Show performance comparison
                double ratio = (automatic_time_ms / manual_time_ms);
                std::cout << "[PERF] PAC automatic/PAC manual ratio: " << std::fixed << std::setprecision(2)
                         << ratio << "x ";

                // Check if within ±10%
                if (ratio >= 0.9 && ratio <= 1.1) {
                    std::cout << "(PAC automatic ~ PAC manual)" << std::endl;
                } else if (ratio > 1.1) {
                    std::cout << "(PAC automatic slower)" << std::endl;
                } else {
                    std::cout << "(PAC automatic faster)" << std::endl;
                }
            } else if (!automatic_success && !manual_success) {
                // Both failed - count as both failed
                std::cout << "[BOTH FAILED] Both automatic and manual queries failed (no comparison possible)" << std::endl;
                both_failed_queries++;
            } else {
                // One succeeded and one failed - this is a mismatch in behavior
                if (automatic_success) {
                    std::cout << "[FAILURE MISMATCH] Automatic succeeded but manual failed" << std::endl;
                } else {
                    std::cout << "[FAILURE MISMATCH] Manual succeeded but automatic failed" << std::endl;
                }
                // Don't increment both_failed_queries here - this was already counted above in the exception counters
            }

        } catch (const std::exception &e) {
            std::cout << "[ERROR] Unexpected error processing query " << q << ": " << e.what() << std::endl;
            total_queries--; // Don't count this query as attempted
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "[" << Timestamp() << "] Benchmark completed" << std::endl;
    std::cout << "========================================" << std::endl;

    // Print summary statistics
    std::cout << "\n========================================" << std::endl;
    std::cout << "SUMMARY STATISTICS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total queries attempted: " << total_queries << std::endl;
    std::cout << "Skipped queries: " << skipped_queries << std::endl;
    std::cout << "Matched queries: " << matched_queries << " (both succeeded with identical results)" << std::endl;
    std::cout << "Mismatched queries: " << mismatched_queries << " (both succeeded but different results)" << std::endl;
    std::cout << "Failed queries: " << both_failed_queries << " (one or both failed)" << std::endl;
    std::cout << "  - Automatic (PRAGMA) exceptions: " << automatic_exceptions << std::endl;
    std::cout << "  - Manual exceptions: " << manual_exceptions << std::endl;

    if (!mismatched_query_list.empty()) {
        std::cout << "\nMismatched queries: ";
        for (size_t i = 0; i < mismatched_query_list.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "Q" << mismatched_query_list[i];
        }
        std::cout << std::endl;
    }

    if (!automatic_exception_list.empty()) {
        std::cout << "\nAutomatic (PRAGMA) exceptions:" << std::endl;
        for (const auto &ex : automatic_exception_list) {
            std::cout << "  Q" << ex.first << ": " << ex.second << std::endl;
        }
    }

    if (!manual_exception_list.empty()) {
        std::cout << "\nManual exceptions:" << std::endl;
        for (const auto &ex : manual_exception_list) {
            std::cout << "  Q" << ex.first << ": " << ex.second << std::endl;
        }
    }

    // Print success message if all queries matched
    if (total_queries > 0 && matched_queries == total_queries &&
        automatic_exceptions == 0 && manual_exceptions == 0) {
        std::cout << "\n✓ SUCCESS: All " << total_queries << " queries matched with no exceptions!" << std::endl;
    } else if (matched_queries == total_queries && (automatic_exceptions > 0 || manual_exceptions > 0)) {
        std::cout << "\n⚠ PARTIAL SUCCESS: All successful queries matched, but "
                  << (automatic_exceptions + manual_exceptions) << " total exceptions occurred." << std::endl;
    }

    std::cout << "========================================" << std::endl;
}

} // namespace duckdb

// Main function
int main(int argc, char **argv) {
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help") {
            std::cout << "Usage: pac_tpch_compiler_benchmark [scale_factor]\n"
                      << "  scale_factor: TPC-H scale factor (e.g., 01 for 0.1, 1 for 1.0, 10 for 10.0)\n"
                      << "\nCompares automatically compiled PAC queries (PRAGMA) against manually written PAC queries.\n"
                      << "Both versions use the same seed for reproducibility.\n";
            return 0;
        }
    }

    double scale_factor = 1.0;  // Default to SF=1
    std::string scale_factor_str = "1";  // Default string for database path

    if (argc > 1) {
        try {
            // Parse scale factor: "01" -> 0.1, "1" -> 1.0, "10" -> 10.0
            std::string arg = argv[1];
            scale_factor_str = arg;  // Keep original string for database path

            if (arg.length() == 2 && arg[0] == '0' && std::isdigit(arg[1])) {
                // Format "0X" means 0.X (e.g., "01" -> 0.1)
                scale_factor = (arg[1] - '0') / 10.0;
            } else {
                // Normal parsing
                scale_factor = std::stod(arg);
            }

            if (scale_factor <= 0) {
                std::cerr << "Error: scale factor must be positive" << std::endl;
                return 1;
            }
        } catch (const std::exception &e) {
            std::cerr << "Error: invalid scale factor: " << argv[1] << std::endl;
            return 1;
        }
    }

    if (argc > 2) {
        std::cerr << "Error: too many arguments" << std::endl;
        std::cout << "Usage: pac_tpch_compiler_benchmark [scale_factor]" << std::endl;
        return 1;
    }

    duckdb::RunTPCHCompilerBenchmark(scale_factor, scale_factor_str);
    return 0;
}
