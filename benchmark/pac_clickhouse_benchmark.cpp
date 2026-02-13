//
// Created by ila on 02/12/26.
//

// Implement the ClickHouse Hits (ClickBench) benchmark runner.

#include "include/pac_clickhouse_benchmark.hpp"

#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/common/printer.hpp"
#include <cstdio>
#include <cstdlib>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <sys/stat.h>
#include <stdexcept>
#include <ctime>
#include <algorithm>
#include <iomanip>
#include <thread>
#include <unistd.h>
#include <limits.h>
#include <regex>

namespace duckdb {

static string Timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return string(buf);
}

// Format a double without trailing zeros
static string FormatNumber(double v) {
    std::ostringstream oss;
    oss << std::setprecision(5) << std::defaultfloat << v;
    string s = oss.str();
    auto pos = s.find('.');
    if (pos != string::npos) {
        while (!s.empty() && s.back() == '0') { s.pop_back(); }
        if (!s.empty() && s.back() == '.') { s.pop_back(); }
    }
    return s;
}

static void Log(const string &msg) {
    Printer::Print("[" + Timestamp() + "] " + msg);
}

static bool FileExists(const string &path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

static string ReadFileToString(const string &path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return string();
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Split a string by newlines, returning non-empty lines
static vector<string> SplitLines(const string &content) {
    vector<string> lines;
    std::istringstream iss(content);
    string line;
    while (std::getline(iss, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start != string::npos && end != string::npos) {
            string trimmed = line.substr(start, end - start + 1);
            // Skip empty lines and comments
            if (!trimmed.empty() && trimmed[0] != '-' && trimmed.substr(0, 2) != "--") {
                lines.push_back(trimmed);
            }
        }
    }
    return lines;
}

// Execute a shell command and return exit code
static int ExecuteCommand(const string &cmd) {
    Log(string("Executing: ") + cmd);
    int ret = system(cmd.c_str());
    if (ret == -1) {
        return -1;
    }
    if (WIFEXITED(ret)) {
        return WEXITSTATUS(ret);
    }
    return ret;
}

// Find a file in common relative locations
static string FindFile(const string &filename) {
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
            string cand = dir + "/" + filename;
            if (FileExists(cand)) {
                return cand;
            }
            cand = dir + "/benchmark/" + filename;
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

// Structure to hold query result statistics
struct BenchmarkQueryResult {
    int query_num;
    string mode;  // "baseline" or "PAC"
    int run;
    double time_ms;
    bool success;
    string error_msg;
};

int RunClickHouseBenchmark(const string &db_path, const string &queries_dir, const string &out_csv, bool micro) {
    try {
        Log(string("Starting ClickBench benchmark"));
        Log(string("micro flag: ") + (micro ? string("true") : string("false")));

        // Determine working directory for dataset
        char cwd[PATH_MAX];
        if (!getcwd(cwd, sizeof(cwd))) {
            throw std::runtime_error("Failed to get current working directory");
        }
        string work_dir = string(cwd);

        // Dataset path - use parquet format
        string parquet_path = work_dir + "/hits.parquet";

        // Check if we need to download the dataset
        if (!FileExists(parquet_path)) {
            Log("Downloading ClickHouse hits dataset (parquet format)...");
            string download_cmd = "wget -O \"" + parquet_path + "\" https://datasets.clickhouse.com/hits_compatible/hits.parquet";
            int ret = ExecuteCommand(download_cmd);
            if (ret != 0) {
                // Try curl as fallback
                Log("wget failed, trying curl...");
                download_cmd = "curl -L -o \"" + parquet_path + "\" https://datasets.clickhouse.com/hits_compatible/hits.parquet";
                ret = ExecuteCommand(download_cmd);
                if (ret != 0) {
                    throw std::runtime_error("Failed to download hits.parquet");
                }
            }
            Log("Download complete.");
        } else {
            Log("hits.parquet already exists, skipping download.");
        }

        // Determine database path - use different db for micro mode
        string db_actual;
        if (!db_path.empty()) {
            db_actual = db_path;
        } else if (micro) {
            db_actual = "clickbench_micro.db";
        } else {
            db_actual = "clickbench.db";
        }

        bool db_exists = FileExists(db_actual);

        if (db_exists) {
            Log(string("Connecting to existing DuckDB database: ") + db_actual);
        } else {
            Log(string("Creating new DuckDB database: ") + db_actual);
        }

        DuckDB db(db_actual.c_str());
        Connection con(db);

        // Find query files
        string create_sql_path = queries_dir + "/create.sql";
        string load_sql_path = queries_dir + "/load.sql";
        string queries_sql_path = queries_dir + "/queries.sql";

        // Try to find the files
        string create_file = FindFile(create_sql_path);
        string load_file = FindFile(load_sql_path);
        string queries_file = FindFile(queries_sql_path);

        if (create_file.empty()) {
            throw std::runtime_error("Cannot find create.sql in " + queries_dir);
        }
        if (load_file.empty()) {
            throw std::runtime_error("Cannot find load.sql in " + queries_dir);
        }
        if (queries_file.empty()) {
            throw std::runtime_error("Cannot find queries.sql in " + queries_dir);
        }

        Log(string("Using create.sql: ") + create_file);
        Log(string("Using load.sql: ") + load_file);
        Log(string("Using queries.sql: ") + queries_file);

        // Read and parse queries
        string create_sql = ReadFileToString(create_file);
        string load_sql = ReadFileToString(load_file);
        string queries_content = ReadFileToString(queries_file);

        if (create_sql.empty()) {
            throw std::runtime_error("create.sql is empty or unreadable");
        }
        if (load_sql.empty()) {
            throw std::runtime_error("load.sql is empty or unreadable");
        }

        vector<string> queries = SplitLines(queries_content);
        if (queries.empty()) {
            throw std::runtime_error("No queries found in queries.sql");
        }

        Log(string("Found ") + std::to_string(queries.size()) + " queries to benchmark");

        // Check if table already exists and has data
        auto check_result = con.Query("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'hits'");
        bool table_exists = false;
        if (check_result && !check_result->HasError()) {
            auto chunk = check_result->Fetch();
            if (chunk && chunk->size() > 0) {
                auto count = chunk->GetValue(0, 0).GetValue<int64_t>();
                table_exists = (count > 0);
            }
        }

        // If table exists, check if it has any rows
        bool needs_reload = false;
        if (table_exists) {
            auto row_count_result = con.Query("SELECT COUNT(*) FROM hits");
            if (row_count_result && !row_count_result->HasError()) {
                auto chunk = row_count_result->Fetch();
                if (chunk && chunk->size() > 0) {
                    auto row_count = chunk->GetValue(0, 0).GetValue<int64_t>();
                    Log(string("hits table has ") + std::to_string(row_count) + " rows");
                    if (row_count == 0) {
                        Log("Table is empty, will reload data");
                        needs_reload = true;
                    }
                }
            }
        }

        if (!table_exists || needs_reload) {
            if (!table_exists) {
                // Create table
                Log("Creating hits table...");
                auto create_result = con.Query(create_sql);
                if (create_result && create_result->HasError()) {
                    throw std::runtime_error("Failed to create table: " + create_result->GetError());
                }
            }

            // Load data - replace placeholder with actual parquet path
            Log("Loading data into hits table... (this may take a while)");
            string actual_load_sql = load_sql;

            // Replace 'hits.parquet' placeholder with actual path
            std::regex parquet_regex("'hits\\.parquet'");
            actual_load_sql = std::regex_replace(actual_load_sql, parquet_regex, "'" + parquet_path + "'");

            // If micro mode, add LIMIT 5000000 to the SELECT statement
            if (micro) {
                Log("Micro mode: limiting data load to 5m rows");
                // Insert LIMIT 5000000 before the final semicolon or at the end of FROM clause
                // Pattern: match the read_parquet(...) and add LIMIT after it
                std::regex from_parquet_regex("(FROM\\s+read_parquet\\([^)]+\\)[^;]*)");
                actual_load_sql = std::regex_replace(actual_load_sql, from_parquet_regex, "$1 LIMIT 5000000");
            }

            Log(string("Executing load SQL: ") + actual_load_sql.substr(0, 200) + "...");

            auto load_result = con.Query(actual_load_sql);
            if (load_result && load_result->HasError()) {
                throw std::runtime_error("Failed to load data: " + load_result->GetError());
            }
            Log("Data loading complete.");
        } else {
            Log("hits table already exists, skipping creation and loading.");
        }

        // Prepare output CSV
        string actual_out = out_csv;
        if (actual_out.empty()) {
            if (micro) {
                actual_out = "benchmark/clickbench_micro_results.csv";
            } else {
                actual_out = "benchmark/clickbench_results.csv";
            }
        }

        std::ofstream csv(actual_out, std::ofstream::out | std::ofstream::trunc);
        if (!csv.is_open()) {
            throw std::runtime_error("Failed to open output CSV: " + actual_out);
        }
        csv << "query,mode,run,time_ms,success,error\n";

        vector<BenchmarkQueryResult> all_results;

        // =====================================================================
        // Phase 1: Baseline runs (no PAC)
        // =====================================================================
        Log("=== Phase 1: Baseline runs (no PAC) ===");

        // Ensure PAC is disabled before baseline runs
        Log("Removing any existing PAC constraints...");
        auto unset_result = con.Query("ALTER PAC TABLE hits DROP PROTECTED (UserID, ClientIP);");
        if (unset_result && unset_result->HasError()) {
            Log(string("Note: Could not drop PROTECTED columns (may not exist): ") + unset_result->GetError());
        }
        unset_result = con.Query("ALTER TABLE hits UNSET PAC;");
        if (unset_result && unset_result->HasError()) {
            Log(string("Note: Could not UNSET PAC (may not be set): ") + unset_result->GetError());
        }

        // Cold run
        Log("--- Cold run ---");
        for (size_t q = 0; q < queries.size(); ++q) {
            int qnum = static_cast<int>(q + 1);
            Log(string("Cold run Q") + std::to_string(qnum));
            auto r = con.Query(queries[q]);
            if (r && r->HasError()) {
                Log(string("Cold run Q") + std::to_string(qnum) + " error: " + r->GetError());
            }
        }

        // Warm runs (3 times)
        for (int run = 1; run <= 3; ++run) {
            Log(string("--- Baseline warm run ") + std::to_string(run) + " ---");

            // Small delay between runs
            std::this_thread::sleep_for(std::chrono::milliseconds(200));

            for (size_t q = 0; q < queries.size(); ++q) {
                int qnum = static_cast<int>(q + 1);

                auto t0 = std::chrono::steady_clock::now();
                auto r = con.Query(queries[q]);
                auto t1 = std::chrono::steady_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                BenchmarkQueryResult result;
                result.query_num = qnum;
                result.mode = "baseline";
                result.run = run;
                result.time_ms = time_ms;
                result.success = !(r && r->HasError());
                result.error_msg = (r && r->HasError()) ? r->GetError() : "";

                all_results.push_back(result);

                if (result.success) {
                    Log(string("Q") + std::to_string(qnum) + " run " + std::to_string(run) +
                        " time: " + FormatNumber(time_ms) + " ms");
                } else {
                    Log(string("Q") + std::to_string(qnum) + " run " + std::to_string(run) +
                        " ERROR: " + result.error_msg);
                }

                csv << qnum << ",baseline," << run << "," << FormatNumber(time_ms) << ","
                    << (result.success ? "true" : "false") << ",\""
                    << (result.error_msg.empty() ? "" : result.error_msg) << "\"\n";
            }
        }

        // =====================================================================
        // Phase 2: Add PAC protection
        // =====================================================================
        Log("=== Phase 2: Adding PAC protection ===");

    	// Add PAC table
    	Log("Marking hits table as a PAC table...");
    	auto pac_result = con.Query("ALTER TABLE hits SET PAC;");
    	if (pac_result && pac_result->HasError()) {
    		Log(string("Warning: Failed to mark table as PAC: ") + pac_result->GetError());
    	}

        // Add PROTECTED columns annotation
        // We use ALTER PAC TABLE to add protected columns
        Log("Adding UserID and ClientIP as protected columns...");

        // Add PROTECTED columns to hits table
        pac_result = con.Query("ALTER PAC TABLE hits ADD PROTECTED (UserID, ClientIP);");
        if (pac_result && pac_result->HasError()) {
            Log(string("Warning: Failed to add PROTECTED columns: ") + pac_result->GetError());
            // Try alternative: SET PAC on the table (requires PRIMARY KEY)
            Log("Attempting to mark table as PAC privacy unit...");
            pac_result = con.Query("ALTER TABLE hits SET PAC;");
            if (pac_result && pac_result->HasError()) {
                Log(string("Warning: Failed to SET PAC: ") + pac_result->GetError());
            }
        }

        Log("PAC protection configured.");

        // =====================================================================
        // Phase 3: PAC runs
        // =====================================================================
        Log("=== Phase 3: PAC runs ===");

        int pac_success_count = 0;
        int pac_rejected_count = 0;
        int pac_crash_count = 0;

        for (int run = 1; run <= 3; ++run) {
            Log(string("--- PAC run ") + std::to_string(run) + " ---");

            std::this_thread::sleep_for(std::chrono::milliseconds(200));

            for (size_t q = 0; q < queries.size(); ++q) {
                int qnum = static_cast<int>(q + 1);

                auto t0 = std::chrono::steady_clock::now();
                auto r = con.Query(queries[q]);
                auto t1 = std::chrono::steady_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                BenchmarkQueryResult result;
                result.query_num = qnum;
                result.mode = "PAC";
                result.run = run;
                result.time_ms = time_ms;
                result.success = !(r && r->HasError());
                result.error_msg = (r && r->HasError()) ? r->GetError() : "";

                all_results.push_back(result);

                if (result.success) {
                    pac_success_count++;
                    Log(string("Q") + std::to_string(qnum) + " run " + std::to_string(run) +
                        " time: " + FormatNumber(time_ms) + " ms");
                } else {
                    // Check if it's a PAC rejection or a crash/other error
                    if (result.error_msg.find("PAC") != string::npos ||
                        result.error_msg.find("privacy") != string::npos ||
                        result.error_msg.find("aggregat") != string::npos ||
                        result.error_msg.find("protected") != string::npos) {
                        pac_rejected_count++;
                        Log(string("Q") + std::to_string(qnum) + " run " + std::to_string(run) +
                            " REJECTED by PAC: " + result.error_msg);
                    } else {
                        pac_crash_count++;
                        Log(string("Q") + std::to_string(qnum) + " run " + std::to_string(run) +
                            " CRASH/ERROR: " + result.error_msg);
                    }
                }

                csv << qnum << ",PAC," << run << "," << FormatNumber(time_ms) << ","
                    << (result.success ? "true" : "false") << ",\""
                    << (result.error_msg.empty() ? "" : result.error_msg) << "\"\n";
            }
        }

        csv.close();
        Log(string("Results written to: ") + actual_out);

        // =====================================================================
        // Post-process: If any PAC run of a query has "sample diversity" error,
        // mark all runs of that query as failed
        // =====================================================================
        std::set<int> unstable_queries;
        for (const auto &r : all_results) {
            if (r.mode == "PAC" && !r.success) {
                if (r.error_msg.find("sample diversity") != string::npos) {
                    unstable_queries.insert(r.query_num);
                }
            }
        }
        if (!unstable_queries.empty()) {
            Log("Detected unstable queries (sample diversity errors), marking all their runs as failed:");
            for (int qnum : unstable_queries) {
                Log(string("  Q") + std::to_string(qnum));
            }
            // Mark all PAC runs of these queries as failed
            for (auto &r : all_results) {
                if (r.mode == "PAC" && unstable_queries.count(r.query_num) > 0) {
                    if (r.success) {
                        r.success = false;
                        r.error_msg = "Marked failed due to sample diversity instability in other runs";
                    }
                }
            }
        }

        // =====================================================================
        // Print Statistics
        // =====================================================================
        Log("=== Benchmark Statistics ===");

        int total_queries = static_cast<int>(queries.size());

        // Count baseline successes and compute average time per run
        int baseline_success = 0;
        int baseline_failed = 0;
        double baseline_total_time = 0;
        for (const auto &r : all_results) {
            if (r.mode == "baseline") {
                if (r.success) {
                    baseline_success++;
                    baseline_total_time += r.time_ms;
                } else {
                    baseline_failed++;
                }
            }
        }

        // PAC stats with error message counting
        int pac_success = 0;
        int pac_rejected = 0;
        int pac_crashed = 0;
        double pac_total_time = 0;
        std::map<string, int> error_counts;  // Count occurrences of each error type

        for (const auto &r : all_results) {
            if (r.mode == "PAC") {
                if (r.success) {
                    pac_success++;
                    pac_total_time += r.time_ms;
                } else {
                    // Categorize error and count
                    string error_category;
                    if (r.error_msg.find("sample diversity") != string::npos) {
                        error_category = "sample diversity instability";
                        pac_rejected++;
                    } else if (r.error_msg.find("protected column") != string::npos) {
                        error_category = "protected column access";
                        pac_rejected++;
                    } else if (r.error_msg.find("pac_min not implemented") != string::npos ||
                               r.error_msg.find("pac_max not implemented") != string::npos) {
                        error_category = "pac_min/max not implemented for VARCHAR";
                        pac_crashed++;
                    } else if (r.error_msg.find("PAC") != string::npos ||
                               r.error_msg.find("privacy") != string::npos ||
                               r.error_msg.find("aggregat") != string::npos) {
                        error_category = "other PAC rejection";
                        pac_rejected++;
                    } else {
                        error_category = "other error";
                        pac_crashed++;
                    }
                    error_counts[error_category]++;
                }
            }
        }

        // Compute averages per run (3 runs)
        int num_runs = 3;
        double baseline_avg_success_per_run = static_cast<double>(baseline_success) / num_runs;
        double baseline_avg_failed_per_run = static_cast<double>(baseline_failed) / num_runs;
        double baseline_avg_time_per_run = baseline_total_time / num_runs;

        double pac_avg_success_per_run = static_cast<double>(pac_success) / num_runs;
        double pac_avg_rejected_per_run = static_cast<double>(pac_rejected) / num_runs;
        double pac_avg_crashed_per_run = static_cast<double>(pac_crashed) / num_runs;
        double pac_avg_time_per_run = pac_total_time / num_runs;

        Log(string("Total queries: ") + std::to_string(total_queries));
        Log(string("Number of runs: ") + std::to_string(num_runs));

        Log("--- Baseline (per run average) ---");
        Log(string("  Successful queries: ") + FormatNumber(baseline_avg_success_per_run));
        Log(string("  Failed queries: ") + FormatNumber(baseline_avg_failed_per_run));
        Log(string("  Total time: ") + FormatNumber(baseline_avg_time_per_run) + " ms");
        if (baseline_success > 0) {
            Log(string("  Avg time per successful query: ") +
                FormatNumber(baseline_total_time / baseline_success) + " ms");
        }

        Log("--- PAC (per run average) ---");
        Log(string("  Successful queries: ") + FormatNumber(pac_avg_success_per_run));
        Log(string("  Rejected (privacy violations): ") + FormatNumber(pac_avg_rejected_per_run));
        Log(string("  Crashed/errors: ") + FormatNumber(pac_avg_crashed_per_run));
        Log(string("  Total time (successful): ") + FormatNumber(pac_avg_time_per_run) + " ms");
        if (pac_success > 0) {
            Log(string("  Avg time per successful query: ") +
                FormatNumber(pac_total_time / pac_success) + " ms");
        }

        // Print error category breakdown (averaged per run)
        if (!error_counts.empty()) {
            Log("--- Error breakdown (per run average) ---");
            for (const auto &ec : error_counts) {
                double avg_count = static_cast<double>(ec.second) / num_runs;
                Log(string("  ") + ec.first + ": " + FormatNumber(avg_count));
            }
        }

        // Compute overhead
        if (baseline_success > 0 && pac_success > 0) {
            double baseline_avg_per_query = baseline_total_time / baseline_success;
            double pac_avg_per_query = pac_total_time / pac_success;
            double overhead = (pac_avg_per_query / baseline_avg_per_query - 1.0) * 100.0;
            Log(string("--- PAC Overhead: ") + FormatNumber(overhead) + "% ---");
        }

        Log("=== Benchmark Complete ===");
        return 0;

    } catch (std::exception &ex) {
        Log(string("Error running benchmark: ") + ex.what());
        return 2;
    }
}

} // namespace duckdb

// Helper for printing usage
static void PrintUsageMain() {
    std::cout << "Usage: pac_clickhouse_benchmark [options]\n"
              << "Options:\n"
              << "  --micro           Run with a smaller dataset (5000000 rows) for quick testing\n"
              << "                    Uses clickbench_micro.db by default\n"
              << "  --db <path>       DuckDB database file (default: clickbench.db or clickbench_micro.db)\n"
              << "  --queries <dir>   Directory containing create.sql, load.sql, queries.sql\n"
              << "                    (default: benchmark/clickbench_queries)\n"
              << "  --out <csv>       Output CSV path (default: benchmark/clickbench_results.csv)\n"
              << "  -h, --help        Show this help message\n";
}

int main(int argc, char **argv) {
    // Parse arguments
    bool micro = false;
    std::string db_path;
    std::string queries_dir = "benchmark/clickbench_queries";
    std::string out_csv;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            PrintUsageMain();
            return 0;
        } else if (arg == "--micro") {
            micro = true;
        } else if (arg == "--db" && i + 1 < argc) {
            db_path = argv[++i];
        } else if (arg == "--queries" && i + 1 < argc) {
            queries_dir = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_csv = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            PrintUsageMain();
            return 1;
        }
    }

    return duckdb::RunClickHouseBenchmark(db_path, queries_dir, out_csv, micro);
}
