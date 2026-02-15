//
// Created by ila on 12/24/25.
//

// Implement the TPCH benchmark runner.

#include "include/pac_tpch_benchmark.hpp"

#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/common/printer.hpp"
#include <cstdio>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <stdexcept>
#include <ctime>
#include <algorithm>
#include <iomanip>
#include <thread>
#include <unistd.h>
#include <limits.h>
#include <dirent.h>
#include <map>

namespace duckdb {
static string Timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return string(buf);
}

// forward declarations for helpers used below
static bool FileExists(const string &path);
static string ReadFileToString(const string &path);

// new helper: try to discover the absolute path to the R plotting script
static string FindPlotScriptAbsolute() {
    // 1) try cwd/benchmark/plot_tpch_results.R
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd))) {
        string p = string(cwd) + "/benchmark/plot_tpch_results.R";
        if (FileExists(p)) {
            char rbuf[PATH_MAX];
            if (realpath(p.c_str(), rbuf)) {
                return string(rbuf);
            }
            return p;
        }
    }
    // 2) try relative to the executable location (walk upward looking for benchmark dir)
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
    if (len != -1) {
        exe_path[len] = '\0';
        string dir = string(exe_path);
        // strip filename
        auto pos = dir.find_last_of('/');
        if (pos != string::npos) {
            dir = dir.substr(0, pos);
        }
        // walk up several levels searching for benchmark/plot_tpch_results.R
        string try_dir = dir;
        for (int i = 0; i < 6; ++i) {
            string cand = try_dir + "/benchmark/plot_tpch_results.R";
            if (FileExists(cand)) {
                char rbuf[PATH_MAX];
                if (realpath(cand.c_str(), rbuf)) {
                    return string(rbuf);
                }
                return cand;
            }
            auto p2 = try_dir.find_last_of('/');
            if (p2 == string::npos) {
                break;
            }
            try_dir = try_dir.substr(0, p2);
        }
    }
    // 3) try some common relative locations
    vector<string> rels = {"benchmark/plot_tpch_results.R", "./benchmark/plot_tpch_results.R", "../benchmark/plot_tpch_results.R", "../../benchmark/plot_tpch_results.R"};
    for (auto &r : rels) {
        if (FileExists(r)) {
            char rbuf[PATH_MAX];
            if (realpath(r.c_str(), rbuf)) {
                return string(rbuf);
            }
            return r;
        }
    }
    return string();
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

static double Median(vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) {
        return v[n / 2];
    }
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

static void Log(const string &msg) {
    Printer::Print("[" + Timestamp() + "] " + msg);
}

static bool FileExists(const string &path) {
	struct stat buffer;
	return (stat(path.c_str(), &buffer) == 0);
}

static string FindQueryFile(const string &dir, int qnum) {
	// files are named q01.sql .. q22.sql
	char buf[1024];
	snprintf(buf, sizeof(buf), "q%02d.sql", qnum);
	return dir + "/" + string(buf);
}

// Info about a discovered query file
struct QueryEntry {
	string filename;     // e.g. "q08-nolambda.sql"
	string label;        // e.g. "q08-nolambda" (for CSV output)
	int query_number;    // e.g. 8 (for PRAGMA tpch baseline)
};

// Scan a directory for q*.sql files, parse query numbers, return sorted
static vector<QueryEntry> DiscoverQueryFiles(const string &dir) {
	vector<QueryEntry> entries;
	DIR *d = opendir(dir.c_str());
	if (!d) {
		return entries;
	}
	struct dirent *ent;
	while ((ent = readdir(d)) != nullptr) {
		string name = ent->d_name;
		if (name.size() < 5 || name[0] != 'q') {
			continue;
		}
		if (name.substr(name.size() - 4) != ".sql") {
			continue;
		}
		// Parse query number from digits after 'q'
		int qnum = 0;
		size_t i = 1;
		while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
			qnum = qnum * 10 + (name[i] - '0');
			i++;
		}
		if (qnum == 0) {
			continue;
		}
		string label = name.substr(0, name.size() - 4);
		entries.push_back({name, label, qnum});
	}
	closedir(d);
	// Sort by query number, then by label length (shorter first, so q08 before q08-nolambda)
	std::sort(entries.begin(), entries.end(), [](const QueryEntry &a, const QueryEntry &b) {
		if (a.query_number != b.query_number) {
			return a.query_number < b.query_number;
		}
		return a.label < b.label;
	});
	return entries;
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

// helper: find Rscript absolute path via 'which'
static string FindRscriptAbsolute() {
    FILE *pipe = popen("which Rscript 2>/dev/null", "r");
    if (!pipe) { return string(); }
    char buf[PATH_MAX];
    string out;
    while (fgets(buf, sizeof(buf), pipe)) {
        out += string(buf);
    }
    // trim newline/whitespace
    while (!out.empty() && (out.back()=='\n' || out.back()=='\r' || out.back()==' ')) { out.pop_back(); }
    if (out.empty()) { return string(); }
    return out;
}

// Invoke the plotting script
// Returns true if a plot was successfully generated.
static bool InvokePlotScript(const string &abs_actual_out, const string &out_dir) {
    string script = FindPlotScriptAbsolute();
    if (script.empty()) { Log(string("Plot script not found (looked in several candidate locations). Skipping plotting.")); return false; }

    string rscript_path = FindRscriptAbsolute();
    if (rscript_path.empty()) {
        Log(string("Rscript executable not found on PATH. Plotting will likely fail. Please ensure R is installed and Rscript is available."));
    }
    // Call the discovered script exactly once. If Rscript is available, use it with --vanilla.
    string rcmd = rscript_path.empty() ? string("Rscript --vanilla") : (rscript_path + string(" --vanilla"));
    string cmd = rcmd + string(" \"") + script + "\" \"" + abs_actual_out + "\" \"" + out_dir + "\"";
    Log(string("Calling plot script: ") + cmd);
    string full_cmd = cmd + " 2>&1";
    FILE *pipe = popen(full_cmd.c_str(), "r");
    if (!pipe) {
        Log(string("popen failed when starting plot script."));
        Log(string("Plot script invocation failed; skipping plotting."));
        return false;
    }
    char buffer[4096];
    string output;
    while (true) {
        size_t n = fread(buffer, 1, sizeof(buffer), pipe);
        if (n > 0) { output.append(buffer, buffer + n); }
        if (n < sizeof(buffer)) { break; }
    }
    int rc = pclose(pipe);
    int exit_code = rc;
    if (rc != -1 && WIFEXITED(rc)) { exit_code = WEXITSTATUS(rc); }
    string out_log = output;
    if (out_log.size() > 4000) { out_log = out_log.substr(0, 4000) + "\n...[truncated]..."; }
    Log(string("Plot script output (truncated):\n") + out_log);
    Log(string("Plot script returned exit code: ") + std::to_string(exit_code));
    if (exit_code == 0) {
        Log(string("Plot script completed successfully."));
        return true;
    }
    if (output.find("libicu") != string::npos || output.find("stringi.so") != string::npos || output.find("libicuuc") != string::npos) {
        Log(string("Plot attempt reported missing ICU/shared library in R output. Suggest installing system ICU (for example 'libicu' or 'libicu-devel' depending on your Linux distribution) or ensuring R's native libraries are discoverable by LD_LIBRARY_PATH. You can also try running 'Rscript --vanilla <script>' in an environment where R is fully installed."));
    }
    Log(string("Plot script failed (non-zero exit). Skipping further attempts."));
    return false;
}

int RunTPCHBenchmark(const string &db_path, const string &queries_dir, double sf, const string &out_csv, bool run_naive, bool run_simple_hash, int threads) {
    try {
        Log(string("run_naive flag: ") + (run_naive ? string("true") : string("false")));
    	Log(string("run_simple_hash flag: ") + (run_simple_hash ? string("true") : string("false")));

        // Open (file-backed) DuckDB database
        // Decide whether the caller explicitly provided a DB path (not the default) so we can
        // decide whether to warn and/or re-create tables in an existing DB.
        bool user_provided_db = !(db_path.empty() || db_path == "tpch.db");
        // Compute actual DB filename: use user-provided path, otherwise derive from scale factor
        string db_actual = db_path;
        if (!user_provided_db) {
            string sf_token = FormatNumber(sf);
            sf_token.erase(std::remove(sf_token.begin(), sf_token.end(), '.'), sf_token.end());
            sf_token.erase(std::remove(sf_token.begin(), sf_token.end(), '_'), sf_token.end());
            char dbfn[256];
            snprintf(dbfn, sizeof(dbfn), "tpch_sf%s.db", sf_token.c_str());
            db_actual = string(dbfn);
        }
        bool db_exists = FileExists(db_actual);
        if (db_exists) {
            Log(string("Connecting to existing DuckDB database: ") + db_actual + string(" (skipping data generation)."));
        } else {
            Log(string("Will create/populate DuckDB database: ") + db_actual);
        }
        DuckDB db(db_actual.c_str());
        Connection con(db);

        // Set thread count
        Log("Setting threads to " + std::to_string(threads));
        con.Query("SET threads TO " + std::to_string(threads) + ";");

        // Decide output filename if empty
        string actual_out = out_csv;
        if (actual_out.empty()) {
            // sanitize scale factor into a string (remove '.' and '_')
            string sf_token = FormatNumber(sf);
            sf_token.erase(std::remove(sf_token.begin(), sf_token.end(), '.'), sf_token.end());
            sf_token.erase(std::remove(sf_token.begin(), sf_token.end(), '_'), sf_token.end());
            char ofn[256];
            // use 'tpch_benchmark_results' (expanded 'benchmark')
            snprintf(ofn, sizeof(ofn), "benchmark/tpch_benchmark_results_sf%s.csv", sf_token.c_str());
            actual_out = string(ofn);
        }

        Log("Installing TPCH extension...");
        auto r_install = con.Query("INSTALL tpch;");
        if (r_install && r_install->HasError()) {
            Log(string("INSTALL tpch error: ") + r_install->GetError());
        }
        auto r_load = con.Query("LOAD tpch;");
        if (r_load && r_load->HasError()) {
            Log(string("LOAD tpch error: ") + r_load->GetError());
        }

        Log("Generating TPCH data (sf=" + FormatNumber(sf) + ")... this may take a while");
        // call dbgen with requested scale factor (may be fractional):
        // - If the DB file doesn't exist -> create it by calling dbgen
        // - If the DB file exists and the user explicitly provided a path -> warn and re-run dbgen (may recreate tables)
        // - If the DB file exists and the DB was derived from sf (user didn't provide a path) -> skip dbgen
        if (!db_exists) {
            // First create schema with PK/FK constraints
            Log("Creating TPC-H schema with constraints...");
            string schema_file = FindSchemaFile("pac_tpch_schema.sql");
            if (schema_file.empty()) {
                throw std::runtime_error("Cannot find pac_tpch_schema.sql");
            }

        	/*
            std::ifstream schema_ifs(schema_file);
            if (!schema_ifs.is_open()) {
                throw std::runtime_error("Cannot open pac_tpch_schema.sql: " + schema_file);
            }
            std::stringstream schema_ss;
            schema_ss << schema_ifs.rdbuf();
            string schema_sql = schema_ss.str();

            auto schema_result = con.Query(schema_sql);
            if (schema_result->HasError()) {
                throw std::runtime_error("Failed to create schema: " + schema_result->GetError());
            }
            */

            // Now generate data
            char callbuf[128];
            snprintf(callbuf, sizeof(callbuf), "CALL dbgen(sf=%g);", sf);
            auto r_dbgen = con.Query(callbuf);
            if (r_dbgen && r_dbgen->HasError()) {
                Log(string("CALL dbgen error: ") + r_dbgen->GetError());
            }
        } else {
            Log(string("Skipping CALL dbgen since database file already exists: ") + db_actual);
        }

        // Prepare CSV output (overwrite if exists)
        std::ofstream csv(actual_out, std::ofstream::out | std::ofstream::trunc);
        if (!csv.is_open()) {
            throw std::runtime_error("Failed to open output CSV: " + actual_out);
        }
        // CSV columns: query,mode,median_ms (median of 5 hot runs)
        csv << "query,mode,median_ms\n";

    	// Locate PAC query directories
    	string bitslice_dir = queries_dir + string("/tpch_pac_queries");
    	string naive_dir = queries_dir + string("/tpch_pac_naive_queries");
    	string simple_hash_dir = queries_dir + string("/tpch_pac_simple_hash_queries");

    	// Discover all .sql files in the bitslice directory
    	auto query_entries = DiscoverQueryFiles(bitslice_dir);
    	if (query_entries.empty()) {
    		throw std::runtime_error("No query files found in " + bitslice_dir);
    	}
    	Log("Discovered " + std::to_string(query_entries.size()) + " query files in " + bitslice_dir);
    	for (auto &e : query_entries) {
    		Log("  " + e.label + " (baseline Q" + std::to_string(e.query_number) + ")");
    	}

    	// Cache baseline median per query number (avoid re-running for variants like q08-nolambda)
    	std::map<int, double> baseline_cache;

        for (auto &entry : query_entries) {
            Log("=== " + entry.label + " (Q" + std::to_string(entry.query_number) + ") ===");

            // Run baseline (PRAGMA tpch) if not already cached for this query number
            if (baseline_cache.find(entry.query_number) == baseline_cache.end()) {
                string pragma = "PRAGMA tpch(" + std::to_string(entry.query_number) + ");";

                // Cold run (do not time)
                {
                    auto r_cold = con.Query(pragma);
                    if (r_cold && r_cold->HasError()) {
                        Log(string("Cold PRAGMA tpch(") + std::to_string(entry.query_number) + ") error: " + r_cold->GetError());
                    }
                    Log("Cold run completed for TPCH query " + std::to_string(entry.query_number));
                }

                // 1st warm run (do not time) to warm caches
                {
                    auto r_warm_init = con.Query(pragma);
                    if (r_warm_init && r_warm_init->HasError()) {
                        Log(string("Warm (init) PRAGMA tpch(") + std::to_string(entry.query_number) + ") error: " + r_warm_init->GetError());
                    }
                    Log("Warm (init) run completed for TPCH query " + std::to_string(entry.query_number));
                }

                // Wait 0.5s before timed hot runs
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                // Hot runs: 5 runs, take median
                vector<double> tpch_times_ms;
                for (int t = 1; t <= 5; ++t) {
                    auto t0 = std::chrono::steady_clock::now();
                    auto r_warm = con.Query(pragma);
                    auto t1 = std::chrono::steady_clock::now();
                    double tpch_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    tpch_times_ms.push_back(tpch_time_ms);
                    Log("TPCH query " + std::to_string(entry.query_number) + " hot run " + std::to_string(t) + " time (ms): " + FormatNumber(tpch_time_ms));
                    if (r_warm && r_warm->HasError()) {
                        Log(string("TPCH hot run error for query ") + std::to_string(entry.query_number) + ": " + r_warm->GetError());
                    }
                }
                double baseline_median = Median(tpch_times_ms);
                Log("TPCH query " + std::to_string(entry.query_number) + " median (ms): " + FormatNumber(baseline_median));
                baseline_cache[entry.query_number] = baseline_median;
            }

            // Write baseline row for this entry (reuse cached median)
            csv << entry.label << ",baseline," << FormatNumber(baseline_cache[entry.query_number]) << "\n";

            // Read the PAC query file
            string pac_sql_bits = ReadFileToString(bitslice_dir + "/" + entry.filename);
            if (pac_sql_bits.empty()) {
                throw std::runtime_error("PAC query file " + entry.filename + " is empty or unreadable");
            }

            // Build variant list: bitslice always; naive and simple_hash if requested and file exists
            vector<pair<string,string>> pac_variants;
            pac_variants.emplace_back("SIMD PAC", pac_sql_bits);
            if (run_naive) {
                string naive_qfile = FindQueryFile(naive_dir, entry.query_number);
                if (FileExists(naive_qfile)) {
                    string sql = ReadFileToString(naive_qfile);
                    if (!sql.empty()) {
                        pac_variants.emplace_back("naive PAC", sql);
                    }
                }
            }
            if (run_simple_hash) {
                string sh_qfile = FindQueryFile(simple_hash_dir, entry.query_number);
                if (FileExists(sh_qfile)) {
                    string sql = ReadFileToString(sh_qfile);
                    if (!sql.empty()) {
                        pac_variants.emplace_back("simple hash PAC", sql);
                    }
                }
            }

            for (auto &pv : pac_variants) {
                const string &mode_str = pv.first;
                const string &pac_sql = pv.second;
                // Run PAC query 5 times, take median
                vector<double> pac_times_ms;
                for (int run = 1; run <= 5; ++run) {
                    auto t0 = std::chrono::steady_clock::now();
                    auto r_pac = con.Query(pac_sql);
                    auto t1 = std::chrono::steady_clock::now();
                    double pac_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (r_pac && r_pac->HasError()) {
                        throw std::runtime_error("PAC (" + mode_str + ") " + entry.label + " run " + std::to_string(run) + " error: " + r_pac->GetError());
                    }
                    pac_times_ms.push_back(pac_time_ms);
                    Log("PAC (" + mode_str + ") " + entry.label + " run " + std::to_string(run) + " time (ms): " + FormatNumber(pac_time_ms));
                }
                double pac_median = Median(pac_times_ms);
                Log("PAC (" + mode_str + ") " + entry.label + " median (ms): " + FormatNumber(pac_median));
                csv << entry.label << "," << mode_str << "," << FormatNumber(pac_median) << "\n";
            }
        }

         csv.close();
         Log(string("Benchmark finished. Results written to ") + actual_out);

        // Automatically call the R plotting script with the generated CSV file (use absolute paths)
        {
            // compute absolute path to actual_out
            char out_real[PATH_MAX];
            string abs_actual_out;
            if (realpath(actual_out.c_str(), out_real)) {
                abs_actual_out = string(out_real);
            } else {
                // fallback: if actual_out is relative, prefix cwd
                char cwd2[PATH_MAX];
                if (getcwd(cwd2, sizeof(cwd2))) {
                    abs_actual_out = string(cwd2) + "/" + actual_out;
                } else {
                    abs_actual_out = actual_out; // last resort
                }
            }
            // derive out_dir from abs_actual_out
            size_t pos = abs_actual_out.find_last_of('/');
            string out_dir = (pos == string::npos) ? string(".") : abs_actual_out.substr(0, pos);

            InvokePlotScript(abs_actual_out, out_dir);
        }

        return 0;
    } catch (std::exception &ex) {
        Log(string("Error running benchmark: ") + ex.what());
        return 2;
    }
}

} // namespace duckdb

// Add a small helper for printing usage (placed outside of namespace to avoid analyzer warnings)
static void PrintUsageMain() {
     std::cout << "Usage: pac_tpch_benchmark [sf] [db_path] [queries_dir] [out_csv] [--run-naive] [--run-simple-hash]\n"
               << "  sf: TPCH scale factor (int, default 10)\n"
               << "  db_path: DuckDB database file (default 'tpch.db')\n"
               << "  queries_dir: root directory containing PAC SQL variants (default 'benchmark').\n"
               << "               subdirectories expected: 'tpch_pac_queries' (bitslice), 'tpch_pac_naive_queries' (naive), 'tpch_pac_simple_hash_queries' (simple hash)\n"
               << "  out_csv: optional output CSV path (auto-named if omitted)\n"
               << "  --run-naive: optional flag to instruct the benchmark to run a naive PAC variant as well\n"
               << "  --run-simple-hash: optional flag to instruct the benchmark to run a simple hash PAC variant as well\n"
               << "  --threads=N: number of DuckDB threads (default 8)\n";
}

// Add a small main so this file builds to an executable
int main(int argc, char **argv) {
    // quick arg validation: help or too many args
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help") {
            PrintUsageMain();
            return 0;
        }
    }
    if (argc > 7) {
        std::cout << "Error: too many arguments provided." << '\n';
        PrintUsageMain();
        return 1;
    }

    // Preprocess argv to detect optional flags and remove them from positional parsing
    bool run_naive = false;
    bool run_simple_hash = false;
    int threads = 8;
    std::vector<char*> filtered_argv;
    filtered_argv.reserve(argc);
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--run-naive") {
            run_naive = true;
            continue;
        }
        if (a == "--run-simple-hash") {
            run_simple_hash = true;
            continue;
        }
        if (a.rfind("--threads=", 0) == 0) {
            threads = std::stoi(a.substr(10));
            continue;
        }
        filtered_argv.push_back(argv[i]);
    }
    int filtered_argc = static_cast<int>(filtered_argv.size());

    // Parse positional arguments as before
    double sf = 10;
    std::string db_path = "tpch.db";
    std::string queries_dir = "benchmark";
    std::string out_csv;
    if (filtered_argc > 1) sf = std::stod(filtered_argv[1]);
    if (filtered_argc > 2) db_path = filtered_argv[2];
    if (filtered_argc > 3) queries_dir = filtered_argv[3];
    if (filtered_argc > 4) out_csv = filtered_argv[4];

    return duckdb::RunTPCHBenchmark(db_path, queries_dir, sf, out_csv, run_naive, run_simple_hash, threads);
}
