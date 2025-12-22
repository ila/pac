// filepath: /home/ila/Code/pac/src/pac_privacy_columns_test.cpp
// Small standalone test runner for FindPrimaryKey (pac_helpers)

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "include/pac_helpers.hpp"

using namespace duckdb;

static bool EqualVectors(vector<string> &a, vector<string> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// Helper to strip schema qualification if present
static std::string Basename(const std::string &qname) {
    auto pos = qname.rfind('.');
    if (pos == std::string::npos) return qname;
    return qname.substr(pos + 1);
}

int main() {
    DuckDB db(nullptr);
    Connection con(db);
	con.BeginTransaction();

    int failures = 0;

    try {
        // Test 1: no primary key
        con.Query("CREATE TABLE IF NOT EXISTS t_no_pk(a INTEGER, b INTEGER);");
        auto pk1 = FindPrimaryKey(*con.context, "t_no_pk");
        if (!pk1.empty()) {
            std::cerr << "FAIL: expected no PK for t_no_pk, got:";
            for (auto &c : pk1) std::cerr << " '" << c << "'";
            std::cerr << std::endl;
            failures++;
        } else {
            std::cerr << "PASS: t_no_pk has no PK\n";
        }

        // Test 2: single-column primary key
        con.Query("CREATE TABLE IF NOT EXISTS t_single_pk(id INTEGER PRIMARY KEY, val INTEGER);");
        auto pk2 = FindPrimaryKey(*con.context, "t_single_pk");
        vector<string> expect2 = {"id"};
        if (!EqualVectors(pk2, expect2)) {
            std::cerr << "FAIL: expected PK [id] for t_single_pk, got:";
            for (auto &c : pk2) std::cerr << " '" << c << "'";
            std::cerr << std::endl;
            failures++;
        } else {
            std::cerr << "PASS: t_single_pk PK==[id]\n";
        }

        // Test 3: multi-column primary key
        con.Query("CREATE TABLE IF NOT EXISTS t_multi_pk(a INTEGER, b INTEGER, c INTEGER, PRIMARY KEY(a, b));");
        auto pk3 = FindPrimaryKey(*con.context, "t_multi_pk");
        vector<string> expect3 = {"a", "b"};
        if (!EqualVectors(pk3, expect3)) {
            std::cerr << "FAIL: expected PK [a,b] for t_multi_pk, got:";
            for (auto &c : pk3) std::cerr << " '" << c << "'";
            std::cerr << std::endl;
            failures++;
        } else {
            std::cerr << "PASS: t_multi_pk PK==[a,b]\n";
        }

        // Test 4: schema-qualified lookup
        con.Query("CREATE SCHEMA IF NOT EXISTS myschema;");
        con.Query("CREATE TABLE IF NOT EXISTS myschema.t_schema_pk(x INTEGER PRIMARY KEY, y INTEGER);");
        auto pk4 = FindPrimaryKey(*con.context, string("myschema.t_schema_pk"));
        vector<string> expect4 = {"x"};
        if (!EqualVectors(pk4, expect4)) {
            std::cerr << "FAIL: expected PK [x] for myschema.t_schema_pk, got:";
            for (auto &c : pk4) std::cerr << " '" << c << "'";
            std::cerr << std::endl;
            failures++;
        } else {
            std::cerr << "PASS: myschema.t_schema_pk PK==[x]\n";
        }

        // --- New FK transitive tests start here ---
        // Test 5: transitive FK detection (t_a -> t_b -> t_c)
        con.Query("CREATE TABLE IF NOT EXISTS t_c(id INTEGER PRIMARY KEY, val INTEGER);");
        con.Query("CREATE TABLE IF NOT EXISTS t_b(id INTEGER PRIMARY KEY, cid INTEGER, FOREIGN KEY(cid) REFERENCES t_c(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_a(id INTEGER PRIMARY KEY, bid INTEGER, FOREIGN KEY(bid) REFERENCES t_b(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_unrelated(x INTEGER);");

        std::vector<std::string> privacy_units = {"t_c"};
        std::vector<std::string> table_names = {"t_a", "t_b", "t_unrelated"};

        auto paths = FindForeignKeyBetween(*con.context, privacy_units, table_names);

        // Expect t_a -> [t_a, t_b, t_c]
        auto it_a = paths.find("t_a");
        if (it_a == paths.end()) {
            std::cerr << "FAIL: expected path for t_a but none found\n";
            failures++;
        } else {
            auto &path = it_a->second;
            if (path.size() != 3) {
                std::cerr << "FAIL: expected path length 3 for t_a, got " << path.size() << "\n";
                failures++;
            } else {
                if (Basename(path[0]) != "t_a" || Basename(path[1]) != "t_b" || Basename(path[2]) != "t_c") {
                    std::cerr << "FAIL: unexpected path for t_a: ";
                    for (auto &p : path) std::cerr << p << " ";
                    std::cerr << "\n";
                    failures++;
                } else {
                    std::cerr << "PASS: t_a -> t_b -> t_c detected\n";
                }
            }
        }

        // Expect t_b -> [t_b, t_c]
        auto it_b = paths.find("t_b");
        if (it_b == paths.end()) {
            std::cerr << "FAIL: expected path for t_b but none found\n";
            failures++;
        } else {
            auto &path = it_b->second;
            if (path.size() != 2) {
                std::cerr << "FAIL: expected path length 2 for t_b, got " << path.size() << "\n";
                failures++;
            } else {
                if (Basename(path[0]) != "t_b" || Basename(path[1]) != "t_c") {
                    std::cerr << "FAIL: unexpected path for t_b: ";
                    for (auto &p : path) std::cerr << p << " ";
                    std::cerr << "\n";
                    failures++;
                } else {
                    std::cerr << "PASS: t_b -> t_c detected\n";
                }
            }
        }

        // Expect t_unrelated not present
        if (paths.find("t_unrelated") != paths.end()) {
            std::cerr << "FAIL: unexpected path found for t_unrelated\n";
            failures++;
        } else {
            std::cerr << "PASS: no path for t_unrelated\n";
        }
        // --- FK transitive tests end here ---

        // Test 6: very long transitive FK chain (t_long_0 -> ... -> t_long_11 -> t_c_long)
        con.Query("CREATE TABLE IF NOT EXISTS t_c_long(id INTEGER PRIMARY KEY);");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_11(id INTEGER PRIMARY KEY, cid INTEGER, FOREIGN KEY(cid) REFERENCES t_c_long(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_10(id INTEGER PRIMARY KEY, n11 INTEGER, FOREIGN KEY(n11) REFERENCES t_long_11(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_9(id INTEGER PRIMARY KEY, n10 INTEGER, FOREIGN KEY(n10) REFERENCES t_long_10(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_8(id INTEGER PRIMARY KEY, n9 INTEGER, FOREIGN KEY(n9) REFERENCES t_long_9(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_7(id INTEGER PRIMARY KEY, n8 INTEGER, FOREIGN KEY(n8) REFERENCES t_long_8(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_6(id INTEGER PRIMARY KEY, n7 INTEGER, FOREIGN KEY(n7) REFERENCES t_long_7(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_5(id INTEGER PRIMARY KEY, n6 INTEGER, FOREIGN KEY(n6) REFERENCES t_long_6(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_4(id INTEGER PRIMARY KEY, n5 INTEGER, FOREIGN KEY(n5) REFERENCES t_long_5(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_3(id INTEGER PRIMARY KEY, n4 INTEGER, FOREIGN KEY(n4) REFERENCES t_long_4(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_2(id INTEGER PRIMARY KEY, n3 INTEGER, FOREIGN KEY(n3) REFERENCES t_long_3(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_1(id INTEGER PRIMARY KEY, n2 INTEGER, FOREIGN KEY(n2) REFERENCES t_long_2(id));");
        con.Query("CREATE TABLE IF NOT EXISTS t_long_0(id INTEGER PRIMARY KEY, n1 INTEGER, FOREIGN KEY(n1) REFERENCES t_long_1(id));");

        std::vector<std::string> privacy_long = {"t_c_long"};
        std::vector<std::string> start_long = {"t_long_0"};
        auto paths_long = FindForeignKeyBetween(*con.context, privacy_long, start_long);
        auto it_long = paths_long.find("t_long_0");
        if (it_long == paths_long.end()) {
            std::cerr << "FAIL: expected path for t_long_0 but none found\n";
            failures++;
        } else {
            auto &path = it_long->second;
            const size_t expected_len = 13; // t_long_0..t_long_11 + t_c_long
            if (path.size() != expected_len) {
                std::cerr << "FAIL: expected path length " << expected_len << " for t_long_0, got " << path.size() << "\n";
                failures++;
            } else {
                bool ok = true;
                for (size_t i = 0; i < 12; ++i) {
                    std::string expect = std::string("t_long_") + std::to_string(i);
                    if (Basename(path[i]) != expect) { ok = false; break; }
                }
                if (Basename(path[12]) != "t_c_long") ok = false;
                if (!ok) {
                    std::cerr << "FAIL: unexpected long path: ";
                    for (auto &p : path) std::cerr << p << " ";
                    std::cerr << "\n";
                    failures++;
                } else {
                    std::cerr << "PASS: long path t_long_0 -> ... -> t_c_long detected\n";
                }
            }
        }

        // Test 7: additional unrelated-table test
        con.Query("CREATE TABLE IF NOT EXISTS t_unrelated_base(id INTEGER PRIMARY KEY);");
        con.Query("CREATE TABLE IF NOT EXISTS t_unrelated2(id INTEGER PRIMARY KEY, bid INTEGER, FOREIGN KEY(bid) REFERENCES t_unrelated_base(id));");
        std::vector<std::string> privacy_none = {"t_c"};
        std::vector<std::string> starts_none = {"t_unrelated2"};
        auto paths_none = FindForeignKeyBetween(*con.context, privacy_none, starts_none);
        if (paths_none.find("t_unrelated2") != paths_none.end()) {
            std::cerr << "FAIL: unexpected path found for t_unrelated2\n";
            failures++;
        } else {
            std::cerr << "PASS: no path for t_unrelated2\n";
        }

    } catch (std::exception &ex) {
	con.Rollback();
        std::cerr << "Exception during tests: " << ex.what() << std::endl;
        return 2;
    }

	con.Rollback();

    if (failures == 0) {
        std::cerr << "ALL TESTS PASSED\n";
        return 0;
    } else {
        std::cerr << failures << " TEST(S) FAILED\n";
        return 1;
    }
}

