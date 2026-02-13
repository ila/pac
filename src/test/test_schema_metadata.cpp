// Test runner for schema metadata functions (FindPrimaryKey, FindForeignKeys, FindForeignKeyBetween)

#include <iostream>

#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "../include/utils/pac_helpers.hpp"
#include "include/test_schema_metadata.hpp"

namespace duckdb {

static bool EqualVectors(vector<string> &a, vector<string> &b) {
	if (a.size() != b.size()) {
		return false;
	}
	for (size_t i = 0; i < a.size(); ++i) {
		if (a[i] != b[i]) {
			return false;
		}
	}
	return true;
}

// Helper to strip schema qualification if present
static string Basename(const string &qname) {
	auto pos = qname.rfind('.');
	if (pos == string::npos) {
		return qname;
	}
	return qname.substr(pos + 1);
}

int RunSchemaMetadataTests() {
	DuckDB db(nullptr);
	Connection con(db);
	con.BeginTransaction();

	int failures = 0;

	try {
		std::cerr << "=== Testing FindPrimaryKey ===\n";

		// Test 1: no primary key
		con.Query("CREATE TABLE IF NOT EXISTS t_no_pk(a INTEGER, b INTEGER);");
		auto pk1 = FindPrimaryKey(*con.context, "t_no_pk");
		if (!pk1.empty()) {
			std::cerr << "FAIL: expected no PK for t_no_pk\n";
			failures++;
		} else {
			std::cerr << "PASS: t_no_pk has no PK\n";
		}

		// Test 2: single-column primary key
		con.Query("CREATE TABLE IF NOT EXISTS t_single_pk(id INTEGER PRIMARY KEY, val INTEGER);");
		auto pk2 = FindPrimaryKey(*con.context, "t_single_pk");
		vector<string> expect2 = {"id"};
		if (!EqualVectors(pk2, expect2)) {
			std::cerr << "FAIL: expected PK [id] for t_single_pk\n";
			failures++;
		} else {
			std::cerr << "PASS: t_single_pk PK==[id]\n";
		}

		// Test 3: multi-column primary key
		con.Query("CREATE TABLE IF NOT EXISTS t_multi_pk(a INTEGER, b INTEGER, c INTEGER, PRIMARY KEY(a, b));");
		auto pk3 = FindPrimaryKey(*con.context, "t_multi_pk");
		vector<string> expect3 = {"a", "b"};
		if (!EqualVectors(pk3, expect3)) {
			std::cerr << "FAIL: expected PK [a,b] for t_multi_pk\n";
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
			std::cerr << "FAIL: expected PK [x] for myschema.t_schema_pk\n";
			failures++;
		} else {
			std::cerr << "PASS: myschema.t_schema_pk PK==[x]\n";
		}

		// Test 5: string (TEXT) primary key should be ignored
		con.Query("CREATE TABLE IF NOT EXISTS t_string_pk(id TEXT PRIMARY KEY, val INTEGER);");
		auto pk5 = FindPrimaryKey(*con.context, "t_string_pk");
		if (!pk5.empty()) {
			std::cerr << "FAIL: expected no PK for t_string_pk (TEXT PK should be ignored)\n";
			failures++;
		} else {
			std::cerr << "PASS: t_string_pk TEXT PK correctly ignored\n";
		}

		// Test 6: composite primary key with mixed types should be treated as no PK
		con.Query("CREATE TABLE IF NOT EXISTS t_mixed_pk(a INTEGER, b TEXT, PRIMARY KEY(a, b));");
		auto pk6 = FindPrimaryKey(*con.context, "t_mixed_pk");
		if (!pk6.empty()) {
			std::cerr << "FAIL: expected no PK for t_mixed_pk\n";
			failures++;
		} else {
			std::cerr << "PASS: t_mixed_pk composite mixed-type PK correctly treated as no PK\n";
		}

		std::cerr << "\n=== Testing FindForeignKeyBetween ===\n";

		// Test 7: transitive FK detection (t_a -> t_b -> t_c)
		con.Query("CREATE TABLE IF NOT EXISTS t_c(id INTEGER PRIMARY KEY, val INTEGER);");
		con.Query("CREATE TABLE IF NOT EXISTS t_b(id INTEGER PRIMARY KEY, cid INTEGER, FOREIGN KEY(cid) REFERENCES "
		          "t_c(id));");
		con.Query("CREATE TABLE IF NOT EXISTS t_a(id INTEGER PRIMARY KEY, bid INTEGER, FOREIGN KEY(bid) REFERENCES "
		          "t_b(id));");
		con.Query("CREATE TABLE IF NOT EXISTS t_unrelated(x INTEGER);");

		auto privacy_units = vector<string> {"t_c"};
		auto table_names = vector<string> {"t_a", "t_b", "t_unrelated"};
		auto paths = FindForeignKeyBetween(*con.context, privacy_units, table_names);

		// Expect t_a -> [t_a, t_b, t_c]
		auto it_a = paths.find("t_a");
		if (it_a == paths.end()) {
			std::cerr << "FAIL: expected path for t_a but none found\n";
			failures++;
		} else {
			auto &path = it_a->second;
			if (path.size() != 3 || Basename(path[0]) != "t_a" || Basename(path[1]) != "t_b" ||
			    Basename(path[2]) != "t_c") {
				std::cerr << "FAIL: unexpected path for t_a\n";
				failures++;
			} else {
				std::cerr << "PASS: t_a -> t_b -> t_c detected\n";
			}
		}

		// Expect t_b -> [t_b, t_c]
		auto it_b = paths.find("t_b");
		if (it_b == paths.end()) {
			std::cerr << "FAIL: expected path for t_b but none found\n";
			failures++;
		} else {
			auto &path = it_b->second;
			if (path.size() != 2 || Basename(path[0]) != "t_b" || Basename(path[1]) != "t_c") {
				std::cerr << "FAIL: unexpected path for t_b\n";
				failures++;
			} else {
				std::cerr << "PASS: t_b -> t_c detected\n";
			}
		}

		// Expect t_unrelated not present
		if (paths.find("t_unrelated") != paths.end()) {
			std::cerr << "FAIL: unexpected path found for t_unrelated\n";
			failures++;
		} else {
			std::cerr << "PASS: no path for t_unrelated\n";
		}

		// Test 8: long FK chain
		con.Query("CREATE TABLE IF NOT EXISTS t_c_long(id INTEGER PRIMARY KEY);");
		con.Query("CREATE TABLE IF NOT EXISTS t_long_2(id INTEGER PRIMARY KEY, cid INTEGER, FOREIGN KEY(cid) "
		          "REFERENCES t_c_long(id));");
		con.Query("CREATE TABLE IF NOT EXISTS t_long_1(id INTEGER PRIMARY KEY, n2 INTEGER, FOREIGN KEY(n2) REFERENCES "
		          "t_long_2(id));");
		con.Query("CREATE TABLE IF NOT EXISTS t_long_0(id INTEGER PRIMARY KEY, n1 INTEGER, FOREIGN KEY(n1) REFERENCES "
		          "t_long_1(id));");

		auto privacy_long = vector<string> {"t_c_long"};
		auto start_long = vector<string> {"t_long_0"};
		auto paths_long = FindForeignKeyBetween(*con.context, privacy_long, start_long);
		auto it_long = paths_long.find("t_long_0");

		if (it_long == paths_long.end()) {
			std::cerr << "FAIL: expected path for t_long_0 but none found\n";
			failures++;
		} else {
			auto &path = it_long->second;
			if (path.size() != 4) {
				std::cerr << "FAIL: expected path length 4 for t_long_0, got " << path.size() << "\n";
				failures++;
			} else {
				std::cerr << "PASS: long path t_long_0 -> ... -> t_c_long detected\n";
			}
		}

		// Test 9: no FK path exists
		con.Query("CREATE TABLE IF NOT EXISTS t_unrelated2(id INTEGER PRIMARY KEY);");
		auto privacy_none = vector<string> {"t_c"};
		auto starts_none = vector<string> {"t_unrelated2"};
		auto paths_none = FindForeignKeyBetween(*con.context, privacy_none, starts_none);

		if (paths_none.find("t_unrelated2") != paths_none.end()) {
			std::cerr << "FAIL: unexpected path found for t_unrelated2\n";
			failures++;
		} else {
			std::cerr << "PASS: no path for t_unrelated2\n";
		}

	} catch (std::exception &ex) {
		con.Rollback();
		std::cerr << "Exception during tests: " << ex.what() << "\n";
		return 2;
	}

	con.Rollback();

	if (failures == 0) {
		std::cerr << "\n=== ALL SCHEMA METADATA TESTS PASSED ===\n";
		return 0;
	} else {
		std::cerr << "\n=== " << failures << " SCHEMA METADATA TEST(S) FAILED ===\n";
		return 1;
	}
}

} // namespace duckdb
