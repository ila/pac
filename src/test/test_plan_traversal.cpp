// Test runner for plan traversal functions

#include <iostream>

#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "../include/pac_plan_traversal.hpp"
#include "include/test_plan_traversal.hpp"

namespace duckdb {

int RunPlanTraversalTests() {
	DuckDB db(nullptr);
	Connection con(db);
	con.BeginTransaction();

	int failures = 0;

	try {
		std::cerr << "=== Testing Plan Traversal Functions ===\n";

		// Setup test tables
		con.Query("CREATE TABLE t_test(id INTEGER PRIMARY KEY, val INTEGER);");
		con.Query("INSERT INTO t_test VALUES (1, 100), (2, 200);");

		// Test 1: FindPrivacyUnitGetNode - find specific table
		std::cerr << "\n--- Test FindPrivacyUnitGetNode ---\n";
		Parser parser;
		parser.ParseQuery("SELECT SUM(val) FROM t_test;");
		Planner planner(*con.context);
		planner.CreatePlan(std::move(parser.statements[0]));

		Optimizer opt(*planner.binder, *con.context);
		auto plan = opt.Optimize(std::move(planner.plan));

		try {
			auto *get_node = FindPrivacyUnitGetNode(plan, "t_test");
			if (get_node && get_node->get()) {
				std::cerr << "PASS: Found t_test node\n";
			} else {
				std::cerr << "FAIL: FindPrivacyUnitGetNode returned null\n";
				failures++;
			}
		} catch (std::exception &ex) {
			std::cerr << "FAIL: Exception finding t_test: " << ex.what() << "\n";
			failures++;
		}

		// Test 2: FindPrivacyUnitGetNode - table not found should throw
		try {
			auto *get_node = FindPrivacyUnitGetNode(plan, "nonexistent_table");
			std::cerr << "FAIL: Should have thrown exception for nonexistent table\n";
			failures++;
		} catch (InternalException &) {
			std::cerr << "PASS: Correctly threw exception for nonexistent table\n";
		}

		// Test 3: FindTopAggregate - find aggregate node
		std::cerr << "\n--- Test FindTopAggregate ---\n";
		try {
			auto *agg = FindTopAggregate(plan);
			if (agg) {
				std::cerr << "PASS: Found aggregate node\n";
			} else {
				std::cerr << "FAIL: FindTopAggregate returned null\n";
				failures++;
			}
		} catch (std::exception &ex) {
			std::cerr << "FAIL: Exception finding aggregate: " << ex.what() << "\n";
			failures++;
		}

		// Test 4: FindTopAggregate - no aggregate should throw
		std::cerr << "\n--- Test FindTopAggregate (no aggregate) ---\n";
		Parser parser2;
		parser2.ParseQuery("SELECT val FROM t_test;");
		Planner planner2(*con.context);
		planner2.CreatePlan(std::move(parser2.statements[0]));
		Optimizer opt2(*planner2.binder, *con.context);
		auto plan2 = opt2.Optimize(std::move(planner2.plan));

		try {
			auto *agg = FindTopAggregate(plan2);
			std::cerr << "FAIL: Should have thrown exception when no aggregate exists\n";
			failures++;
		} catch (InternalException &) {
			std::cerr << "PASS: Correctly threw exception when no aggregate exists\n";
		}

		// Test 5: FindNodeRefByTable
		std::cerr << "\n--- Test FindNodeRefByTable ---\n";
		LogicalOperator *parent = nullptr;
		idx_t child_idx = 0;
		auto *node_ref = FindNodeRefByTable(&plan, "t_test", &parent, &child_idx);

		if (node_ref && node_ref->get()) {
			std::cerr << "PASS: FindNodeRefByTable found t_test\n";
		} else {
			std::cerr << "FAIL: FindNodeRefByTable did not find t_test\n";
			failures++;
		}

		// Test 6: FindNodeRefByTable - nonexistent table
		auto *node_ref2 = FindNodeRefByTable(&plan, "nonexistent", nullptr, nullptr);
		if (node_ref2 == nullptr) {
			std::cerr << "PASS: FindNodeRefByTable correctly returned nullptr for nonexistent table\n";
		} else {
			std::cerr << "FAIL: FindNodeRefByTable should return nullptr for nonexistent table\n";
			failures++;
		}

	} catch (std::exception &ex) {
		con.Rollback();
		std::cerr << "Exception during tests: " << ex.what() << "\n";
		return 2;
	}

	con.Rollback();

	if (failures == 0) {
		std::cerr << "\n=== ALL PLAN TRAVERSAL TESTS PASSED ===\n";
		return 0;
	} else {
		std::cerr << "\n=== " << failures << " PLAN TRAVERSAL TEST(S) FAILED ===\n";
		return 1;
	}
}

} // namespace duckdb
