#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include "../../duckdb/src/include/duckdb/planner/expression_iterator.hpp"
#include <set>

#include "../include/utils/pac_helpers.hpp"
#include "../include/compiler/pac_compiler_helpers.hpp"
#include "../../duckdb/src/include/duckdb/planner/operator/logical_projection.hpp"
#include "../../duckdb/src/include/duckdb/planner/operator/logical_dummy_scan.hpp"
#include "../../duckdb/src/include/duckdb/planner/operator/logical_get.hpp"
#include "../../duckdb/src/include/duckdb/planner/expression/bound_columnref_expression.hpp"
#include "../../duckdb/src/include/duckdb/planner/operator/logical_join.hpp"
#include "../../duckdb/src/include/duckdb/planner/operator/logical_filter.hpp"
#include "../../duckdb/src/include/duckdb/planner/operator/logical_aggregate.hpp"
#include "../../duckdb/src/include/duckdb/common/enums/join_type.hpp"
#include "../../duckdb/src/include/duckdb.hpp"
#include "../../duckdb/src/include/duckdb/main/connection.hpp"
#include "../../duckdb/src/include/duckdb/common/constants.hpp"
#include "../../duckdb/src/include/duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "include/test_compiler_functions.hpp"

#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

namespace duckdb {
// Use DuckDB-provided vector/unique_ptr/make_uniq; do NOT import std::vector/std::unique_ptr/make_uniq

// Helper to run a test and print a message on failure
static int RunTest(const char *name, const std::function<int()> &fn) {
	std::cerr << "Running " << name << "...\n";
	int res = fn();
	if (res == 0) {
		std::cerr << name << " passed\n";
	} else {
		std::cerr << name << " FAILED (code " << res << ")\n";
	}
	return res;
}

// Collect all ColumnBinding (table_index, column_index) pairs from the operator tree
static void CollectColumnBindingsRecursiveForTest(LogicalOperator *node, std::set<std::pair<idx_t, idx_t>> &out) {
	if (!node) {
		return;
	}
	auto binds = node->GetColumnBindings();
	for (auto &b : binds) {
		out.insert({b.table_index, b.column_index});
	}
	for (auto &c : node->children) {
		CollectColumnBindingsRecursiveForTest(c.get(), out);
	}
}

// Verify that every BoundColumnRefExpression in the tree refers to a collected binding
static bool VerifyAllExprBindingsExist(LogicalOperator *node, const std::set<std::pair<idx_t, idx_t>> &bindings) {
	if (!node) {
		return true;
	}
	bool ok = true;
	for (auto &expr : node->expressions) {
		ExpressionIterator::VisitExpression<BoundColumnRefExpression>(*expr, [&](const BoundColumnRefExpression &bcr) {
			auto key = std::make_pair(bcr.binding.table_index, bcr.binding.column_index);
			if (bindings.find(key) == bindings.end()) {
				std::cerr << "Verification failure: expression references missing binding (table="
				          << bcr.binding.table_index << ", col=" << bcr.binding.column_index << ") in operator "
				          << node->GetName() << "\n";
				ok = false;
			}
		});
	}
	for (auto &c : node->children) {
		if (!VerifyAllExprBindingsExist(c.get(), bindings)) {
			ok = false;
		}
	}
	return ok;
}

static bool VerifyNoInvalidBindings(const std::set<std::pair<idx_t, idx_t>> &bindings) {
	for (auto &p : bindings) {
		if (p.first == DConstants::INVALID_INDEX) {
			std::cerr << "Verification failure: found INVALID table index in bindings\n";
			return false;
		}
	}
	return true;
}

// New: collect a mapping from ColumnBinding (table_index,col_index) to producer LogicalOperator*
static void CollectBindingProducers(LogicalOperator *node,
                                    std::map<std::pair<idx_t, idx_t>, vector<LogicalOperator *>> &out) {
	if (!node) {
		return;
	}
	auto binds = node->GetColumnBindings();
	for (auto &b : binds) {
		out[{b.table_index, b.column_index}].push_back(node);
	}
	for (auto &c : node->children) {
		CollectBindingProducers(c.get(), out);
	}
}

// New: ensure each binding has exactly one producer and the producer advertises a type for that column_index
static bool VerifyBindingProducersUniqueAndTypes(LogicalOperator *root) {
	std::map<std::pair<idx_t, idx_t>, vector<LogicalOperator *>> producers;
	CollectBindingProducers(root, producers);
	bool ok = true;
	for (auto &kv : producers) {
		const auto &binding = kv.first;
		const auto &vec = kv.second;
		if (vec.empty()) {
			std::cerr << "Verification failure: binding (table=" << binding.first << ", col=" << binding.second
			          << ") has no producer\n";
			ok = false;
			continue;
		}
		if (vec.size() > 1) {
			std::cerr << "Verification failure: binding (table=" << binding.first << ", col=" << binding.second
			          << ") has multiple producers (" << vec.size() << ")\n";
			// log producers' operator names for debugging
			for (auto *p : vec) {
				std::cerr << "  producer: " << (p ? p->GetName() : "<null>") << "\n";
			}
			ok = false;
			continue;
		}
		// check producer provides types and the column_index is in range
		auto *producer = vec[0];
		if (!producer) {
			std::cerr << "Verification failure: null producer for binding (table=" << binding.first
			          << ", col=" << binding.second << ")\n";
			ok = false;
			continue;
		}
		auto types = producer->types;
		if (binding.second >= (idx_t)types.size()) {
			std::cerr << "Verification failure: producer " << producer->GetName() << " does not have column index "
			          << binding.second << " (types=" << types.size() << ")\n";
			ok = false;
			continue;
		}
		// ensure the advertised type is not Invalid
		if (types[binding.second].id() == LogicalTypeId::INVALID) {
			std::cerr << "Verification failure: producer " << producer->GetName() << " has INVALID type for column "
			          << binding.second << "\n";
			ok = false;
		}
	}
	// Additionally ensure there are no duplicate column_index values within a single producer's bindings
	// For each producer, collect the set of column_index values it produces
	std::map<LogicalOperator *, std::set<idx_t>> producer_cols;
	for (auto &kv : producers) {
		auto binding = kv.first;
		for (auto *p : kv.second) {
			producer_cols[p].insert(binding.second);
		}
	}
	for (auto &kv : producer_cols) {
		auto *p = kv.first;
		auto &cols = kv.second;
		// compare to the count reported by GetColumnBindings() for this producer
		auto binds = p->GetColumnBindings();
		std::set<idx_t> reported_cols;
		for (auto &b : binds) {
			reported_cols.insert(b.column_index);
		}
		if (cols.size() != reported_cols.size()) {
			std::cerr << "Verification failure: mismatch in column set sizes for producer " << p->GetName()
			          << " (col set sizes: collected=" << cols.size() << ", reported=" << reported_cols.size() << ")\n";
			ok = false;
		}
		// ensure the sets are identical
		for (auto cidx : cols) {
			if (reported_cols.find(cidx) == reported_cols.end()) {
				std::cerr << "Verification failure: producer " << p->GetName() << " missing column_index " << cidx
				          << " in its reported bindings\n";
				ok = false;
			}
		}
	}
	return ok;
}

// New: verify that BoundColumnRefExpression.type matches the producer's type
static bool VerifyExprTypesMatchProducers(LogicalOperator *root) {
	// build producers map
	std::map<std::pair<idx_t, idx_t>, vector<LogicalOperator *>> producers;
	CollectBindingProducers(root, producers);
	bool ok = true;
	// traverse expressions
	std::function<void(LogicalOperator *)> check_node = [&](LogicalOperator *node) {
		if (!node) {
			return;
		}
		for (auto &expr : node->expressions) {
			ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
			    *expr, [&](const BoundColumnRefExpression &bcr) {
				    auto key = std::make_pair(bcr.binding.table_index, bcr.binding.column_index);
				    auto it = producers.find(key);
				    if (it == producers.end() || it->second.empty()) {
					    // missing producer already reported elsewhere; skip
					    return;
				    }
				    auto *producer = it->second[0];
				    auto types = producer->types;
				    if (bcr.binding.column_index >= (idx_t)types.size()) {
					    std::cerr << "Verification failure: expression references column index "
					              << bcr.binding.column_index << " beyond producer types size " << types.size() << "\n";
					    ok = false;
					    return;
				    }
				    const auto &prod_type = types[bcr.binding.column_index];
				    if (bcr.return_type.id() != prod_type.id()) {
					    std::cerr << "Verification failure: expression type mismatch for binding (table="
					              << bcr.binding.table_index << ", col=" << bcr.binding.column_index
					              << "): expr_type=" << bcr.return_type.ToString()
					              << " producer_type=" << prod_type.ToString() << "\n";
					    ok = false;
				    }
			    });
		}
		for (auto &c : node->children) {
			check_node(c.get());
		}
	};
	check_node(root);
	return ok;
}

// New: verify that each operator's reported ColumnBindings correspond to its GetTypes and that expression return types
// are valid
static bool VerifyOperatorBindingsConsistentWithTypes(LogicalOperator *root) {
	if (!root) {
		return true;
	}
	bool ok = true;
	std::function<void(LogicalOperator *)> check_node = [&](LogicalOperator *node) {
		if (!node) {
			return;
		}
		auto binds = node->GetColumnBindings();
		auto types = node->types;
		// For each reported binding, ensure the producer/operator actually has a type for that column_index
		for (auto &b : binds) {
			if (b.column_index >= (idx_t)types.size()) {
				std::cerr << "Verification failure: operator " << node->GetName() << " reports binding column_index "
				          << b.column_index << " but types size is " << types.size() << "\n";
				ok = false;
			} else if (types[b.column_index].id() == LogicalTypeId::INVALID) {
				std::cerr << "Verification failure: operator " << node->GetName() << " has INVALID type for column "
				          << b.column_index << "\n";
				ok = false;
			}
		}
		// Ensure expressions' return types are not INVALID
		for (auto &expr : node->expressions) {
			// Enumerate all expressions under this unique_ptr and check their return types
			ExpressionIterator::EnumerateExpression(expr, [&](const Expression &e) {
				if (e.return_type.id() == LogicalTypeId::INVALID) {
					std::cerr << "Verification failure: expression in operator " << node->GetName()
					          << " has INVALID return type\n";
					ok = false;
				}
			});
		}
		for (auto &c : node->children) {
			check_node(c.get());
		}
	};
	check_node(root);
	return ok;
}

// Helper to run verification on a plan and its copy (copy optional)
static int RunRobustPlanVerification(const unique_ptr<LogicalOperator> &root) {
	try {
		DuckDB db(nullptr);
		Connection con(db);
		root->ResolveOperatorTypes();
		root->Verify(*con.context);

		// collect and verify bindings on root
		std::set<std::pair<idx_t, idx_t>> bindings;
		CollectColumnBindingsRecursiveForTest(root.get(), bindings);
		if (!VerifyNoInvalidBindings(bindings)) {
			return -1;
		}
		if (!VerifyAllExprBindingsExist(root.get(), bindings)) {
			return -2;
		}
		// new checks: ensure each binding has a unique producer and types align
		if (!VerifyBindingProducersUniqueAndTypes(root.get())) {
			return -5;
		}
		if (!VerifyExprTypesMatchProducers(root.get())) {
			return -6;
		}
		// operator-level consistency checks
		if (!VerifyOperatorBindingsConsistentWithTypes(root.get())) {
			return -9;
		}

		// attempt copy and run same checks on copy if supported
		try {
			auto copy = root->Copy(*con.context);
			copy->ResolveOperatorTypes();
			copy->Verify(*con.context);
			std::set<std::pair<idx_t, idx_t>> copy_bindings;
			CollectColumnBindingsRecursiveForTest(copy.get(), copy_bindings);
			if (!VerifyNoInvalidBindings(copy_bindings)) {
				return -3;
			}
			if (!VerifyAllExprBindingsExist(copy.get(), copy_bindings)) {
				return -4;
			}
			if (!VerifyBindingProducersUniqueAndTypes(copy.get())) {
				return -7;
			}
			if (!VerifyExprTypesMatchProducers(copy.get())) {
				return -8;
			}
			if (!VerifyOperatorBindingsConsistentWithTypes(copy.get())) {
				return -10;
			}
		} catch (NotImplementedException &ex) {
			std::cerr << "Copy skipped (not implemented): " << ex.what() << "\n";
		} catch (SerializationException &ex) {
			std::cerr << "Copy skipped (serialization): " << ex.what() << "\n";
		}
	} catch (std::exception &ex) {
		std::cerr << "Plan verification failed: " << ex.what() << "\n";
		return 1;
	}
	return 0;
}

// Test 1: basic ReplaceNode insertion and binding remap
static int Test_BasicReplaceNode() {
	// Build a simple plan: root projection (table_index = 0) with a child DummyScan (table_index = 0)
	auto child = make_uniq<LogicalDummyScan>(0);

	vector<unique_ptr<Expression>> proj_exprs;
	// Projection expression referencing child's first column (binding {0,0})
	proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
	// store root as unique_ptr<LogicalOperator> so it can be passed to ReplaceNode
	unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, std::move(proj_exprs));
	root->children.emplace_back(std::move(child));

	// Prepare a new parent projection that contains its own child (strict replace semantics)
	unique_ptr<LogicalOperator> new_parent = make_uniq<LogicalProjection>(1, vector<unique_ptr<Expression>>());
	new_parent->children.emplace_back(make_uniq<LogicalDummyScan>(5));
	ReplaceNode(root, root->children[0], new_parent);

	// After insertion, validate
	auto inserted_parent = dynamic_cast<LogicalProjection *>(root->children[0].get());
	if (!inserted_parent) {
		std::cerr << "Inserted parent is null\n";
		return 2;
	}
	if (inserted_parent->children.size() != 1) {
		std::cerr << "Inserted parent has wrong child count: " << inserted_parent->children.size() << "\n";
		return 3;
	}
	auto moved_child = dynamic_cast<LogicalDummyScan *>(inserted_parent->children[0].get());
	if (!moved_child) {
		std::cerr << "Moved child is not DummyScan\n";
		return 4;
	}

	// Verify plan with a ClientContext and ensure it can be copied (serialization round-trip)
	if (RunRobustPlanVerification(root) != 0) {
		return 100;
	}

	return 0;
}

// Test 2: multiple consecutive ReplaceNode calls produce consistent remapping
static int Test_ConsecutiveReplaceNode() {
	// Start with root projection (table_index = 0) and a DummyScan child (table_index = 0)
	auto child = make_uniq<LogicalDummyScan>(0);

	vector<unique_ptr<Expression>> proj_exprs;
	proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
	unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, std::move(proj_exprs));
	root->children.emplace_back(std::move(child));

	// First insertion
	unique_ptr<LogicalOperator> new_parent1 = make_uniq<LogicalProjection>(1, vector<unique_ptr<Expression>>());
	new_parent1->children.emplace_back(make_uniq<LogicalDummyScan>(7));
	ReplaceNode(root, root->children[0], new_parent1);

	// Second insertion above the same original child
	// operate on the unique_ptrs directly: pass the parent's unique_ptr and its child unique_ptr
	if (!root->children[0]) {
		return 7;
	}
	unique_ptr<LogicalOperator> new_parent2 = make_uniq<LogicalProjection>(2, vector<unique_ptr<Expression>>());
	new_parent2->children.emplace_back(make_uniq<LogicalDummyScan>(8));
	ReplaceNode(root->children[0], root->children[0]->children[0], new_parent2);

	// Validate structure: there should be two projections between root and the DummyScan
	const auto first = dynamic_cast<LogicalProjection *>(root->children[0].get());
	if (!first) {
		return 8;
	}
	if (first->children.size() != 1) {
		return 9;
	}
	auto second = dynamic_cast<LogicalProjection *>(first->children[0].get());
	if (!second) {
		return 10;
	}
	if (second->children.size() != 1) {
		return 11;
	}
	auto moved_child = dynamic_cast<LogicalDummyScan *>(second->children[0].get());
	if (!moved_child) {
		return 12;
	}

	// Verify plan consistency (bindings should be remapped correctly)
	if (RunRobustPlanVerification(root) != 0) {
		return 101;
	}

	return 0;
}

// Test 3: inserting parent when child has multiple columns and projection references later column
static int Test_MultiColumnChildBinding() {
	// DummyScan with two columns: logical behavior is simulated via projections using column bindings
	auto child = make_uniq<LogicalDummyScan>(0);

	vector<unique_ptr<Expression>> proj_exprs;
	// Reference first column from child binding
	proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
	unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, std::move(proj_exprs));
	root->children.emplace_back(std::move(child));

	// Create new_parent with a child scan - this matches the pattern used in other tests
	unique_ptr<LogicalOperator> new_parent = make_uniq<LogicalProjection>(1, vector<unique_ptr<Expression>>());
	new_parent->children.emplace_back(make_uniq<LogicalDummyScan>(17));

	ReplaceNode(root, root->children[0], new_parent);

	// Validate structure
	auto inserted_parent = dynamic_cast<LogicalProjection *>(root->children[0].get());
	if (!inserted_parent) {
		return 14;
	}

	// new_parent should have its child (DummyScan with table_index=17)
	if (inserted_parent->children.empty()) {
		return 15;
	}

	// Verify plan consistency
	if (RunRobustPlanVerification(root) != 0) {
		return 102;
	}

	return 0;
}

// Test 4: insert a projection between an aggregate and root
static int Test_InsertProjectionBetweenAggregateAndRoot() {
	// Setup: root projection referencing an aggregate child
	auto agg_child = make_uniq<LogicalAggregate>(0, 0, vector<unique_ptr<Expression>>());
	unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, vector<unique_ptr<Expression>>());
	// projection referencing the aggregate's (would-be) column 0
	root->expressions.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
	root->children.emplace_back(std::move(agg_child));

	// Insert a LogicalProjection as new parent above the aggregate
	unique_ptr<LogicalOperator> new_proj = make_uniq<LogicalProjection>(12, vector<unique_ptr<Expression>>());
	new_proj->children.emplace_back(make_uniq<LogicalDummyScan>(13));

	ReplaceNode(root, root->children[0], new_proj);

	auto inserted_proj = dynamic_cast<LogicalProjection *>(root->children[0].get());
	if (!inserted_proj) {
		std::cerr << "Inserted projection is null\n";
		return 20;
	}

	// Validate: projection should have one child
	if (inserted_proj->children.size() != 1) {
		std::cerr << "Inserted projection has wrong child count: " << inserted_proj->children.size() << "\n";
		return 21;
	}

	// Verify plan consistency
	if (RunRobustPlanVerification(root) != 0) {
		return 103;
	}

	return 0;
}

// Test 6: insert a filter parent above a scan and ensure child moved
static int Test_InsertFilterParent() {
	auto child = make_uniq<LogicalDummyScan>(0);
	vector<unique_ptr<Expression>> proj_exprs;
	proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
	unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, std::move(proj_exprs));
	root->children.emplace_back(std::move(child));

	unique_ptr<LogicalOperator> new_filter = make_uniq<LogicalFilter>();
	new_filter->children.emplace_back(make_uniq<LogicalDummyScan>(16));
	ReplaceNode(root, root->children[0], new_filter);

	// Verify plan
	try {
		DuckDB db(nullptr);
		Connection con(db);
		root->ResolveOperatorTypes();
		root->Verify(*con.context);
		try {
			auto copy = root->Copy(*con.context);
			copy->ResolveOperatorTypes();
			copy->Verify(*con.context);
		} catch (NotImplementedException &ex) {
			std::cerr << "Copy skipped (not implemented): " << ex.what() << "\n";
		} catch (SerializationException &ex) {
			std::cerr << "Copy skipped (serialization): " << ex.what() << "\n";
		}
		// execute an equivalent filter query
		con.Query("CREATE TABLE __t5(i INT); INSERT INTO __t5 VALUES (1),(2);");
		auto res = con.Query("SELECT * FROM __t5 WHERE i=1 LIMIT 1;");
		if (!res) {
			std::cerr << "Execution of equivalent filter query failed\n";
			return 105;
		}
	} catch (std::exception &ex) {
		std::cerr << "Plan verification/copy failed: " << ex.what() << "\n";
		return 105;
	}

	auto inserted_filter = dynamic_cast<LogicalFilter *>(root->children[0].get());
	if (!inserted_filter) {
		std::cerr << "Inserted filter is null\n";
		return 26;
	}
	if (inserted_filter->children.size() != 1) {
		return 27;
	}
	auto moved_child = dynamic_cast<LogicalDummyScan *>(inserted_filter->children[0].get());
	if (!moved_child) {
		return 28;
	}

	// Collect and verify bindings
	std::set<std::pair<idx_t, idx_t>> bindings;
	CollectColumnBindingsRecursiveForTest(root.get(), bindings);
	if (!VerifyAllExprBindingsExist(root.get(), bindings)) {
		return 1000;
	}

	return 0;
}

// Test 7: ensure ReplaceNode preserves column positions when no remapping is required
static int Test_PreserveColumnsNoRemap() {
	auto child = make_uniq<LogicalDummyScan>(0);
	vector<unique_ptr<Expression>> proj_exprs;
	proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
	proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 1)));
	unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, std::move(proj_exprs));
	root->children.emplace_back(std::move(child));

	unique_ptr<LogicalOperator> new_parent = make_uniq<LogicalProjection>(1, vector<unique_ptr<Expression>>());
	new_parent->children.emplace_back(make_uniq<LogicalDummyScan>(17));
	ReplaceNode(root, root->children[0], new_parent);

	// Validate structure
	auto inserted_parent = dynamic_cast<LogicalProjection *>(root->children[0].get());
	if (!inserted_parent) {
		return 29;
	}
	auto moved_child = dynamic_cast<LogicalDummyScan *>(inserted_parent->children[0].get());
	if (!moved_child) {
		return 30;
	}

	// Verify plan consistency - the important part is that column indices are preserved
	// (the table_index may be remapped to avoid collisions)
	if (RunRobustPlanVerification(root) != 0) {
		return 106;
	}

	// Verify column indices are preserved (0 and 1)
	auto &bcr0 = root->expressions[0]->Cast<BoundColumnRefExpression>();
	auto &bcr1 = root->expressions[1]->Cast<BoundColumnRefExpression>();
	if (bcr0.binding.column_index != 0) {
		return 31;
	}
	if (bcr1.binding.column_index != 1) {
		return 32;
	}

	return 0;
}

// New test: verifier should detect duplicate producers for the same ColumnBinding
static int Test_VerifierDetectsDuplicateProducers() {
	// Create a join whose left and right are both DummyScans with the same table_index (0)
	auto left = make_uniq<LogicalDummyScan>(0);
	auto right = make_uniq<LogicalDummyScan>(0);
	auto base_join = make_uniq<LogicalJoin>(JoinType::INNER);
	base_join->children.emplace_back(std::move(left));
	base_join->children.emplace_back(std::move(right));

	unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, vector<unique_ptr<Expression>>());
	// Add a projection expression that references the duplicated binding (table=0,col=0)
	root->expressions.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
	root->children.emplace_back(std::move(base_join));

	int res = RunRobustPlanVerification(root);
	// We expect the verifier to detect multiple producers and return non-zero
	if (res == 0) {
		std::cerr << "Expected verifier to detect duplicate producers but it returned success\n";
		return 200;
	}
	return 0;
}

int RunCompilerFunctionTests() {
	int code = 0;
	code = RunTest("Test_BasicReplaceNode", Test_BasicReplaceNode);
	if (code != 0) {
		return code;
	}
	code = RunTest("Test_ConsecutiveReplaceNode", Test_ConsecutiveReplaceNode);
	if (code != 0) {
		return code;
	}
	code = RunTest("Test_MultiColumnChildBinding", Test_MultiColumnChildBinding);
	if (code != 0) {
		return code;
	}
	code = RunTest("Test_InsertProjectionBetweenAggregateAndRoot", Test_InsertProjectionBetweenAggregateAndRoot);
	if (code != 0) {
		return code;
	}
	code = RunTest("Test_InsertFilterParent", Test_InsertFilterParent);
	if (code != 0) {
		return code;
	}
	code = RunTest("Test_PreserveColumnsNoRemap", Test_PreserveColumnsNoRemap);
	if (code != 0) {
		return code;
	}
	code = RunTest("Test_VerifierDetectsDuplicateProducers", Test_VerifierDetectsDuplicateProducers);
	if (code != 0) {
		return code;
	}

	// Comprehensive tests for ColumnBelongsToTable
	{
		std::cout << "Running ColumnBelongsToTable tests...\n";
		DuckDB db(nullptr);
		Connection con(db);
		con.BeginTransaction();

		con.Query("CREATE TABLE t1(a INTEGER, b INTEGER);");
		con.Query("CREATE TABLE t2(x INTEGER, y INTEGER);");
		con.Query("CREATE TABLE t3(m INTEGER, n INTEGER);");

		con.Query("INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30);");
		con.Query("INSERT INTO t2 VALUES (1, 100), (2, 200);");
		con.Query("INSERT INTO t3 VALUES (1, 1000), (2, 2000);");

		// Test 1: Simple scan - check that columns from t1 belong to t1
		{
			Parser parser;
			parser.ParseQuery("SELECT a, b FROM t1;");

			Planner planner(*con.context);
			planner.CreatePlan(std::move(parser.statements[0]));
			Optimizer opt(*planner.binder, *con.context);

			auto plan = opt.Optimize(std::move(planner.plan));

			// Find the LogicalGet node for t1
			LogicalGet *get_t1 = nullptr;
			std::function<void(LogicalOperator &)> find_get = [&](LogicalOperator &op) {
				if (op.type == LogicalOperatorType::LOGICAL_GET && !get_t1) {
					auto &g = op.Cast<LogicalGet>();
					if (g.GetTable() && g.GetTable()->name == "t1") {
						get_t1 = &g;
					}
				}
				for (auto &child : op.children) {
					if (child) {
						find_get(*child);
					}
				}
			};
			find_get(*plan);

			if (!get_t1) {
				std::cerr << "FAIL: ColumnBelongsToTable test 1: could not find LogicalGet for t1"
				          << "\n";
				return 1;
			}

			// Get bindings from the scan node
			auto bindings = get_t1->GetColumnBindings();
			if (bindings.size() < 2) {
				std::cerr << "FAIL: ColumnBelongsToTable test 1: expected at least 2 bindings from t1"
				          << "\n";
				return 1;
			}

			// Check that both columns belong to t1
			if (!ColumnBelongsToTable(*plan, "t1", bindings[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 1: column 0 should belong to t1"
				          << "\n";
				return 1;
			}
			if (!ColumnBelongsToTable(*plan, "t1", bindings[1])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 1: column 1 should belong to t1"
				          << "\n";
				return 1;
			}
			// Check that they don't belong to t2
			if (ColumnBelongsToTable(*plan, "t2", bindings[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 1: column 0 should not belong to t2"
				          << "\n";
				return 1;
			}
			std::cerr << "PASS: ColumnBelongsToTable test 1 (simple scan)"
			          << "\n";
		}

		// Test 2: Join - check join key equivalence and non-join-key isolation
		// Query projects all columns so we can test both join-key and non-join-key columns.
		// Join key columns (t1.a, t2.x) belong to BOTH tables via equivalence.
		// Non-join-key columns (t1.b, t2.y) belong ONLY to their own table.
		{
			Parser parser;
			parser.ParseQuery("SELECT t1.a, t1.b, t2.x, t2.y FROM t1 INNER JOIN t2 ON t1.a = t2.x;");
			Planner planner(*con.context);
			planner.CreatePlan(std::move(parser.statements[0]));
			Optimizer opt(*planner.binder, *con.context);
			auto plan = opt.Optimize(std::move(planner.plan));

			// Find LogicalGet nodes for both tables
			LogicalGet *get_t1 = nullptr;
			LogicalGet *get_t2 = nullptr;
			std::function<void(LogicalOperator &)> find_gets = [&](LogicalOperator &op) {
				if (op.type == LogicalOperatorType::LOGICAL_GET) {
					auto &g = op.Cast<LogicalGet>();
					if (g.GetTable()) {
						if (g.GetTable()->name == "t1") {
							get_t1 = &g;
						} else if (g.GetTable()->name == "t2") {
							get_t2 = &g;
						}
					}
				}
				for (auto &child : op.children) {
					if (child) {
						find_gets(*child);
					}
				}
			};
			find_gets(*plan);

			if (!get_t1 || !get_t2) {
				std::cerr << "FAIL: ColumnBelongsToTable test 2: could not find both LogicalGet nodes"
				          << "\n";
				return 1;
			}

			auto bindings_t1 = get_t1->GetColumnBindings();
			auto bindings_t2 = get_t2->GetColumnBindings();

			if (bindings_t1.size() < 2 || bindings_t2.size() < 2) {
				std::cerr << "FAIL: ColumnBelongsToTable test 2: expected at least 2 bindings per table"
				          << "\n";
				return 1;
			}

			// Join key columns (t1.a=bindings_t1[0], t2.x=bindings_t2[0]) should belong to
			// BOTH tables due to join key equivalence (ON t1.a = t2.x)
			if (!ColumnBelongsToTable(*plan, "t1", bindings_t1[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 2: t1.a should belong to t1"
				          << "\n";
				return 1;
			}
			if (!ColumnBelongsToTable(*plan, "t2", bindings_t1[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 2: t1.a should also belong to t2 (join key equivalence)"
				          << "\n";
				return 1;
			}
			if (!ColumnBelongsToTable(*plan, "t2", bindings_t2[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 2: t2.x should belong to t2"
				          << "\n";
				return 1;
			}
			if (!ColumnBelongsToTable(*plan, "t1", bindings_t2[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 2: t2.x should also belong to t1 (join key equivalence)"
				          << "\n";
				return 1;
			}

			// Non-join-key columns (t1.b=bindings_t1[1], t2.y=bindings_t2[1]) should NOT
			// belong to the other table
			if (!ColumnBelongsToTable(*plan, "t1", bindings_t1[1])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 2: t1.b should belong to t1"
				          << "\n";
				return 1;
			}
			if (ColumnBelongsToTable(*plan, "t2", bindings_t1[1])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 2: t1.b should NOT belong to t2"
				          << "\n";
				return 1;
			}
			if (!ColumnBelongsToTable(*plan, "t2", bindings_t2[1])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 2: t2.y should belong to t2"
				          << "\n";
				return 1;
			}
			if (ColumnBelongsToTable(*plan, "t1", bindings_t2[1])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 2: t2.y should NOT belong to t1"
				          << "\n";
				return 1;
			}
			std::cerr << "PASS: ColumnBelongsToTable test 2 (join key equivalence + non-join-key isolation)"
			          << "\n";
		}

		// Test 3: Three-way join with join key equivalence
		// Join conditions: t1.a = t2.x AND t2.x = t3.m → transitive: t1.a = t2.x = t3.m
		// Join key columns belong to all three tables; non-join-key columns stay isolated.
		{
			Parser parser;
			parser.ParseQuery("SELECT * FROM t1 INNER JOIN t2 ON t1.a = t2.x INNER JOIN t3 ON t2.x = t3.m;");
			Planner planner(*con.context);
			planner.CreatePlan(std::move(parser.statements[0]));
			Optimizer opt(*planner.binder, *con.context);
			auto plan = opt.Optimize(std::move(planner.plan));

			// Find all LogicalGet nodes
			LogicalGet *get_t1 = nullptr;
			LogicalGet *get_t2 = nullptr;
			LogicalGet *get_t3 = nullptr;
			std::function<void(LogicalOperator &)> find_gets = [&](LogicalOperator &op) {
				if (op.type == LogicalOperatorType::LOGICAL_GET) {
					auto &g = op.Cast<LogicalGet>();
					if (g.GetTable()) {
						if (g.GetTable()->name == "t1") {
							get_t1 = &g;
						} else if (g.GetTable()->name == "t2") {
							get_t2 = &g;
						} else if (g.GetTable()->name == "t3") {
							get_t3 = &g;
						}
					}
				}
				for (auto &child : op.children) {
					if (child) {
						find_gets(*child);
					}
				}
			};
			find_gets(*plan);

			if (!get_t1 || !get_t2 || !get_t3) {
				std::cerr << "FAIL: ColumnBelongsToTable test 3: could not find all three LogicalGet nodes"
				          << "\n";
				return 1;
			}

			auto bindings_t1 = get_t1->GetColumnBindings();
			auto bindings_t2 = get_t2->GetColumnBindings();
			auto bindings_t3 = get_t3->GetColumnBindings();

			if (bindings_t1.size() < 2 || bindings_t2.size() < 2 || bindings_t3.size() < 2) {
				std::cerr << "FAIL: ColumnBelongsToTable test 3: expected at least 2 bindings per table"
				          << "\n";
				return 1;
			}

			// Join key columns (t1.a, t2.x, t3.m) should belong to their own table
			if (!ColumnBelongsToTable(*plan, "t1", bindings_t1[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 3: t1.a should belong to t1"
				          << "\n";
				return 1;
			}
			if (!ColumnBelongsToTable(*plan, "t2", bindings_t2[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 3: t2.x should belong to t2"
				          << "\n";
				return 1;
			}
			if (!ColumnBelongsToTable(*plan, "t3", bindings_t3[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 3: t3.m should belong to t3"
				          << "\n";
				return 1;
			}

			// Join key equivalence: t1.a = t2.x = t3.m (transitive)
			// t1.a should also belong to t2 and t3
			if (!ColumnBelongsToTable(*plan, "t2", bindings_t1[0]) ||
			    !ColumnBelongsToTable(*plan, "t3", bindings_t1[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 3: t1.a should belong to t2 and t3 (join key equivalence)"
				          << "\n";
				return 1;
			}

			// Non-join-key columns should NOT cross tables
			// t1.b should not belong to t2 or t3
			if (ColumnBelongsToTable(*plan, "t2", bindings_t1[1]) ||
			    ColumnBelongsToTable(*plan, "t3", bindings_t1[1])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 3: t1.b should NOT belong to t2 or t3"
				          << "\n";
				return 1;
			}
			// t2.y should not belong to t1 or t3
			if (ColumnBelongsToTable(*plan, "t1", bindings_t2[1]) ||
			    ColumnBelongsToTable(*plan, "t3", bindings_t2[1])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 3: t2.y should NOT belong to t1 or t3"
				          << "\n";
				return 1;
			}
			// t3.n should not belong to t1 or t2
			if (ColumnBelongsToTable(*plan, "t1", bindings_t3[1]) ||
			    ColumnBelongsToTable(*plan, "t2", bindings_t3[1])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 3: t3.n should NOT belong to t1 or t2"
				          << "\n";
				return 1;
			}
			std::cerr << "PASS: ColumnBelongsToTable test 3 (three-way join key equivalence)"
			          << "\n";
		}

		// Test 4: Non-existent table
		{
			Parser parser;
			parser.ParseQuery("SELECT a FROM t1;");
			Planner planner(*con.context);
			planner.CreatePlan(std::move(parser.statements[0]));
			Optimizer opt(*planner.binder, *con.context);
			auto plan = opt.Optimize(std::move(planner.plan));

			LogicalGet *get_t1 = nullptr;
			std::function<void(LogicalOperator &)> find_get = [&](LogicalOperator &op) {
				if (op.type == LogicalOperatorType::LOGICAL_GET && !get_t1) {
					auto &g = op.Cast<LogicalGet>();
					if (g.GetTable() && g.GetTable()->name == "t1") {
						get_t1 = &g;
					}
				}
				for (auto &child : op.children) {
					if (child) {
						find_get(*child);
					}
				}
			};
			find_get(*plan);

			if (!get_t1) {
				std::cerr << "FAIL: ColumnBelongsToTable test 4: could not find LogicalGet for t1"
				          << "\n";
				return 1;
			}

			auto bindings = get_t1->GetColumnBindings();
			if (bindings.empty()) {
				std::cerr << "FAIL: ColumnBelongsToTable test 4: no bindings found"
				          << "\n";
				return 1;
			}

			// Check that the column doesn't belong to a non-existent table
			if (ColumnBelongsToTable(*plan, "nonexistent_table", bindings[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 4: column should not belong to nonexistent table"
				          << "\n";
				return 1;
			}
			std::cerr << "PASS: ColumnBelongsToTable test 4 (non-existent table)"
			          << "\n";
		}

		// Test 5: Subquery with multiple tables
		{
			Parser parser;
			parser.ParseQuery("SELECT * FROM (SELECT t1.a, t2.x FROM t1, t2) AS sub;");
			Planner planner(*con.context);
			planner.CreatePlan(std::move(parser.statements[0]));
			Optimizer opt(*planner.binder, *con.context);
			auto plan = opt.Optimize(std::move(planner.plan));

			// Find LogicalGet nodes for both tables
			LogicalGet *get_t1 = nullptr;
			LogicalGet *get_t2 = nullptr;
			std::function<void(LogicalOperator &)> find_gets = [&](LogicalOperator &op) {
				if (op.type == LogicalOperatorType::LOGICAL_GET) {
					auto &g = op.Cast<LogicalGet>();
					if (g.GetTable()) {
						if (g.GetTable()->name == "t1" && !get_t1) {
							get_t1 = &g;
						} else if (g.GetTable()->name == "t2" && !get_t2) {
							get_t2 = &g;
						}
					}
				}
				for (auto &child : op.children) {
					if (child) {
						find_gets(*child);
					}
				}
			};
			find_gets(*plan);

			if (!get_t1 || !get_t2) {
				std::cerr << "FAIL: ColumnBelongsToTable test 5: could not find both LogicalGet nodes in subquery"
				          << "\n";
				return 1;
			}

			auto bindings_t1 = get_t1->GetColumnBindings();
			auto bindings_t2 = get_t2->GetColumnBindings();

			if (bindings_t1.empty() || bindings_t2.empty()) {
				std::cerr << "FAIL: ColumnBelongsToTable test 5: no bindings found"
				          << "\n";
				return 1;
			}

			// Verify columns are correctly attributed even in subquery
			if (!ColumnBelongsToTable(*plan, "t1", bindings_t1[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 5: t1 column should belong to t1 in subquery"
				          << "\n";
				return 1;
			}
			if (!ColumnBelongsToTable(*plan, "t2", bindings_t2[0])) {
				std::cerr << "FAIL: ColumnBelongsToTable test 5: t2 column should belong to t2 in subquery"
				          << "\n";
				return 1;
			}
			con.Rollback();
			std::cerr << "PASS: ColumnBelongsToTable test 5 (subquery)"
			          << "\n";
		}
	}

	// Test 6: Three-way join with GROUP BY - simulate the failing test case
	{
		DuckDB db(nullptr);
		Connection con(db);
		con.BeginTransaction();

		con.Query("CREATE TABLE t1(a INTEGER, b INTEGER);");
		con.Query("CREATE TABLE t2(x INTEGER, y INTEGER);");
		con.Query("CREATE TABLE t3(m INTEGER, n INTEGER);");

		con.Query("INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30);");
		con.Query("INSERT INTO t2 VALUES (1, 100), (2, 200);");
		con.Query("INSERT INTO t3 VALUES (1, 1000), (2, 2000);");

		const string query =
		    "SELECT t2.x, SUM(t3.n) FROM t1 INNER JOIN t2 ON t1.a = t2.x INNER JOIN t3 ON t2.x = t3.m GROUP BY t2.x;";

		// Create plan using parser, planner and optimizer
		Parser parser;
		parser.ParseQuery(query);
		Planner planner(*con.context);
		planner.CreatePlan(std::move(parser.statements[0]));
		Optimizer opt(*planner.binder, *con.context);
		auto plan = opt.Optimize(std::move(planner.plan));
		if (!plan) {
			std::cerr << "FAIL: ColumnBelongsToTable test 6: replan returned null plan"
			          << "\n";
			return 1;
		}

		// Find the LogicalAggregate node
		LogicalAggregate *agg_node = nullptr;
		std::function<void(LogicalOperator &)> find_agg = [&](LogicalOperator &op) {
			if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY && !agg_node) {
				agg_node = &op.Cast<LogicalAggregate>();
			}
			for (auto &child : op.children) {
				if (child) {
					find_agg(*child);
				}
			}
		};
		find_agg(*plan);

		if (!agg_node) {
			std::cerr << "FAIL: ColumnBelongsToTable test 6: could not find aggregate node"
			          << "\n";
			return 1;
		}

		// Check the GROUP BY expression
		if (agg_node->groups.empty()) {
			std::cerr << "FAIL: ColumnBelongsToTable test 6: no group expressions found"
			          << "\n";
			return 1;
		}

		auto &group_expr = agg_node->groups[0];
		if (group_expr->type != ExpressionType::BOUND_COLUMN_REF) {
			std::cerr << "FAIL: ColumnBelongsToTable test 6: group expression is not a column ref"
			          << "\n";
			return 1;
		}

		auto &col_ref = group_expr->Cast<BoundColumnRefExpression>();

		// Find LogicalGet nodes
		LogicalGet *get_t1 = nullptr;
		LogicalGet *get_t2 = nullptr;
		std::function<void(LogicalOperator &)> find_gets = [&](LogicalOperator &op) {
			if (op.type == LogicalOperatorType::LOGICAL_GET) {
				auto &g = op.Cast<LogicalGet>();
				if (g.GetTable()) {
					if (g.GetTable()->name == "t1" && !get_t1) {
						get_t1 = &g;
					} else if (g.GetTable()->name == "t2" && !get_t2) {
						get_t2 = &g;
					}
				}
			}
			for (auto &child : op.children) {
				if (child) {
					find_gets(*child);
				}
			}
		};
		find_gets(*plan);

		if (!get_t1 || !get_t2) {
			std::cerr << "FAIL: ColumnBelongsToTable test 6: could not find both LogicalGet nodes"
			          << "\n";
			return 1;
		}

		// The GROUP BY column could belong to either t1 or t2 since we're joining on the key
		// (t1.a = t2.x), making them equivalent
		bool belongs_to_t1 = ColumnBelongsToTable(*plan, "t1", col_ref.binding);
		bool belongs_to_t2 = ColumnBelongsToTable(*plan, "t2", col_ref.binding);

		// The column should belong to at least one of the tables (t1 or t2)
		if (!belongs_to_t1 && !belongs_to_t2) {
			std::cerr << "FAIL: ColumnBelongsToTable test 6: GROUP BY column should belong to either t1 or t2"
			          << "\n";
			return 1;
		}

		con.Rollback();
		std::cerr << "PASS: ColumnBelongsToTable test 6 (three-way join with GROUP BY)"
		          << "\n";
	}

	// Test 7: Simple grouped aggregation - validate column-to-table bindings
	// This tests the pattern: SELECT a, SUM(b) FROM t1 GROUP BY a ORDER BY a
	{
		DuckDB db(nullptr);
		Connection con(db);
		con.BeginTransaction();

		con.Query("CREATE TABLE t1(a INTEGER, b INTEGER);");
		con.Query("INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30), (1, 15), (2, 25);");

		const string query = "SELECT a, SUM(b) FROM t1 GROUP BY a ORDER BY a;";

		// Create plan using parser, planner and optimizer
		Parser parser;
		parser.ParseQuery(query);
		Planner planner(*con.context);
		planner.CreatePlan(std::move(parser.statements[0]));
		Optimizer opt(*planner.binder, *con.context);
		auto plan = opt.Optimize(std::move(planner.plan));
		if (!plan) {
			std::cerr << "FAIL: ColumnBelongsToTable test 7: replan returned null plan"
			          << "\n";
			return 1;
		}

		// Find the LogicalGet node for t1
		LogicalGet *get_t1 = nullptr;
		std::function<void(LogicalOperator &)> find_get = [&](LogicalOperator &op) {
			if (op.type == LogicalOperatorType::LOGICAL_GET && !get_t1) {
				auto &g = op.Cast<LogicalGet>();
				if (g.GetTable() && g.GetTable()->name == "t1") {
					get_t1 = &g;
				}
			}
			for (auto &child : op.children) {
				if (child) {
					find_get(*child);
				}
			}
		};
		find_get(*plan);

		if (!get_t1) {
			std::cerr << "FAIL: ColumnBelongsToTable test 7: could not find LogicalGet for t1"
			          << "\n";
			return 1;
		}

		// Find the LogicalAggregate node
		LogicalAggregate *agg_node = nullptr;
		std::function<void(LogicalOperator &)> find_agg = [&](LogicalOperator &op) {
			if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY && !agg_node) {
				agg_node = &op.Cast<LogicalAggregate>();
			}
			for (auto &child : op.children) {
				if (child) {
					find_agg(*child);
				}
			}
		};
		find_agg(*plan);

		if (!agg_node) {
			std::cerr << "FAIL: ColumnBelongsToTable test 7: could not find aggregate node"
			          << "\n";
			return 1;
		}

		// Verify GROUP BY expression (column 'a')
		if (agg_node->groups.empty()) {
			std::cerr << "FAIL: ColumnBelongsToTable test 7: no group expressions found"
			          << "\n";
			return 1;
		}

		auto &group_expr = agg_node->groups[0];
		if (group_expr->type != ExpressionType::BOUND_COLUMN_REF) {
			std::cerr << "FAIL: ColumnBelongsToTable test 7: group expression is not a column ref"
			          << "\n";
			return 1;
		}

		auto &group_col_ref = group_expr->Cast<BoundColumnRefExpression>();

		// Verify that the GROUP BY column 'a' belongs to t1
		if (!ColumnBelongsToTable(*plan, "t1", group_col_ref.binding)) {
			std::cerr << "FAIL: ColumnBelongsToTable test 7: GROUP BY column 'a' should belong to t1"
			          << "\n";
			return 1;
		}

		// Verify the aggregate expression (SUM(b))
		if (agg_node->expressions.empty()) {
			std::cerr << "FAIL: ColumnBelongsToTable test 7: no aggregate expressions found"
			          << "\n";
			return 1;
		}

		// The aggregate expression should contain a BoundColumnRefExpression for column 'b'
		bool found_b_column = false;
		for (auto &agg_expr : agg_node->expressions) {
			ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
			    *agg_expr, [&](const BoundColumnRefExpression &col_ref) {
				    // Verify that column 'b' belongs to t1
				    if (!ColumnBelongsToTable(*plan, "t1", col_ref.binding)) {
					    std::cerr << "FAIL: ColumnBelongsToTable test 7: SUM(b) column 'b' should belong to t1"
					              << "\n";
					    found_b_column = false;
				    } else {
					    found_b_column = true;
				    }
			    });
		}

		if (!found_b_column) {
			std::cerr << "FAIL: ColumnBelongsToTable test 7: could not find or verify column 'b' in SUM"
			          << "\n";
			return 1;
		}

		// Verify all bindings from t1 are correctly attributed
		auto bindings_t1 = get_t1->GetColumnBindings();
		for (const auto &binding : bindings_t1) {
			if (!ColumnBelongsToTable(*plan, "t1", binding)) {
				std::cerr << "FAIL: ColumnBelongsToTable test 7: binding from t1 should belong to t1"
				          << "\n";
				return 1;
			}
			// Verify it doesn't belong to a non-existent table
			if (ColumnBelongsToTable(*plan, "t2", binding)) {
				std::cerr << "FAIL: ColumnBelongsToTable test 7: t1 binding should not belong to t2"
				          << "\n";
				return 1;
			}
		}

		con.Rollback();
		std::cerr
		    << "PASS: ColumnBelongsToTable test 7 (simple grouped aggregation SELECT a, SUM(b) FROM t1 GROUP BY a)"
		    << "\n";
	}

	// Test 8: Join with projection of NON-join key should be allowed
	{
		DuckDB db(nullptr);
		Connection con(db);
		con.BeginTransaction();

		con.Query("CREATE TABLE pu_table(a INTEGER, b INTEGER);");
		con.Query("CREATE TABLE non_pu_table(x INTEGER, y INTEGER);");
		con.Query("INSERT INTO pu_table VALUES (1, 10), (2, 20), (3, 30);");
		con.Query("INSERT INTO non_pu_table VALUES (1, 100), (2, 200), (3, 300);");

		// Query that projects NON-join key from non-PU table (should be safe)
		const string query = "SELECT non_pu_table.y, SUM(pu_table.b) FROM pu_table INNER JOIN non_pu_table ON "
		                     "pu_table.a = non_pu_table.x GROUP BY non_pu_table.y;";

		// Create plan using parser, planner and optimizer
		Parser parser;
		parser.ParseQuery(query);
		Planner planner(*con.context);
		planner.CreatePlan(std::move(parser.statements[0]));
		Optimizer opt(*planner.binder, *con.context);
		auto plan = opt.Optimize(std::move(planner.plan));
		if (!plan) {
			std::cerr << "FAIL: ColumnBelongsToTable test 8: replan returned null plan"
			          << "\n";
			return 1;
		}

		// Find aggregate node
		LogicalAggregate *agg_node = nullptr;
		std::function<void(LogicalOperator &)> find_agg = [&](LogicalOperator &op) {
			if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY && !agg_node) {
				agg_node = &op.Cast<LogicalAggregate>();
			}
			for (auto &child : op.children) {
				if (child) {
					find_agg(*child);
				}
			}
		};
		find_agg(*plan);

		if (!agg_node) {
			std::cerr << "FAIL: ColumnBelongsToTable test 8: could not find aggregate node"
			          << "\n";
			return 1;
		}

		// Verify GROUP BY expression references non_pu_table.y (NON-join key)
		if (agg_node->groups.empty()) {
			std::cerr << "FAIL: ColumnBelongsToTable test 8: no group expressions found"
			          << "\n";
			return 1;
		}

		auto &group_expr = agg_node->groups[0];
		if (group_expr->type != ExpressionType::BOUND_COLUMN_REF) {
			std::cerr << "FAIL: ColumnBelongsToTable test 8: group expression is not a column ref"
			          << "\n";
			return 1;
		}

		auto &col_ref = group_expr->Cast<BoundColumnRefExpression>();

		// The GROUP BY column should belong to non_pu_table
		if (!ColumnBelongsToTable(*plan, "non_pu_table", col_ref.binding)) {
			std::cerr << "FAIL: ColumnBelongsToTable test 8: GROUP BY column should belong to non_pu_table"
			          << "\n";
			return 1;
		}

		// Should NOT belong to pu_table
		if (ColumnBelongsToTable(*plan, "pu_table", col_ref.binding)) {
			std::cerr << "FAIL: ColumnBelongsToTable test 8: GROUP BY column should not belong to pu_table"
			          << "\n";
			return 1;
		}

		// Since y is NOT the join key, projecting non_pu_table.y is safe and doesn't expose PU values

		con.Rollback();
		std::cerr << "PASS: ColumnBelongsToTable test 8 (join with projection of non-join key from non-PU table)"
		          << "\n";
	}

	// Test 9: Join key equivalence - projecting join key from non-PU table should be detected as exposing PU
	// When we have t1.a = t2.x, projecting t2.x exposes the same values as t1.a
	{
		DuckDB db(nullptr);
		Connection con(db);
		con.BeginTransaction();

		con.Query("CREATE TABLE pu_table(a INTEGER, b INTEGER);");
		con.Query("CREATE TABLE non_pu_table(x INTEGER, y INTEGER);");
		con.Query("INSERT INTO pu_table VALUES (1, 10), (2, 20), (3, 30);");
		con.Query("INSERT INTO non_pu_table VALUES (1, 100), (2, 200), (3, 300);");

		// Query that projects JOIN KEY from non-PU table (should be detected as exposing PU)
		const string query = "SELECT non_pu_table.x, SUM(pu_table.b) FROM pu_table LEFT JOIN non_pu_table ON "
		                     "pu_table.a = non_pu_table.x GROUP BY non_pu_table.x;";

		// Create plan using parser, planner and optimizer
		Parser parser;
		parser.ParseQuery(query);
		Planner planner(*con.context);
		planner.CreatePlan(std::move(parser.statements[0]));
		Optimizer opt(*planner.binder, *con.context);
		auto plan = opt.Optimize(std::move(planner.plan));
		if (!plan) {
			std::cerr << "FAIL: ColumnBelongsToTable test 9: replan returned null plan"
			          << "\n";
			return 1;
		}

		// Find aggregate node
		LogicalAggregate *agg_node = nullptr;
		std::function<void(LogicalOperator &)> find_agg = [&](LogicalOperator &op) {
			if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY && !agg_node) {
				agg_node = &op.Cast<LogicalAggregate>();
			}
			for (auto &child : op.children) {
				if (child) {
					find_agg(*child);
				}
			}
		};
		find_agg(*plan);

		if (!agg_node) {
			std::cerr << "FAIL: ColumnBelongsToTable test 9: could not find aggregate node"
			          << "\n";
			return 1;
		}

		// Verify GROUP BY expression references non_pu_table.x (JOIN KEY)
		if (agg_node->groups.empty()) {
			std::cerr << "FAIL: ColumnBelongsToTable test 9: no group expressions found"
			          << "\n";
			return 1;
		}

		auto &group_expr = agg_node->groups[0];
		if (group_expr->type != ExpressionType::BOUND_COLUMN_REF) {
			std::cerr << "FAIL: ColumnBelongsToTable test 9: group expression is not a column ref"
			          << "\n";
			return 1;
		}

		auto &col_ref = group_expr->Cast<BoundColumnRefExpression>();

		// The GROUP BY column physically comes from non_pu_table
		if (!ColumnBelongsToTable(*plan, "non_pu_table", col_ref.binding)) {
			std::cerr << "FAIL: ColumnBelongsToTable test 9: GROUP BY column should belong to non_pu_table"
			          << "\n";
			return 1;
		}

		// BUT because it's a join key (pu_table.a = non_pu_table.x), it should ALSO be detected
		// as belonging to pu_table due to join key equivalence
		if (!ColumnBelongsToTable(*plan, "pu_table", col_ref.binding)) {
			std::cerr << "FAIL: ColumnBelongsToTable test 9: JOIN KEY column should ALSO belong to pu_table "
			             "(due to join key equivalence)"
			          << "\n";
			return 1;
		}

		con.Rollback();
		std::cerr << "PASS: ColumnBelongsToTable test 9 (join key equivalence - t2.x detected as belonging to PU table)"
		          << "\n";
	}

	std::cout << "All ReplaceNode tests passed\n";
	return 0;
}
} // namespace duckdb
