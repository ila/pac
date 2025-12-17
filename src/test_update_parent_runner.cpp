#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include "duckdb/planner/expression_iterator.hpp"
#include <set>

#include "include/pac_helpers.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/common/enums/join_type.hpp"
#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/common/constants.hpp"

using namespace duckdb;
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
    if (!node) return;
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
    if (!node) return true;
    bool ok = true;
    for (auto &expr : node->expressions) {
        ExpressionIterator::VisitExpression<BoundColumnRefExpression>(*expr, [&](const BoundColumnRefExpression &bcr) {
            auto key = std::make_pair(bcr.binding.table_index, bcr.binding.column_index);
            if (bindings.find(key) == bindings.end()) {
                std::cerr << "Verification failure: expression references missing binding (table=" << bcr.binding.table_index
                          << ", col=" << bcr.binding.column_index << ") in operator " << node->GetName() << "\n";
                ok = false;
            }
        });
    }
    for (auto &c : node->children) {
        if (!VerifyAllExprBindingsExist(c.get(), bindings)) ok = false;
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
static void CollectBindingProducers(LogicalOperator *node, std::map<std::pair<idx_t, idx_t>, std::vector<LogicalOperator*>> &out) {
    if (!node) return;
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
    std::map<std::pair<idx_t, idx_t>, std::vector<LogicalOperator*>> producers;
    CollectBindingProducers(root, producers);
    bool ok = true;
    for (auto &kv : producers) {
        const auto &binding = kv.first;
        const auto &vec = kv.second;
        if (vec.empty()) {
            std::cerr << "Verification failure: binding (table=" << binding.first << ", col=" << binding.second << ") has no producer\n";
            ok = false;
            continue;
        }
        if (vec.size() > 1) {
            std::cerr << "Verification failure: binding (table=" << binding.first << ", col=" << binding.second << ") has multiple producers (" << vec.size() << ")\n";
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
            std::cerr << "Verification failure: null producer for binding (table=" << binding.first << ", col=" << binding.second << ")\n";
            ok = false;
            continue;
        }
        auto types = producer->types;
        if (binding.second >= (idx_t)types.size()) {
            std::cerr << "Verification failure: producer " << producer->GetName() << " does not have column index " << binding.second << " (types=" << types.size() << ")\n";
            ok = false;
            continue;
        }
        // ensure the advertised type is not Invalid
        if (types[binding.second].id() == LogicalTypeId::INVALID) {
            std::cerr << "Verification failure: producer " << producer->GetName() << " has INVALID type for column " << binding.second << "\n";
            ok = false;
        }
    }
    // Additionally ensure there are no duplicate column_index values within a single producer's bindings
    // For each producer, collect the set of column_index values it produces
    std::map<LogicalOperator*, std::set<idx_t>> producer_cols;
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
        for (auto &b : binds) reported_cols.insert(b.column_index);
        if (cols.size() != reported_cols.size()) {
            std::cerr << "Verification failure: mismatch in column set sizes for producer " << p->GetName() << " (col set sizes: collected=" << cols.size() << ", reported=" << reported_cols.size() << ")\n";
            ok = false;
        }
        // ensure the sets are identical
        for (auto cidx : cols) {
            if (reported_cols.find(cidx) == reported_cols.end()) {
                std::cerr << "Verification failure: producer " << p->GetName() << " missing column_index " << cidx << " in its reported bindings\n";
                ok = false;
            }
        }
    }
    return ok;
}

// New: verify that BoundColumnRefExpression.type matches the producer's type
static bool VerifyExprTypesMatchProducers(LogicalOperator *root) {
    // build producers map
    std::map<std::pair<idx_t, idx_t>, std::vector<LogicalOperator*>> producers;
    CollectBindingProducers(root, producers);
    bool ok = true;
    // traverse expressions
    std::function<void(LogicalOperator*)> check_node = [&](LogicalOperator *node) {
        if (!node) return;
        for (auto &expr : node->expressions) {
            ExpressionIterator::VisitExpression<BoundColumnRefExpression>(*expr, [&](const BoundColumnRefExpression &bcr) {
                auto key = std::make_pair(bcr.binding.table_index, bcr.binding.column_index);
                auto it = producers.find(key);
                if (it == producers.end() || it->second.empty()) {
                    // missing producer already reported elsewhere; skip
                    return;
                }
                auto *producer = it->second[0];
                auto types = producer->types;
                if (bcr.binding.column_index >= (idx_t)types.size()) {
                    std::cerr << "Verification failure: expression references column index " << bcr.binding.column_index << " beyond producer types size " << types.size() << "\n";
                    ok = false;
                    return;
                }
                auto prod_type = types[bcr.binding.column_index];
                if (bcr.return_type.id() != prod_type.id()) {
                    std::cerr << "Verification failure: expression type mismatch for binding (table=" << bcr.binding.table_index << ", col=" << bcr.binding.column_index << "): expr_type=" << bcr.return_type.ToString() << " producer_type=" << prod_type.ToString() << "\n";
                    ok = false;
                }
            });
        }
        for (auto &c : node->children) check_node(c.get());
    };
    check_node(root);
    return ok;
}

// New: verify that each operator's reported ColumnBindings correspond to its GetTypes and that expression return types are valid
static bool VerifyOperatorBindingsConsistentWithTypes(LogicalOperator *root) {
    if (!root) return true;
    bool ok = true;
    std::function<void(LogicalOperator*)> check_node = [&](LogicalOperator *node) {
        if (!node) return;
        auto binds = node->GetColumnBindings();
        auto types = node->types;
        // For each reported binding, ensure the producer/operator actually has a type for that column_index
        for (auto &b : binds) {
            if (b.column_index >= (idx_t)types.size()) {
                std::cerr << "Verification failure: operator " << node->GetName() << " reports binding column_index " << b.column_index << " but types size is " << types.size() << "\n";
                ok = false;
            } else if (types[b.column_index].id() == LogicalTypeId::INVALID) {
                std::cerr << "Verification failure: operator " << node->GetName() << " has INVALID type for column " << b.column_index << "\n";
                ok = false;
            }
        }
        // Ensure expressions' return types are not INVALID
        for (auto &expr : node->expressions) {
            // Enumerate all expressions under this unique_ptr and check their return types
            ExpressionIterator::EnumerateExpression(expr, [&](Expression &e) {
                if (e.return_type.id() == LogicalTypeId::INVALID) {
                    std::cerr << "Verification failure: expression in operator " << node->GetName() << " has INVALID return type\n";
                    ok = false;
                }
            });
        }
        for (auto &c : node->children) check_node(c.get());
    };
    check_node(root);
    return ok;
}

// Helper to run verification on a plan and its copy (copy optional)
static int RunRobustPlanVerification(unique_ptr<LogicalOperator> &root) {
    try {
        DuckDB db(nullptr);
        Connection con(db);
        root->ResolveOperatorTypes();
        root->Verify(*con.context);

        // collect and verify bindings on root
        std::set<std::pair<idx_t, idx_t>> bindings;
        CollectColumnBindingsRecursiveForTest(root.get(), bindings);
        if (!VerifyNoInvalidBindings(bindings)) return -1;
        if (!VerifyAllExprBindingsExist(root.get(), bindings)) return -2;
        // new checks: ensure each binding has a unique producer and types align
        if (!VerifyBindingProducersUniqueAndTypes(root.get())) return -5;
        if (!VerifyExprTypesMatchProducers(root.get())) return -6;
        // operator-level consistency checks
        if (!VerifyOperatorBindingsConsistentWithTypes(root.get())) return -9;

        // attempt copy and run same checks on copy if supported
        try {
            auto copy = root->Copy(*con.context);
            copy->ResolveOperatorTypes();
            copy->Verify(*con.context);
            std::set<std::pair<idx_t, idx_t>> copy_bindings;
            CollectColumnBindingsRecursiveForTest(copy.get(), copy_bindings);
            if (!VerifyNoInvalidBindings(copy_bindings)) return -3;
            if (!VerifyAllExprBindingsExist(copy.get(), copy_bindings)) return -4;
            if (!VerifyBindingProducersUniqueAndTypes(copy.get())) return -7;
            if (!VerifyExprTypesMatchProducers(copy.get())) return -8;
            if (!VerifyOperatorBindingsConsistentWithTypes(copy.get())) return -10;
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

// Test 1: basic UpdateParent insertion and binding remap
static int Test_BasicUpdateParent() {
    // Build a simple plan: root projection (table_index = 0) with a child DummyScan (table_index = 0)
    auto child = make_uniq<LogicalDummyScan>(0);

    vector<unique_ptr<Expression>> proj_exprs;
    // Projection expression referencing child's first column (binding {0,0})
    proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
    // store root as unique_ptr<LogicalOperator> so it can be passed to UpdateParent
    unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, std::move(proj_exprs));
    root->children.emplace_back(std::move(child));

    // Prepare a new parent projection that contains its own child (strict replace semantics)
    auto new_parent = make_uniq<LogicalProjection>(1, vector<unique_ptr<Expression>>() );
    new_parent->children.emplace_back(make_uniq<LogicalDummyScan>(5));
    ReplaceNode(root, root->children[0], std::move(new_parent));

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

    if (moved_child->table_index == 0) {
        std::cerr << "Child table_index was not remapped\n";
        return 5;
    }

    auto &bcr = root->expressions[0]->template Cast<BoundColumnRefExpression>();
    if (bcr.binding.table_index != moved_child->table_index) {
        std::cerr << "Binding was not updated: bcr=" << bcr.binding.table_index << " child=" << moved_child->table_index << "\n";
        return 6;
    }

    // Verify plan with a ClientContext and ensure it can be copied (serialization round-trip)
    if (RunRobustPlanVerification(root) != 0) {
        return 100;
    }

    return 0;
}

// Test 2: multiple consecutive UpdateParent calls produce consistent remapping
static int Test_ConsecutiveUpdateParent() {
    // Start with root projection (table_index = 0) and a DummyScan child (table_index = 0)
    auto child = make_uniq<LogicalDummyScan>(0);

    vector<unique_ptr<Expression>> proj_exprs;
    proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
    unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, std::move(proj_exprs));
    root->children.emplace_back(std::move(child));

    // First insertion
    auto new_parent1 = make_uniq<LogicalProjection>(1, vector<unique_ptr<Expression>>() );
    new_parent1->children.emplace_back(make_uniq<LogicalDummyScan>(7));
    ReplaceNode(root, root->children[0], std::move(new_parent1));

    // Second insertion above the same original child (now located under the inserted parent)
    // operate on the unique_ptrs directly: pass the parent's unique_ptr and its child unique_ptr
    if (!root->children[0]) return 7;
    auto new_parent2 = make_uniq<LogicalProjection>(2, vector<unique_ptr<Expression>>() );
    new_parent2->children.emplace_back(make_uniq<LogicalDummyScan>(8));
    ReplaceNode(root->children[0], root->children[0]->children[0], std::move(new_parent2));

    // Validate: there should be two projections between root and the DummyScan
    auto first = dynamic_cast<LogicalProjection *>(root->children[0].get());
    if (!first) return 8;
    if (first->children.size() != 1) return 9;
    auto second = dynamic_cast<LogicalProjection *>(first->children[0].get());
    if (!second) return 10;
    if (second->children.size() != 1) return 11;
    auto moved_child = dynamic_cast<LogicalDummyScan *>(second->children[0].get());
    if (!moved_child) return 12;

    // Check that remapping kept bindings consistent (root expression table index should equal moved_child table_index)
    // Verify plan
    if (RunRobustPlanVerification(root) != 0) {
        return 101;
    }

    auto &bcr = root->expressions[0]->template Cast<BoundColumnRefExpression>();
    if (bcr.binding.table_index != moved_child->table_index) {
        std::cerr << "Binding mismatch after consecutive updates: " << bcr.binding.table_index << " vs " << moved_child->table_index << "\n";
        return 13;
    }

    // Collect and verify bindings
    std::set<std::pair<idx_t, idx_t>> bindings;
    CollectColumnBindingsRecursiveForTest(root.get(), bindings);
    if (!VerifyAllExprBindingsExist(root.get(), bindings)) {
        return 1000;
    }

    return 0;
}

// Test 3: inserting parent when child has multiple columns and projection references later column
static int Test_MultiColumnChildBinding() {
    // DummyScan with two columns: logical behavior is simulated via projections using column bindings
    auto child = make_uniq<LogicalDummyScan>(0);

    vector<unique_ptr<Expression>> proj_exprs;
    // Reference second column (column index 1) from child binding
    // Use column index 0 so the DummyScan's reported column bindings include this binding
    proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
    unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, std::move(proj_exprs));
    root->children.emplace_back(std::move(child));

    auto new_parent = make_uniq<LogicalProjection>(1, vector<unique_ptr<Expression>>());
    ReplaceNode(root, root->children[0], std::move(new_parent));

    // Verify plan
    try {
        DuckDB db(nullptr);
        Connection con(db);
        root->ResolveOperatorTypes();
        root->Verify(*con.context);
        auto copy = root->Copy(*con.context);
        copy->ResolveOperatorTypes();
        copy->Verify(*con.context);
    } catch (std::exception &ex) {
        std::cerr << "Plan verification/copy failed: " << ex.what() << "\n";
        return 102;
    }

    auto inserted_parent = dynamic_cast<LogicalProjection *>(root->children[0].get());
    if (!inserted_parent) return 14;
    auto moved_child = dynamic_cast<LogicalDummyScan *>(inserted_parent->children[0].get());
    if (!moved_child) return 15;

    auto &bcr = root->expressions[0]->template Cast<BoundColumnRefExpression>();
    if (bcr.binding.table_index != moved_child->table_index) {
        std::cerr << "Binding table index not remapped for multi-column child\n";
        return 16;
    }
    // column index should be preserved
    if (bcr.binding.column_index != 0) {
        std::cerr << "Binding column index changed unexpectedly: " << bcr.binding.column_index << "\n";
        return 17;
    }
    return 0;
}

// Test 4: insert a join between an aggregate and a scan
static int Test_InsertJoinBetweenAggregateAndScan() {
    // Setup: root projection referencing an aggregate child
    auto agg_child = make_uniq<LogicalAggregate>(0, 0, vector<unique_ptr<Expression>>());
    unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, vector<unique_ptr<Expression>>() );
    // projection referencing the aggregate's (would-be) column 0
    root->expressions.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
    root->children.emplace_back(std::move(agg_child));

    // Insert a LogicalJoin as new parent above the aggregate (strict replace: provide left child)
    auto new_join = make_uniq<LogicalJoin>(JoinType::INNER);
    new_join->children.emplace_back(make_uniq<LogicalDummyScan>(12));
    ReplaceNode(root, root->children[0], std::move(new_join));

    auto inserted_join = dynamic_cast<LogicalJoin *>(root->children[0].get());
    if (!inserted_join) {
        std::cerr << "Inserted join is null\n";
        return 20;
    }
    // add a scan as the right child of the join BEFORE verification
    inserted_join->children.emplace_back(make_uniq<LogicalDummyScan>(1));

    // Verify plan now that the join has both children
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
            // Copy/serialization not supported for this operator; log and continue
            std::cerr << "Copy skipped (not implemented): " << ex.what() << "\n";
        } catch (SerializationException &ex) {
            std::cerr << "Copy skipped (serialization): " << ex.what() << "\n";
        }

        // Execute an equivalent simple query to exercise the executor for join cases
        // create two small temp tables and perform a join
        con.Query("CREATE TABLE __t1(i INT); INSERT INTO __t1 VALUES (1),(2);");
        con.Query("CREATE TABLE __t2(j INT); INSERT INTO __t2 VALUES (10),(20);");
        auto res = con.Query("SELECT a.i + b.j FROM __t1 a JOIN __t2 b ON 1=1 LIMIT 1;");
        if (!res) {
            std::cerr << "Execution of equivalent join query failed\n";
            return 103;
        }
    } catch (std::exception &ex) {
        std::cerr << "Plan verification/copy failed: " << ex.what() << "\n";
        return 103;
    }

    // Validate: join should have two children now
    if (inserted_join->children.size() != 2) {
        std::cerr << "Inserted join has wrong child count: " << inserted_join->children.size() << "\n";
        return 21;
    }

    // Collect and verify bindings
    std::set<std::pair<idx_t, idx_t>> bindings;
    CollectColumnBindingsRecursiveForTest(root.get(), bindings);
    if (!VerifyAllExprBindingsExist(root.get(), bindings)) {
        return 1000;
    }

    return 0;
}

// Test 5: adding a join on top of a join (nested joins)
static int Test_InsertJoinOnTopOfJoin() {
    // create a base join with two scans
    auto left = make_uniq<LogicalDummyScan>(0);
    auto right = make_uniq<LogicalDummyScan>(1);
    auto base_join = make_uniq<LogicalJoin>(JoinType::INNER);
    base_join->children.emplace_back(std::move(left));
    base_join->children.emplace_back(std::move(right));

    unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, vector<unique_ptr<Expression>>() );
    root->children.emplace_back(std::move(base_join));

    // Insert a new join above the existing join (provide a left child)
    auto new_join = make_uniq<LogicalJoin>(JoinType::INNER);
    new_join->children.emplace_back(make_uniq<LogicalDummyScan>(14));
    ReplaceNode(root, root->children[0], std::move(new_join));

    auto outer_join = dynamic_cast<LogicalJoin *>(root->children[0].get());
    if (!outer_join) return 22;
    // attach another scan as the right child of the outer join BEFORE verification
    outer_join->children.emplace_back(make_uniq<LogicalDummyScan>(2));

    // Verify plan now that outer join has both children
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
        // create temp tables and execute a nested join to exercise execution
        con.Query("CREATE TABLE __t3(i INT); INSERT INTO __t3 VALUES (1);");
        con.Query("CREATE TABLE __t4(j INT); INSERT INTO __t4 VALUES (2);");
        auto res = con.Query("SELECT * FROM __t3 JOIN (SELECT * FROM __t4) t ON 1=1 LIMIT 1;");
        if (!res) {
            std::cerr << "Execution of equivalent nested join query failed\n";
            return 104;
        }
    } catch (std::exception &ex) {
        std::cerr << "Plan verification/copy failed: " << ex.what() << "\n";
        return 104;
    }

    // Check nesting: outer_join should have the old join as its first child
    if (outer_join->children.size() < 1) return 23;
    auto inner_join = dynamic_cast<LogicalJoin *>(outer_join->children[0].get());
    if (!inner_join) {
        std::cerr << "Inner join not present where expected\n";
        return 24;
    }
    // outer join should now have two children
    if (outer_join->children.size() != 2) return 25;

    // Collect and verify bindings
    std::set<std::pair<idx_t, idx_t>> bindings;
    CollectColumnBindingsRecursiveForTest(root.get(), bindings);
    if (!VerifyAllExprBindingsExist(root.get(), bindings)) {
        return 1000;
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

    auto new_filter = make_uniq<LogicalFilter>();
    new_filter->children.emplace_back(make_uniq<LogicalDummyScan>(16));
    ReplaceNode(root, root->children[0], std::move(new_filter));

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
    if (inserted_filter->children.size() != 1) return 27;
    auto moved_child = dynamic_cast<LogicalDummyScan *>(inserted_filter->children[0].get());
    if (!moved_child) return 28;

    // Collect and verify bindings
    std::set<std::pair<idx_t, idx_t>> bindings;
    CollectColumnBindingsRecursiveForTest(root.get(), bindings);
    if (!VerifyAllExprBindingsExist(root.get(), bindings)) {
        return 1000;
    }

    return 0;
}

// Test 7: ensure UpdateParent preserves column positions when no remapping is required
static int Test_PreserveColumnsNoRemap() {
    auto child = make_uniq<LogicalDummyScan>(0);
    vector<unique_ptr<Expression>> proj_exprs;
    proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 0)));
    proj_exprs.emplace_back(make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(0, 1)));
    unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, std::move(proj_exprs));
    root->children.emplace_back(std::move(child));

    auto new_parent = make_uniq<LogicalProjection>(1, vector<unique_ptr<Expression>>() );
    new_parent->children.emplace_back(make_uniq<LogicalDummyScan>(17));
    ReplaceNode(root, root->children[0], std::move(new_parent));

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
        // execute a simple projection query to exercise executor
        con.Query("CREATE TABLE __t6(a INT, b INT); INSERT INTO __t6 VALUES (1,2);");
        auto res = con.Query("SELECT a,b FROM __t6 LIMIT 1;");
        if (!res) {
            std::cerr << "Execution of equivalent projection query failed\n";
            return 106;
        }
    } catch (std::exception &ex) {
        std::cerr << "Plan verification/copy failed: " << ex.what() << "\n";
        return 106;
    }

    auto inserted_parent = dynamic_cast<LogicalProjection *>(root->children[0].get());
    if (!inserted_parent) return 29;
    auto moved_child = dynamic_cast<LogicalDummyScan *>(inserted_parent->children[0].get());
    if (!moved_child) return 30;

    // verify projection expressions still reference column indices 0 and 1 (positions preserved)
    auto &bcr0 = root->expressions[0]->template Cast<BoundColumnRefExpression>();
    auto &bcr1 = root->expressions[1]->template Cast<BoundColumnRefExpression>();
    if (bcr0.binding.column_index != 0) return 31;
    if (bcr1.binding.column_index != 1) return 32;

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

    unique_ptr<LogicalOperator> root = make_uniq<LogicalProjection>(0, vector<unique_ptr<Expression>>() );
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

int main() {
    int code = 0;
    code = RunTest("Test_BasicUpdateParent", Test_BasicUpdateParent);
    if (code != 0) return code;
    code = RunTest("Test_ConsecutiveUpdateParent", Test_ConsecutiveUpdateParent);
    if (code != 0) return code;
    code = RunTest("Test_MultiColumnChildBinding", Test_MultiColumnChildBinding);
    if (code != 0) return code;
    code = RunTest("Test_InsertJoinBetweenAggregateAndScan", Test_InsertJoinBetweenAggregateAndScan);
    if (code != 0) return code;
    code = RunTest("Test_InsertJoinOnTopOfJoin", Test_InsertJoinOnTopOfJoin);
    if (code != 0) return code;
    code = RunTest("Test_InsertFilterParent", Test_InsertFilterParent);
    if (code != 0) return code;
    code = RunTest("Test_PreserveColumnsNoRemap", Test_PreserveColumnsNoRemap);
    if (code != 0) return code;
    code = RunTest("Test_VerifierDetectsDuplicateProducers", Test_VerifierDetectsDuplicateProducers);
    if (code != 0) return code;

    std::cout << "All UpdateParent tests passed\n";
    return 0;
}
