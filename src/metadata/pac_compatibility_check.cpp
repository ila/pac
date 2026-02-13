#include "metadata/pac_compatibility_check.hpp"
#include "pac_debug.hpp"
#include "utils/pac_helpers.hpp"
#include "parser/pac_parser.hpp"

#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

#include <algorithm>
#include "compiler/pac_compiler_helpers.hpp"
#include "core/pac_optimizer.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {

static bool IsPacAggregate(const string &func) {
	static const std::unordered_set<string> pac_aggs = {
	    "pac_sum",          "pac_count",          "pac_avg",          "pac_min",          "pac_max",
	    "pac_sum_counters", "pac_count_counters", "pac_avg_counters", "pac_min_counters", "pac_max_counters"};
	string lower_func = func;
	std::transform(lower_func.begin(), lower_func.end(), lower_func.begin(), ::tolower);
	return pac_aggs.count(lower_func) > 0;
}

static bool IsAllowedAggregate(const string &func) {
	static const std::unordered_set<string> allowed = {"sum", "sum_no_overflow", "count", "count_star", "avg", "min",
	                                                   "max"};
	string lower_func = func;
	std::transform(lower_func.begin(), lower_func.end(), lower_func.begin(), ::tolower);
	return allowed.count(lower_func) > 0 || IsPacAggregate(lower_func);
}

static bool ContainsDisallowedJoin(const LogicalOperator &op) {
	// Handle different logical join operator types that derive from LogicalJoin
	if (op.type == LogicalOperatorType::LOGICAL_JOIN || op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    op.type == LogicalOperatorType::LOGICAL_DEPENDENT_JOIN || op.type == LogicalOperatorType::LOGICAL_ASOF_JOIN ||
	    op.type == LogicalOperatorType::LOGICAL_POSITIONAL_JOIN || op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
	    op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &join = op.Cast<LogicalJoin>();
		if (join.join_type != JoinType::INNER && join.join_type != JoinType::LEFT &&
		    join.join_type != JoinType::RIGHT && join.join_type != JoinType::SEMI &&
		    join.join_type != JoinType::SINGLE && join.join_type != JoinType::ANTI &&
		    join.join_type != JoinType::RIGHT_ANTI && join.join_type != JoinType::RIGHT_SEMI &&
		    join.join_type != JoinType::MARK) {
			return true;
		}
	} else if (op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		// CROSS_PRODUCT is allowed for PAC compilation
		// Don't return true here, just continue checking children
	} else if (op.type == LogicalOperatorType::LOGICAL_EXCEPT || op.type == LogicalOperatorType::LOGICAL_INTERSECT) {
		// These operator types are disallowed for PAC compilation
		// Note: UNION, UNION ALL, CROSS_PRODUCT, and ANY_JOIN are allowed
		return true;
	}
	for (auto &child : op.children) {
		if (ContainsDisallowedJoin(*child)) {
			return true;
		}
	}
	return false;
}

static bool ContainsWindowFunction(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_WINDOW) {
		return true;
	}
	for (auto &child : op.children) {
		if (ContainsWindowFunction(*child)) {
			return true;
		}
	}
	return false;
}

static bool ContainsLogicalDistinct(const LogicalOperator &op) {
	// Only check for explicit DISTINCT operator (SELECT DISTINCT), not aggregate DISTINCT
	if (op.type == LogicalOperatorType::LOGICAL_DISTINCT) {
		return true;
	}

	for (auto &child : op.children) {
		if (ContainsLogicalDistinct(*child)) {
			return true;
		}
	}
	return false;
}

static bool ContainsAggregation(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = op.Cast<LogicalAggregate>();
		for (auto &expr : aggr.expressions) {
			if (expr && expr->IsAggregate()) {
				auto &ag = expr->Cast<BoundAggregateExpression>();
				if (IsAllowedAggregate(ag.function.name)) {
					return true;
				}
			}
		}
	}
	for (auto &child : op.children) {
		if (ContainsAggregation(*child)) {
			return true;
		}
	}
	return false;
}

// Helper: Get all table names that are scanned in a subtree (stops at subquery boundaries)
static void GetScannedTablesInScope(const LogicalOperator &op, std::unordered_set<string> &tables) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		auto table_entry = get.GetTable();
		if (table_entry) {
			tables.insert(table_entry->name);
		}
	}

	// Stop at subquery boundaries
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		// Only traverse left child (main query)
		if (!op.children.empty() && op.children[0]) {
			GetScannedTablesInScope(*op.children[0], tables);
		}
		return;
	}

	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN || op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &join = op.Cast<LogicalJoin>();
		if (join.join_type == JoinType::SINGLE || join.join_type == JoinType::MARK) {
			// Only traverse left child (main query)
			if (!op.children.empty() && op.children[0]) {
				GetScannedTablesInScope(*op.children[0], tables);
			}
			return;
		}
	}

	// Traverse children
	for (auto &child : op.children) {
		GetScannedTablesInScope(*child, tables);
	}
}

// Helper: Check if PAC aggregates in a subtree are properly joined with PU/FK path tables
// This recursively checks each aggregate scope and validates PAC aggregates
// Returns true if PAC aggregates were found in the query
static bool CheckPacAggregatesHaveProperJoins(const LogicalOperator &op, const PACCompatibilityResult &compat_result,
                                              const vector<string> &all_pu_tables) {
	bool found_pac_aggregate = false;

	// If this is an aggregate node, check if it contains PAC aggregates
	if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = op.Cast<LogicalAggregate>();

		// Check if this aggregate contains any PAC aggregates
		bool has_pac_aggregate = false;
		bool has_regular_aggregate = false;

		for (auto &expr : aggr.expressions) {
			if (expr && expr->IsAggregate()) {
				auto &ag = expr->Cast<BoundAggregateExpression>();
				if (IsPacAggregate(ag.function.name)) {
					has_pac_aggregate = true;
					found_pac_aggregate = true;
				} else if (IsAllowedAggregate(ag.function.name)) {
					has_regular_aggregate = true;
				}
			}
		}

		// If this aggregate has PAC aggregates, check that it's joined with PU or FK path tables
		if (has_pac_aggregate) {
			// Get all tables scanned in this aggregate's scope (below the aggregate)
			std::unordered_set<string> scanned_tables;
			for (auto &child : op.children) {
				GetScannedTablesInScope(*child, scanned_tables);
			}

			// Check if any PU table is scanned
			bool has_pu_table = false;
			for (auto &pu : all_pu_tables) {
				if (scanned_tables.find(pu) != scanned_tables.end()) {
					has_pu_table = true;
					break;
				}
			}

			// If no PU table is directly scanned, check if any scanned table has an FK path to a PU
			bool has_fk_to_pu = false;
			if (!has_pu_table) {
				for (auto &table : scanned_tables) {
					// Check if this table has an FK path to any PU
					auto it = compat_result.fk_paths.find(table);
					if (it != compat_result.fk_paths.end() && !it->second.empty()) {
						// The FK path should lead to a PU table (last element in the path)
						const string &target_table = it->second.back();
						for (auto &pu : all_pu_tables) {
							if (target_table == pu) {
								has_fk_to_pu = true;
								break;
							}
						}
						if (has_fk_to_pu) {
							break;
						}
					}
				}
			}

			// PAC aggregate is valid if:
			// 1. PU table is directly scanned, OR
			// 2. A scanned table has an FK path to the PU
			if (!has_pu_table && !has_fk_to_pu) {
				throw InvalidInputException(
				    "PAC rewrite: PAC aggregates (pac_sum, pac_count, etc.) must be joined with the privacy unit table "
				    "or a table that has a foreign key path to the privacy unit");
			}
		}

		// If this aggregate has regular aggregates wrapping PAC results, check children for PAC aggregates
		// This handles cases like COUNT(...) on top of a subquery with pac_count(...)
		if (has_regular_aggregate) {
			// Recursively check children - they might contain PAC aggregates in subqueries
			for (auto &child : op.children) {
				if (CheckPacAggregatesHaveProperJoins(*child, compat_result, all_pu_tables)) {
					found_pac_aggregate = true;
				}
			}
		}
	}

	// Handle subquery boundaries - check each subquery independently
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		// Check both main query (left) and subquery (right)
		for (auto &child : op.children) {
			if (CheckPacAggregatesHaveProperJoins(*child, compat_result, all_pu_tables)) {
				found_pac_aggregate = true;
			}
		}
		return found_pac_aggregate;
	}

	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN || op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &join = op.Cast<LogicalJoin>();
		if (join.join_type == JoinType::SINGLE || join.join_type == JoinType::MARK) {
			// Check both main query (left) and subquery (right)
			for (auto &child : op.children) {
				if (CheckPacAggregatesHaveProperJoins(*child, compat_result, all_pu_tables)) {
					found_pac_aggregate = true;
				}
			}
			return found_pac_aggregate;
		}
	}

	// Recurse into children for non-aggregate operators
	for (auto &child : op.children) {
		if (CheckPacAggregatesHaveProperJoins(*child, compat_result, all_pu_tables)) {
			found_pac_aggregate = true;
		}
	}

	return found_pac_aggregate;
}

// Helper: Find the operator in the plan that produces a given table_index
static LogicalOperator *FindOperatorByTableIndex(LogicalOperator &op, idx_t table_index) {
	// Check if this operator produces the table_index
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		if (get.table_index == table_index) {
			return &op;
		}
	} else if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = op.Cast<LogicalAggregate>();
		if (aggr.group_index == table_index || aggr.aggregate_index == table_index) {
			return &op;
		}
	} else if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = op.Cast<LogicalProjection>();
		if (proj.table_index == table_index) {
			return &op;
		}
	}

	// Recurse into children
	for (auto &child : op.children) {
		auto *result = FindOperatorByTableIndex(*child, table_index);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

// Trace a binding down through the plan to check if it ultimately comes from a PU table.
// If the binding comes from an aggregate expression, it's safe (the value has been aggregated).
// If the binding comes from a GROUP BY column, we need to trace that column's source further.
// If we reach a PU table column directly (or via join key equivalence), we reject.
static void TraceBindingToPUTable(LogicalOperator &op, const ColumnBinding &binding, const vector<string> &pu_tables,
                                  LogicalOperator &root) {
	// Find the operator that produces this binding's table_index
	auto *source_op = FindOperatorByTableIndex(root, binding.table_index);
	if (!source_op) {
		return; // Can't find source, assume safe
	}

	if (source_op->type == LogicalOperatorType::LOGICAL_GET) {
		// This binding comes directly from a table scan
		// Use ColumnBelongsToTable which handles join key equivalences
		for (auto &pu_table : pu_tables) {
			if (ColumnBelongsToTable(root, pu_table, binding)) {
				throw InvalidInputException(
				    "PAC rewrite: columns from privacy unit tables can only be accessed inside aggregate "
				    "functions (e.g., SUM, COUNT, AVG, MIN, MAX)");
			}
		}
	} else if (source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = source_op->Cast<LogicalAggregate>();

		// Check if this binding is from the aggregate's group output or aggregate output
		if (binding.table_index == aggr.group_index) {
			// This is a grouped column - trace it further down
			idx_t group_idx = binding.column_index;
			if (group_idx < aggr.groups.size() && aggr.groups[group_idx]) {
				// Find column refs in this group expression and trace them
				ExpressionIterator::EnumerateExpression(
				    const_cast<unique_ptr<Expression> &>(aggr.groups[group_idx]), [&](Expression &expr) {
					    if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
						    auto &col_ref = expr.Cast<BoundColumnRefExpression>();
						    TraceBindingToPUTable(*source_op, col_ref.binding, pu_tables, root);
					    }
				    });
			}
		}
		// If binding.table_index == aggr.aggregate_index, it's an aggregate result - safe, don't trace further
	} else if (source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = source_op->Cast<LogicalProjection>();
		// Trace the expression that produces this column
		if (binding.column_index < proj.expressions.size() && proj.expressions[binding.column_index]) {
			ExpressionIterator::EnumerateExpression(proj.expressions[binding.column_index], [&](Expression &expr) {
				if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = expr.Cast<BoundColumnRefExpression>();
					TraceBindingToPUTable(*source_op, col_ref.binding, pu_tables, root);
				}
			});
		}
	}
}

// Helper: Check if a binding refers to a protected column (PU PKs, LINK FKs, or metadata PROTECTED).
// Returns the table name and column name if protected, empty strings otherwise.
static std::pair<string, string>
GetProtectedColumnInfo(LogicalOperator &root, const ColumnBinding &binding,
                       const std::unordered_map<string, std::unordered_set<string>> &protected_columns) {
	// Find the LogicalGet that produces this binding
	auto *source_op = FindOperatorByTableIndex(root, binding.table_index);
	if (!source_op || source_op->type != LogicalOperatorType::LOGICAL_GET) {
		return {"", ""};
	}

	auto &get = source_op->Cast<LogicalGet>();
	auto table_entry = get.GetTable();
	if (!table_entry) {
		return {"", ""};
	}

	// Get column name from the binding
	// The binding.column_index refers to the position in the scan's output columns (GetColumnIds),
	// not the position in get.names (which contains ALL table columns)
	string col_name;
	const auto &column_ids = get.GetColumnIds();
	if (binding.column_index < column_ids.size()) {
		col_name = get.GetColumnName(column_ids[binding.column_index]);
	}
	if (col_name.empty()) {
		return {"", ""};
	}

	// Check against the unified protected_columns set
	string table_lower = StringUtil::Lower(table_entry->name);
	string col_lower = StringUtil::Lower(col_name);
	auto it = protected_columns.find(table_lower);
	if (it != protected_columns.end() && it->second.count(col_lower) > 0) {
		return {table_entry->name, col_name};
	}

	return {"", ""};
}

// Check that no PU table columns are exposed in the final query output.
// Start from the root operator's output and trace each binding down.
// plan_root is the full plan root (used for tracing bindings)
// current_op is the operator we're currently checking
// tables_with_protected_cols is a list of tables that have protected columns (should be skipped here)
static void CheckOutputColumnsNotFromPU(LogicalOperator &current_op, LogicalOperator &plan_root,
                                        const vector<string> &pu_tables,
                                        const vector<string> &tables_with_protected_cols) {
	// Create a set for quick lookup of tables with protected columns (normalized to lowercase)
	std::unordered_set<string> protected_set;
	for (const auto &t : tables_with_protected_cols) {
		protected_set.insert(StringUtil::Lower(t));
	}

	// Filter pu_tables to exclude tables with protected columns
	vector<string> actual_pu_tables;
	for (auto &pu_table : pu_tables) {
		string normalized = StringUtil::Lower(pu_table);
		if (protected_set.find(normalized) == protected_set.end()) {
			actual_pu_tables.push_back(pu_table);
		}
	}

	// If no actual PU tables remain, skip the check
	if (actual_pu_tables.empty()) {
		return;
	}

	auto trace_expressions = [&](vector<unique_ptr<Expression>> &expressions) {
		for (auto &expr : expressions) {
			if (!expr) {
				continue;
			}
			ExpressionIterator::EnumerateExpression(expr, [&](Expression &e) {
				if (e.type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = e.Cast<BoundColumnRefExpression>();
					TraceBindingToPUTable(plan_root, col_ref.binding, actual_pu_tables, plan_root);
				}
			});
		}
	};

	if (current_op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = current_op.Cast<LogicalProjection>();
		trace_expressions(proj.expressions);
	} else if (current_op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = current_op.Cast<LogicalAggregate>();
		// For root aggregate: check groups (aggregate expressions are safe by definition)
		trace_expressions(aggr.groups);
	} else if (current_op.type == LogicalOperatorType::LOGICAL_GET) {
		// Direct table scan as root - check if it's a PU table
		auto &get = current_op.Cast<LogicalGet>();
		auto table_entry = get.GetTable();
		if (table_entry) {
			for (auto &pu_table : actual_pu_tables) {
				if (table_entry->name == pu_table) {
					throw InvalidInputException(
					    "PAC rewrite: columns from privacy unit tables can only be accessed inside aggregate "
					    "functions (e.g., SUM, COUNT, AVG, MIN, MAX)");
				}
			}
		}
	} else if (current_op.type == LogicalOperatorType::LOGICAL_ORDER_BY ||
	           current_op.type == LogicalOperatorType::LOGICAL_TOP_N ||
	           current_op.type == LogicalOperatorType::LOGICAL_LIMIT) {
		// For ORDER BY, TOP N, and LIMIT: check the child operator's output
		// These operators just reorder/filter rows, they don't change the columns
		for (auto &child : current_op.children) {
			CheckOutputColumnsNotFromPU(*child, plan_root, pu_tables, tables_with_protected_cols);
		}
	}
}

// Forward declaration for protected column tracing
static void
TraceBindingForProtectedColumns(LogicalOperator &op, const ColumnBinding &binding, LogicalOperator &root,
                                const std::unordered_map<string, std::unordered_set<string>> &protected_columns);

// Trace a binding to check if it comes from a protected column (unified set)
static void
TraceBindingForProtectedColumns(LogicalOperator &op, const ColumnBinding &binding, LogicalOperator &root,
                                const std::unordered_map<string, std::unordered_set<string>> &protected_columns) {
	auto *source_op = FindOperatorByTableIndex(root, binding.table_index);
	if (!source_op) {
		return;
	}

	if (source_op->type == LogicalOperatorType::LOGICAL_GET) {
		// Check if this is a protected column
		std::pair<string, string> protected_info = GetProtectedColumnInfo(root, binding, protected_columns);
		if (!protected_info.first.empty()) {
			throw InvalidInputException("PAC rewrite: protected column '%s.%s' can only be accessed inside aggregate "
			                            "functions (e.g., SUM, COUNT, AVG, MIN, MAX)",
			                            protected_info.first.c_str(), protected_info.second.c_str());
		}
	} else if (source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = source_op->Cast<LogicalAggregate>();
		if (binding.table_index == aggr.group_index) {
			idx_t group_idx = binding.column_index;
			if (group_idx < aggr.groups.size() && aggr.groups[group_idx]) {
				ExpressionIterator::EnumerateExpression(
				    const_cast<unique_ptr<Expression> &>(aggr.groups[group_idx]), [&](Expression &expr) {
					    if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
						    auto &col_ref = expr.Cast<BoundColumnRefExpression>();
						    TraceBindingForProtectedColumns(*source_op, col_ref.binding, root, protected_columns);
					    }
				    });
			}
		}
	} else if (source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = source_op->Cast<LogicalProjection>();
		if (binding.column_index < proj.expressions.size() && proj.expressions[binding.column_index]) {
			ExpressionIterator::EnumerateExpression(proj.expressions[binding.column_index], [&](Expression &expr) {
				if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = expr.Cast<BoundColumnRefExpression>();
					TraceBindingForProtectedColumns(*source_op, col_ref.binding, root, protected_columns);
				}
			});
		}
	}
}

// Check that no protected columns (unified set) are exposed in the final query output
static void
CheckOutputColumnsNotProtected(LogicalOperator &current_op, LogicalOperator &plan_root,
                               const std::unordered_map<string, std::unordered_set<string>> &protected_columns) {
	if (protected_columns.empty()) {
		return;
	}

	auto trace_expressions = [&](vector<unique_ptr<Expression>> &expressions) {
		for (auto &expr : expressions) {
			if (!expr) {
				continue;
			}
			ExpressionIterator::EnumerateExpression(expr, [&](Expression &e) {
				if (e.type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = e.Cast<BoundColumnRefExpression>();
					TraceBindingForProtectedColumns(plan_root, col_ref.binding, plan_root, protected_columns);
				}
			});
		}
	};

	if (current_op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = current_op.Cast<LogicalProjection>();
		trace_expressions(proj.expressions);
	} else if (current_op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = current_op.Cast<LogicalAggregate>();
		trace_expressions(aggr.groups);
	} else if (current_op.type == LogicalOperatorType::LOGICAL_GET) {
		// Direct table scan - check if any scanned columns are protected
		auto &get = current_op.Cast<LogicalGet>();
		auto table_entry = get.GetTable();
		if (table_entry) {
			string table_lower = StringUtil::Lower(table_entry->name);
			auto it = protected_columns.find(table_lower);
			if (it != protected_columns.end()) {
				const auto &column_ids = get.GetColumnIds();
				for (const auto &col_idx : column_ids) {
					const string &col_name = get.GetColumnName(col_idx);
					string col_lower = StringUtil::Lower(col_name);
					if (it->second.count(col_lower) > 0) {
						throw InvalidInputException(
						    "PAC rewrite: protected column '%s.%s' can only be accessed inside aggregate "
						    "functions (e.g., SUM, COUNT, AVG, MIN, MAX)",
						    table_entry->name.c_str(), col_name.c_str());
					}
				}
			}
		}
	} else if (current_op.type == LogicalOperatorType::LOGICAL_ORDER_BY ||
	           current_op.type == LogicalOperatorType::LOGICAL_TOP_N ||
	           current_op.type == LogicalOperatorType::LOGICAL_LIMIT) {
		for (auto &child : current_op.children) {
			CheckOutputColumnsNotProtected(*child, plan_root, protected_columns);
		}
	}
}

// Check that no protected columns appear in HAVING filter expressions.
// Protected columns can appear in WHERE filters (below aggregate) but not in HAVING filters (above aggregate).
// On the raw binder plan, both WHERE and HAVING are LOGICAL_FILTER nodes, so we track whether
// we've descended past an aggregate to distinguish them.
static void
CheckFiltersNotUsingProtectedColumns(LogicalOperator &op, LogicalOperator &plan_root,
                                     const std::unordered_map<string, std::unordered_set<string>> &protected_columns,
                                     bool above_aggregate = true) {
	if (protected_columns.empty()) {
		return;
	}

	if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		// Below this point, filters are WHERE clauses (allowed to reference protected columns)
		above_aggregate = false;
	}

	if (op.type == LogicalOperatorType::LOGICAL_FILTER && above_aggregate) {
		// This is a HAVING filter — check for protected column references
		for (auto &expr : op.expressions) {
			if (!expr) {
				continue;
			}
			ExpressionIterator::EnumerateExpression(expr, [&](Expression &e) {
				if (e.type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = e.Cast<BoundColumnRefExpression>();
					auto protected_info = GetProtectedColumnInfo(plan_root, col_ref.binding, protected_columns);
					if (!protected_info.first.empty()) {
						throw InvalidInputException(
						    "PAC rewrite: protected column '%s.%s' cannot be used in WHERE/HAVING filters. "
						    "Protected columns can only appear inside aggregate functions (e.g., SUM, COUNT, AVG).",
						    protected_info.first.c_str(), protected_info.second.c_str());
					}
				}
			});
		}
	}

	for (auto &child : op.children) {
		if (child) {
			CheckFiltersNotUsingProtectedColumns(*child, plan_root, protected_columns, above_aggregate);
		}
	}
}

// helper: traverse the plan and count how many times each table/CTE name is scanned
void CountScans(const LogicalOperator &op, std::unordered_map<string, idx_t> &counts);

// helper: traverse the plan and count how many times each table/CTE name is scanned GLOBALLY (including subqueries)
void CountScans(const LogicalOperator &op, std::unordered_map<string, idx_t> &counts) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &scan = op.Cast<LogicalGet>();
		auto table_entry = scan.GetTable();
		if (table_entry) {
			counts[table_entry->name]++;
		}
	}
	// Handle CTEs: traverse into CTE definitions to find base table scans
	if (op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
		auto &cte = op.Cast<LogicalMaterializedCTE>();
		if (!cte.children.empty() && cte.children[0]) {
			CountScans(*cte.children[0], counts);
		}
		if (cte.children.size() > 1 && cte.children[1]) {
			CountScans(*cte.children[1], counts);
		}
		return;
	}
	// For global counting, traverse ALL children including subqueries
	for (auto &child : op.children) {
		CountScans(*child, counts);
	}
}

/**
 * DetectCycleInFKGraph: Detects cycles in the foreign key graph
 *
 * A cycle exists when following foreign keys from a table can eventually lead back to itself.
 * For example: A -> B -> C -> A forms a cycle.
 *
 * This is important because PAC compilation follows FK paths from scanned tables to privacy units,
 * and cycles would cause infinite loops during path traversal.
 *
 * @param context - Client context for accessing catalog
 * @param start_tables - Tables to start cycle detection from (typically scanned tables)
 * @return true if a cycle is detected, false otherwise
 */
static bool DetectCycleInFKGraph(ClientContext &context, const vector<string> &start_tables) {
	// Build adjacency list for the FK graph
	std::unordered_map<string, vector<string>> graph;
	std::unordered_set<string> all_tables;

	// Start with the initial tables
	std::queue<string> to_process;
	for (auto &table : start_tables) {
		to_process.push(table);
		all_tables.insert(table);
	}

	// Build the FK graph by following all FK edges
	while (!to_process.empty()) {
		string current = to_process.front();
		to_process.pop();

		// Get foreign keys from this table
		auto fks = FindForeignKeys(context, current);
		for (auto &fk : fks) {
			string referenced_table = fk.first;

			// Add edge to graph
			graph[current].push_back(referenced_table);

			// Add referenced table to processing queue if not seen before
			if (all_tables.find(referenced_table) == all_tables.end()) {
				all_tables.insert(referenced_table);
				to_process.push(referenced_table);
			}
		}
	}

	// Perform DFS-based cycle detection using three-color algorithm
	// WHITE (0): unvisited, GRAY (1): being processed, BLACK (2): fully processed
	std::unordered_map<string, int> colors;
	for (auto &table : all_tables) {
		colors[table] = 0; // WHITE
	}

	// DFS helper function
	std::function<bool(const string &)> has_cycle_dfs = [&](const string &node) -> bool {
		colors[node] = 1; // GRAY - currently processing

		// Visit all neighbors
		auto it = graph.find(node);
		if (it != graph.end()) {
			for (auto &neighbor : it->second) {
				if (colors[neighbor] == 1) {
					// Back edge detected - cycle found
					return true;
				}
				if (colors[neighbor] == 0) {
					// Unvisited - recurse
					if (has_cycle_dfs(neighbor)) {
						return true;
					}
				}
				// If neighbor is BLACK (2), no need to visit (already fully processed)
			}
		}

		colors[node] = 2; // BLACK - fully processed
		return false;
	};

	// Run DFS from each unvisited node
	for (auto &table : all_tables) {
		if (colors[table] == 0) {
			if (has_cycle_dfs(table)) {
				return true;
			}
		}
	}

	return false;
}

PACCompatibilityResult PACRewriteQueryCheck(unique_ptr<LogicalOperator> &plan, ClientContext &context,
                                            const vector<string> &pac_tables, PACOptimizerInfo *optimizer_info) {
	PACCompatibilityResult result;

	// If a replan/compilation is already in progress by the optimizer extension, skip compatibility checks
	// to avoid re-entrant behavior and infinite loops.
	if (optimizer_info && optimizer_info->replan_in_progress.load(std::memory_order_acquire)) {
		return result;
	}

	// count all scanned tables/CTEs in the plan
	std::unordered_map<string, idx_t> scan_counts;
	CountScans(*plan, scan_counts);

	// Record which configured PAC tables were scanned in this plan
	for (auto &t : pac_tables) {
		if (scan_counts[t] > 0) {
			result.scanned_pu_tables.push_back(t);
		}
	}

	// Build a vector of scanned table names
	vector<string> scanned_tables;
	for (auto &kv : scan_counts) {
		scanned_tables.push_back(kv.first);
	}
	// Sort for deterministic behavior across platforms (unordered_map iteration order is not guaranteed)
	std::sort(scanned_tables.begin(), scanned_tables.end());

	// Record scanned tables that are NOT configured PAC tables
	// This is needed for the compiler to correctly identify present tables
	std::unordered_set<string> pac_tables_set(pac_tables.begin(), pac_tables.end());
	for (auto &kv : scan_counts) {
		if (kv.second > 0 && pac_tables_set.find(kv.first) == pac_tables_set.end()) {
			result.scanned_non_pu_tables.push_back(kv.first);
		}
	}
	// Sort for deterministic behavior across platforms
	std::sort(result.scanned_non_pu_tables.begin(), result.scanned_non_pu_tables.end());

	// Discover tables with PROTECTED columns in PAC metadata
	// Note: Tables with protected columns are NOT automatically privacy units anymore.
	// A table is a privacy unit only if it has is_privacy_unit = true (via CREATE PAC TABLE or ALTER TABLE SET PAC)
	auto &metadata_mgr = PACMetadataManager::Get();
	vector<string> tables_with_protected_columns;
	for (auto &kv : scan_counts) {
		if (kv.second > 0) {
			auto *table_metadata = metadata_mgr.GetTableMetadata(kv.first);
			if (table_metadata && !table_metadata->protected_columns.empty()) {
				tables_with_protected_columns.push_back(kv.first);
			}
			// Check if this table is explicitly marked as a privacy unit
			if (table_metadata && table_metadata->is_privacy_unit) {
				if (std::find(result.scanned_pu_tables.begin(), result.scanned_pu_tables.end(), kv.first) ==
				    result.scanned_pu_tables.end()) {
					result.scanned_pu_tables.push_back(kv.first);
				}
			}
		}
	}
	// Sort for deterministic behavior across platforms
	std::sort(tables_with_protected_columns.begin(), tables_with_protected_columns.end());
	std::sort(result.scanned_pu_tables.begin(), result.scanned_pu_tables.end());

	// Also check tables reachable via PAC_LINKs for protected columns
	// (FindForeignKeys already includes PAC_LINKs, but we need to find protected columns
	// in tables that may not be directly scanned)
	{
		std::unordered_set<string> visited;
		std::queue<string> to_check;
		for (auto &t : scanned_tables) {
			visited.insert(t);
			to_check.push(t);
		}
		while (!to_check.empty()) {
			string current = to_check.front();
			to_check.pop();

			// Get outgoing links (both FK and PAC_LINK)
			auto fks = FindForeignKeys(context, current);
			for (auto &fk : fks) {
				string ref_table = fk.first;
				if (visited.find(ref_table) != visited.end()) {
					continue;
				}
				visited.insert(ref_table);
				to_check.push(ref_table);

				// Check if referenced table has protected columns
				auto *ref_metadata = metadata_mgr.GetTableMetadata(ref_table);
				if (ref_metadata && !ref_metadata->protected_columns.empty()) {
					if (std::find(tables_with_protected_columns.begin(), tables_with_protected_columns.end(),
					              ref_table) == tables_with_protected_columns.end()) {
						tables_with_protected_columns.push_back(ref_table);
					}
				}
			}
		}
	}

	// Store in result
	result.tables_with_protected_columns = tables_with_protected_columns;
	bool has_protected_columns = !tables_with_protected_columns.empty();

	// Build the combined privacy unit list:
	// 1. Configured PAC tables (pac_tables from the CSV file)
	// 2. Tables explicitly marked as privacy units (is_privacy_unit = true via CREATE PAC TABLE or ALTER TABLE SET PAC)
	// Note: Tables with protected columns are NOT automatically privacy units anymore
	// A table is a privacy unit only if it has is_privacy_unit = true (via CREATE PAC TABLE or ALTER TABLE SET PAC)
	vector<string> all_privacy_units = pac_tables;
	for (auto &t : result.scanned_pu_tables) {
		if (std::find(all_privacy_units.begin(), all_privacy_units.end(), t) == all_privacy_units.end()) {
			all_privacy_units.push_back(t);
		}
	}
	// Also include ALL tables from metadata that have is_privacy_unit=true (even if not scanned)
	for (const auto &table_name : metadata_mgr.GetAllTableNames()) {
		auto *table_metadata = metadata_mgr.GetTableMetadata(table_name);
		if (table_metadata && table_metadata->is_privacy_unit) {
			if (std::find(all_privacy_units.begin(), all_privacy_units.end(), table_name) == all_privacy_units.end()) {
				all_privacy_units.push_back(table_name);
			}
		}
	}

	// --- Populate per-table metadata (PKs and FKs) for scanned tables ---
	for (auto &name : scanned_tables) {
		ColumnMetadata md;
		md.table_name = name;
		md.pks = FindPrimaryKey(context, name);
		md.fks = FindForeignKeys(context, name);
		result.table_metadata[name] = std::move(md);
	}

	// Compute FK/LINK paths from scanned tables to any privacy unit (transitive)
	// FindForeignKeyBetween uses FindForeignKeys which already includes PAC_LINKs
	auto fk_paths = FindForeignKeyBetween(context, all_privacy_units, scanned_tables);

	// Populate metadata for tables in FK paths that aren't scanned
	for (auto &kv : fk_paths) {
		for (auto &tbl : kv.second) {
			if (result.table_metadata.find(tbl) == result.table_metadata.end()) {
				ColumnMetadata md;
				md.table_name = tbl;
				md.pks = FindPrimaryKey(context, tbl);
				md.fks = FindForeignKeys(context, tbl);
				result.table_metadata[tbl] = std::move(md);
			}
		}
	}

	// Ensure PU tables have metadata populated
	for (auto &t : result.scanned_pu_tables) {
		if (result.table_metadata.find(t) == result.table_metadata.end()) {
			ColumnMetadata md;
			md.table_name = t;
			md.pks = FindPrimaryKey(context, t);
			md.fks = FindForeignKeys(context, t);
			result.table_metadata[t] = std::move(md);
		} else if (result.table_metadata[t].pks.empty()) {
			auto pk = FindPrimaryKey(context, t);
			if (!pk.empty()) {
				result.table_metadata[t].pks = pk;
			}
		}
	}

	// Infer PKs for PU tables from incoming PAC_LINK references.
	// If a PU table has no PK from DB constraints or PAC_KEY, check if any
	// other table has a PAC_LINK that REFERENCES this PU table. The
	// link.referenced_columns identify the key columns of the PU.
	for (auto &pu_table : result.scanned_pu_tables) {
		auto pu_it = result.table_metadata.find(pu_table);
		if (pu_it == result.table_metadata.end() || !pu_it->second.pks.empty()) {
			continue;
		}
		string pu_lower = StringUtil::Lower(pu_table);
		bool found = false;
		for (const auto &other_table_name : metadata_mgr.GetAllTableNames()) {
			if (found) {
				break;
			}
			auto *other_meta = metadata_mgr.GetTableMetadata(other_table_name);
			if (!other_meta) {
				continue;
			}
			for (auto &link : other_meta->links) {
				if (StringUtil::Lower(link.referenced_table) == pu_lower && !link.referenced_columns.empty()) {
					pu_it->second.pks = link.referenced_columns;
					found = true;
					break;
				}
			}
		}
	}

	// Attach discovered fk_paths to the result
	result.fk_paths = std::move(fk_paths);

	// --- Build the unified protected_columns set ---
	// Source 1: PU PK columns
	for (auto &pu_table : result.scanned_pu_tables) {
		auto it = result.table_metadata.find(pu_table);
		if (it != result.table_metadata.end()) {
			for (auto &pk : it->second.pks) {
				result.protected_columns[StringUtil::Lower(pu_table)].insert(StringUtil::Lower(pk));
			}
		}
	}
	// Source 2: FK columns that transitively reach a PU (via fk_paths or direct FK to PU)
	for (auto &kv : result.table_metadata) {
		const string &table_name = kv.first;
		for (auto &fk_pair : kv.second.fks) {
			const string &ref_table = fk_pair.first;
			string ref_lower = StringUtil::Lower(ref_table);
			// Check if ref_table is a PU directly
			bool reaches_pu = false;
			for (auto &pu : all_privacy_units) {
				if (StringUtil::Lower(pu) == ref_lower) {
					reaches_pu = true;
					break;
				}
			}
			// Or check if ref_table has an fk_path to a PU
			if (!reaches_pu) {
				auto path_it = result.fk_paths.find(ref_table);
				if (path_it != result.fk_paths.end() && !path_it->second.empty()) {
					reaches_pu = true;
				}
			}
			if (reaches_pu) {
				string table_lower = StringUtil::Lower(table_name);
				for (auto &fk_col : fk_pair.second) {
					result.protected_columns[table_lower].insert(StringUtil::Lower(fk_col));
				}
			}
		}
	}
	// Source 3: Metadata PROTECTED columns
	for (const auto &table_name : metadata_mgr.GetAllTableNames()) {
		auto *meta = metadata_mgr.GetTableMetadata(table_name);
		if (meta && !meta->protected_columns.empty()) {
			string table_lower = StringUtil::Lower(table_name);
			for (auto &col : meta->protected_columns) {
				result.protected_columns[table_lower].insert(StringUtil::Lower(col));
			}
		}
	}

	// Determine if we have tables linked to privacy units
	bool has_fk_linked_tables = !result.fk_paths.empty();

#if PAC_DEBUG
	PAC_DEBUG_PRINT("PAC compatibility check: scanned_pu_tables = " + std::to_string(result.scanned_pu_tables.size()));
	PAC_DEBUG_PRINT("PAC compatibility check: tables_with_protected_columns = " +
	                std::to_string(tables_with_protected_columns.size()));
	PAC_DEBUG_PRINT("PAC compatibility check: fk_paths = " + std::to_string(result.fk_paths.size()));
	for (auto &kv : result.fk_paths) {
		string path_str = kv.first + " -> ";
		for (auto &p : kv.second) {
			path_str += p + " -> ";
		}
		PAC_DEBUG_PRINT("  path: " + path_str);
	}
#endif

	// Check for protected columns FIRST (before other structural checks)
	// This ensures we get the correct error message for protected column violations.
	// The unified set covers PU PKs, LINK FKs, and metadata PROTECTED columns.
	if (!result.protected_columns.empty()) {
		CheckOutputColumnsNotProtected(*plan, *plan, result.protected_columns);
		CheckFiltersNotUsingProtectedColumns(*plan, *plan, result.protected_columns);
	}

	// Structural checks BEFORE deciding eligibility (throw when invalid)
	// These checks must run for ALL queries that:
	// - scan privacy unit tables directly, OR
	// - scan tables linked to PU via FK/LINK paths
	//
	// However, if ALL scanned privacy unit tables have protected columns defined,
	// then we allow non-aggregated queries on non-protected columns.
	// The CheckOutputColumnsNotProtected check already ensures protected columns
	// can only be accessed inside aggregates.
	bool all_pu_tables_have_protected_columns = true;
	for (const auto &pu_table : result.scanned_pu_tables) {
		if (std::find(tables_with_protected_columns.begin(), tables_with_protected_columns.end(), pu_table) ==
		    tables_with_protected_columns.end()) {
			all_pu_tables_have_protected_columns = false;
			break;
		}
	}

	// Only enforce structural checks (including aggregation requirement) if:
	// - There are FK-linked tables, OR
	// - There are scanned PU tables that do NOT have protected columns defined
	bool requires_structural_checks =
	    has_fk_linked_tables || (!result.scanned_pu_tables.empty() && !all_pu_tables_have_protected_columns);

	if (requires_structural_checks) {
		// Get conservative mode setting
		bool is_conservative = GetBooleanSetting(context, "pac_conservative_mode", true);

		// Check for cycles in the FK graph FIRST
		// This prevents infinite loops during FK path traversal
		if (DetectCycleInFKGraph(context, scanned_tables)) {
			if (is_conservative) {
				throw InvalidInputException("PAC rewrite: circular foreign key dependencies detected. "
				                            "PAC compilation requires acyclic foreign key relationships.");
			}
			return result;
		}

		if (ContainsWindowFunction(*plan)) {
			if (is_conservative) {
				throw InvalidInputException("PAC rewrite: window functions are not supported for PAC compilation");
			}
			return result;
		}
		if (!ContainsAggregation(*plan)) {
			if (is_conservative) {
				throw InvalidInputException(
				    "Query does not contain any allowed aggregation (sum, count, avg, min, max)!");
			}
			return result;
		}
		if (ContainsLogicalDistinct(*plan)) {
			if (is_conservative) {
				throw InvalidInputException("PAC rewrite: DISTINCT is not supported for PAC compilation");
			}
			return result;
		}
		if (ContainsDisallowedJoin(*plan)) {
			if (is_conservative) {
				throw InvalidInputException("PAC rewrite: subqueries are not supported for PAC compilation");
			}
			return result;
		}

		// Check that GROUP BY columns don't come from PU tables
		// NOTE: The plan is already optimized without COLUMN_LIFETIME and COMPRESSED_MATERIALIZATION
		// because the pre-optimizer disabled them before built-in optimizers ran.
		if (!result.scanned_pu_tables.empty()) {
			CheckOutputColumnsNotFromPU(*plan, *plan, result.scanned_pu_tables, tables_with_protected_columns);
		}

		// Check that PAC aggregates are properly joined with PU or FK path tables
		// This validates that pac_sum, pac_count, etc. have access to the privacy unit
		// Returns true if PAC aggregates were found
		bool has_pac_aggregates = false;
		if (!all_privacy_units.empty()) {
			has_pac_aggregates = CheckPacAggregatesHaveProperJoins(*plan, result, all_privacy_units);
		}

		// If the query already has PAC aggregates with proper joins, don't trigger rewrite
		// The query is already using PAC functions correctly, so allow it as-is
		if (has_pac_aggregates) {
#if PAC_DEBUG
			PAC_DEBUG_PRINT("PAC compatibility check: Query has PAC aggregates with proper joins - allowing as-is");
			PAC_DEBUG_PRINT("=== QUERY PLAN (PAC aggregates with joins) ===");
			plan->Print();
			PAC_DEBUG_PRINT("=== END QUERY PLAN ===");
#endif
			// Return empty result to skip PAC compilation
			result.eligible_for_rewrite = false;
			return result;
		}
	}

	// Trigger PAC compilation if we have FK/LINK paths or scanned PU tables
	// For PU tables with protected columns:
	// - Non-aggregation queries on non-protected columns are allowed without PAC compilation
	// - Aggregation queries still need PAC compilation to add noise
	if (has_fk_linked_tables) {
		result.eligible_for_rewrite = true;
		return result;
	}

	if (!result.scanned_pu_tables.empty()) {
		// If query contains aggregation, we need PAC compilation even if all PU tables have protected columns
		// because aggregates need to be noised
		if (ContainsAggregation(*plan)) {
			result.eligible_for_rewrite = true;
			return result;
		}
		// For non-aggregation queries, only trigger PAC compilation if some PU tables
		// don't have protected columns (those require aggregation)
		if (!all_pu_tables_have_protected_columns) {
			result.eligible_for_rewrite = true;
			return result;
		}
		// All PU tables have protected columns and query has no aggregation
		// This is allowed (accessing non-protected columns) - no PAC compilation needed
	}

	if (result.fk_paths.empty() && result.scanned_pu_tables.empty() && !has_protected_columns) {
		// No FK paths, no scanned PAC tables, and no protected columns: nothing to do
		return result;
	}

	// If we reach here with protected columns but no PU tables and no aggregation,
	// the query is accessing non-protected columns from tables with protected columns defined.
	// If there's aggregation, we still need PAC compilation.
	if (has_protected_columns && ContainsAggregation(*plan)) {
		result.eligible_for_rewrite = true;
	}

	return result;
}

} // namespace duckdb
