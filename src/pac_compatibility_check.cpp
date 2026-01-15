#include "include/pac_compatibility_check.hpp"
#include "include/pac_helpers.hpp"

#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

#include <algorithm>
#include <pac_compiler_helpers.hpp>
#include <pac_optimizer.hpp>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {

// Helper: Get boolean setting value with default
static bool GetBooleanSetting(ClientContext &context, const string &setting_name, bool default_value) {
	Value val;
	if (context.TryGetCurrentSetting(setting_name, val) && !val.IsNull()) {
		return val.GetValue<bool>();
	}
	return default_value;
}

static bool IsAllowedAggregate(const string &func) {
	static const std::unordered_set<string> allowed = {"sum", "sum_no_overflow", "count", "count_star", "avg", "min",
	                                                   "max"};
	string lower_func = func;
	std::transform(lower_func.begin(), lower_func.end(), lower_func.begin(), ::tolower);
	return allowed.count(lower_func) > 0;
}

static bool ContainsDisallowedJoin(const LogicalOperator &op) {
	// Handle different logical join operator types that derive from LogicalJoin
	if (op.type == LogicalOperatorType::LOGICAL_JOIN || op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    op.type == LogicalOperatorType::LOGICAL_DEPENDENT_JOIN || op.type == LogicalOperatorType::LOGICAL_ASOF_JOIN ||
	    op.type == LogicalOperatorType::LOGICAL_POSITIONAL_JOIN || op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		auto &join = op.Cast<LogicalJoin>();
		// Allow INNER and LEFT joins; reject all others
		if (join.join_type != JoinType::INNER && join.join_type != JoinType::LEFT &&
		    join.join_type != JoinType::RIGHT && join.join_type != JoinType::SEMI) {
			// Non-inner/left join detected: signal disallowed join via boolean return to let caller throw
			return true;
		}
	} else if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN || op.type == LogicalOperatorType::LOGICAL_ANY_JOIN ||
	           op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT ||
	           op.type == LogicalOperatorType::LOGICAL_EXCEPT || op.type == LogicalOperatorType::LOGICAL_INTERSECT) {
		// These operator types are disallowed for PAC compilation
		// Note: UNION and UNION ALL are allowed
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

// Forward declaration
static void TraceBindingToPUTable(LogicalOperator &op, const ColumnBinding &binding, const vector<string> &pu_tables,
                                  LogicalOperator &root);

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

// Check that no PU table columns are exposed in the final query output.
// Start from the root operator's output and trace each binding down.
// plan_root is the full plan root (used for tracing bindings)
// current_op is the operator we're currently checking
static void CheckOutputColumnsNotFromPU(LogicalOperator &current_op, LogicalOperator &plan_root,
                                        const vector<string> &pu_tables) {
	auto trace_expressions = [&](vector<unique_ptr<Expression>> &expressions) {
		for (auto &expr : expressions) {
			if (!expr) {
				continue;
			}
			ExpressionIterator::EnumerateExpression(expr, [&](Expression &e) {
				if (e.type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = e.Cast<BoundColumnRefExpression>();
					TraceBindingToPUTable(plan_root, col_ref.binding, pu_tables, plan_root);
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
			for (auto &pu_table : pu_tables) {
				if (table_entry->name == pu_table) {
					throw InvalidInputException(
					    "PAC rewrite: columns from privacy unit tables can only be accessed inside aggregate "
					    "functions (e.g., SUM, COUNT, AVG, MIN, MAX)");
				}
			}
		}
	} else if (current_op.type == LogicalOperatorType::LOGICAL_ORDER_BY) {
		// For ORDER BY, check the child operator's output
		for (auto &child : current_op.children) {
			CheckOutputColumnsNotFromPU(*child, plan_root, pu_tables);
		}
	}
}

// helper: traverse the plan and count how many times each table/CTE name is scanned
void CountScans(const LogicalOperator &op, std::unordered_map<string, idx_t> &counts) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &scan = op.Cast<LogicalGet>();
		auto table_entry = scan.GetTable();
		if (table_entry) {
			counts[table_entry->name]++;
		}
	}
	for (auto &child : op.children) {
		CountScans(*child, counts);
	}
}

// Helper: check if the plan contains self-joins of privacy unit tables
// (same PU table scanned multiple times)
static bool ContainsSelfJoinOfPU(const LogicalOperator &op, const vector<string> &pu_tables) {
	std::unordered_map<string, idx_t> scan_counts;
	CountScans(op, scan_counts);

	// Create a set of PU table names for quick lookup
	std::unordered_set<string> pu_set(pu_tables.begin(), pu_tables.end());

	// Check if any PU table is scanned more than once
	for (auto &kv : scan_counts) {
		// Skip internal PAC sample tables
		if (kv.first.rfind("_pac_internal_sample_", 0) == 0) {
			continue;
		}
		// Only check PU tables
		if (pu_set.find(kv.first) != pu_set.end() && kv.second > 1) {
			return true;
		}
	}
	return false;
}

// Helper: check if any expression in a vector contains subqueries
static bool ExpressionsContainSubquery(const vector<unique_ptr<Expression>> &expressions) {
	for (auto &expr : expressions) {
		if (expr) {
			bool has_subquery = false;
			ExpressionIterator::EnumerateExpression(const_cast<unique_ptr<Expression> &>(expr), [&](Expression &e) {
				if (e.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY) {
					has_subquery = true;
				}
			});
			if (has_subquery) {
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

	// Record which configured PAC tables were scanned in this plan. The optimizer
	// rule will still want to compile queries that directly read from a privacy
	// unit even when no FK paths or PKs were discovered.
	for (auto &t : pac_tables) {
		if (scan_counts[t] > 0) {
			result.scanned_pu_tables.push_back(t);
		}
	}

	// For each scanned PAC table, require that the corresponding sample-CTE
	// "_pac_internal_sample_<table_name>" is scanned the same number of times.
	// If all PAC tables have matching sample scans, then nothing to do (return empty result).
	// If some tables have PAC scans but zero sample scans, we proceed to check eligibility.
	// If there is a mismatch where both counts are non-zero but unequal, that's an error.
	bool all_matched = true;
	for (auto &t : pac_tables) {
		idx_t pac_count = scan_counts[t];
		if (pac_count == 0) {
			continue;
		}
		string sample_name = string("_pac_internal_sample_") + t;
		idx_t sample_count = 0;
		auto it = scan_counts.find(sample_name);
		if (it != scan_counts.end()) {
			sample_count = it->second;
		}
		if (pac_count != sample_count) {
			all_matched = false;
			if (sample_count == 0) {
				// only PAC table scanned for this table, proceed to eligibility checks
			} else {
				// mismatch where both are non-zero -> ambiguous/invalid plan for PAC rewriting
				throw InvalidInputException(
				    "PAC rewrite: mismatch between PAC table scans (%s=%llu) and internal sample scans (%s=%llu)",
				    t.c_str(), pac_count, sample_name.c_str(), sample_count);
			}
		}
	}

	if (all_matched && !result.scanned_pu_tables.empty()) {
		// All PAC tables already have corresponding sample CTE scans the same number of times.
		// Nothing for the PAC rewriter to do.
		return result;
	}

	// Build a vector of scanned table names to check FK links
	vector<string> scanned_tables;
	for (auto &kv : scan_counts) {
		scanned_tables.push_back(kv.first);
	}

	// Populate scanned_non_pac_tables: scanned tables that are not in the configured pac_tables
	// and are not internal sample tables (named _pac_internal_sample_<table>). This is useful
	// to know which external/non-PAC tables were read by the query.
	std::unordered_set<string> pac_set(pac_tables.begin(), pac_tables.end());
	for (auto &name : scanned_tables) {
		if (name.rfind("_pac_internal_sample_", 0) == 0) {
			// internal sample table, skip
			continue;
		}
		if (pac_set.find(name) == pac_set.end()) {
			result.scanned_non_pu_tables.push_back(name);
		}
	}

	// --- Populate per-table metadata (PKs and FKs) for scanned tables ---
	for (auto &name : scanned_tables) {
		if (name.rfind("_pac_internal_sample_", 0) == 0) {
			continue; // skip internal sample tables
		}
		ColumnMetadata md;
		md.table_name = name;
		// primary keys (may be empty)
		auto pk = FindPrimaryKey(context, name);
		md.pks = pk;
		// foreign keys declared on this table
		auto fks = FindForeignKeys(context, name);
		md.fks = fks;
		result.table_metadata[name] = std::move(md);
	}

	// Compute FK paths from scanned tables to any privacy unit (transitive)
	auto fk_paths = FindForeignKeyBetween(context, pac_tables, scanned_tables);

	// Populate metadata (PKs/FKs) for every table that appears on any discovered FK path.
	// Compatibility check should provide metadata for scanned tables already; for any path
	// tables that were not scanned we must populate metadata here so downstream consumers
	// (the bitslice compiler) can rely solely on `result.table_metadata` without further
	// catalog lookups.
	for (auto &kv : fk_paths) {
		auto &path = kv.second;
		for (auto &tbl : path) {
			if (result.table_metadata.find(tbl) == result.table_metadata.end()) {
				ColumnMetadata md;
				md.table_name = tbl;
				auto pk = FindPrimaryKey(context, tbl);
				md.pks = pk;
				auto fks = FindForeignKeys(context, tbl);
				md.fks = fks;
				result.table_metadata[tbl] = std::move(md);
			} else {
				// if metadata exists but pks empty, try to fill
				if (result.table_metadata[tbl].pks.empty()) {
					auto pk = FindPrimaryKey(context, tbl);
					if (!pk.empty()) {
						result.table_metadata[tbl].pks = pk;
					}
				}
			}
		}
	}

	// If the privacy unit table itself is scanned directly (no FK path), ensure its PK info is present in
	// table_metadata.
	for (auto &t : result.scanned_pu_tables) {
		if (result.table_metadata.find(t) == result.table_metadata.end()) {
			ColumnMetadata md;
			md.table_name = t;
			auto pk = FindPrimaryKey(context, t);
			md.pks = pk;
			auto fks = FindForeignKeys(context, t);
			md.fks = fks;
			result.table_metadata[t] = std::move(md);
		} else if (result.table_metadata[t].pks.empty()) {
			auto pk = FindPrimaryKey(context, t);
			if (!pk.empty()) {
				result.table_metadata[t].pks = pk;
			}
		}
	}

	// Attach discovered fk_paths to the result
	result.fk_paths = std::move(fk_paths);

	// Determine if we need to run structural checks:
	// - Either we scan PU tables directly, OR
	// - We scan tables linked to PU via FK paths
	bool has_fk_linked_tables = false;
	for (auto &kv : result.fk_paths) {
		if (!kv.second.empty()) {
			has_fk_linked_tables = true;
			break;
		}
	}

	// Structural checks BEFORE deciding eligibility (throw when invalid)
	// These checks must run for ALL queries that scan privacy unit tables OR FK-linked tables
	if (!result.scanned_pu_tables.empty() || has_fk_linked_tables) {
		// Get conservative mode setting
		bool is_conservative = GetBooleanSetting(context, "pac_conservative_mode", true);

		if (ContainsWindowFunction(*plan)) {
			if (is_conservative) {
				throw InvalidInputException("PAC rewrite: window functions are not supported for PAC compilation");
			}
			return result; // Skip PAC compilation, execute query normally
		}
		if (!ContainsAggregation(*plan)) {
			if (is_conservative) {
				throw InvalidInputException(
				    "Query does not contain any allowed aggregation (sum, count, avg, min, max)!");
			}
			return result; // Skip PAC compilation, execute query normally
		}
		if (ContainsLogicalDistinct(*plan)) {
			if (is_conservative) {
				throw InvalidInputException("PAC rewrite: DISTINCT is not supported for PAC compilation");
			}
			return result; // Skip PAC compilation, execute query normally
		}
		if (ContainsSelfJoinOfPU(*plan, result.scanned_pu_tables)) {
			if (is_conservative) {
				throw InvalidInputException("PAC rewrite: self-joins are not supported for PAC compilation");
			}
			return result; // Skip PAC compilation, execute query normally
		}
		if (ContainsDisallowedJoin(*plan)) {
			if (is_conservative) {
				throw InvalidInputException("PAC rewrite: subqueries are not supported for PAC compilation");
			}
			return result; // Skip PAC compilation, execute query normally
		}

		// Check that GROUP BY columns don't come from PU tables
		// (PU columns can only be accessed inside aggregate functions)
		if (!result.scanned_pu_tables.empty()) {
			ReplanGuard guard(optimizer_info);
			ReplanWithoutOptimizers(context, context.GetCurrentQuery(), plan);
			CheckOutputColumnsNotFromPU(*plan, *plan, result.scanned_pu_tables);
		}
	}

	// If any scanned table is linked to a privacy unit via FKs, trigger PAC compilation.
	// This should not raise errors — we accept the plan and let the rewriter handle it.
	if (has_fk_linked_tables) {
		result.eligible_for_rewrite = true;
		return result;
	}

	if (result.fk_paths.empty() && result.scanned_pu_tables.empty()) {
		// No FK paths and no scanned PAC tables: nothing to do
		return result;
	}

	// If we reach here, the plan is eligible for rewrite/compilation
	result.eligible_for_rewrite = true;
	return result;
}

} // namespace duckdb
