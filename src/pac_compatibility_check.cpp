#include "include/pac_compatibility_check.hpp"
#include "include/pac_helpers.hpp"

#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {

static bool ContainsCrossJoinWithGenerateSeries(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		auto &cross = op.Cast<LogicalJoin>();
		for (auto &child : cross.children) {
			if (child->type == LogicalOperatorType::LOGICAL_GET) {
				auto child_get = dynamic_cast<LogicalGet *>(child.get());
				if (child_get->function.name == "generate_series") {
					return true;
				}
			}
		}
	}
	for (auto &child : op.children) {
		if (ContainsCrossJoinWithGenerateSeries(*child)) {
			return true;
		}
	}
	return false;
}

static bool IsAllowedAggregate(const std::string &func) {
	static const std::unordered_set<std::string> allowed = {"sum", "sum_no_overflow", "count", "count_star", "avg"};
	std::string lower_func = func;
	std::transform(lower_func.begin(), lower_func.end(), lower_func.begin(), ::tolower);
	return allowed.count(lower_func) > 0;
}

static bool ContainsDisallowedJoin(const LogicalOperator &op) {
	// Handle different logical join operator types that derive from LogicalJoin
	if (op.type == LogicalOperatorType::LOGICAL_JOIN || op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    op.type == LogicalOperatorType::LOGICAL_DEPENDENT_JOIN || op.type == LogicalOperatorType::LOGICAL_ASOF_JOIN ||
	    op.type == LogicalOperatorType::LOGICAL_POSITIONAL_JOIN) {
		auto &join = op.Cast<LogicalJoin>();
		if (join.join_type != JoinType::INNER) {
			// Non-inner join detected: signal disallowed join via boolean return to let caller throw
			return true;
		}
	} else if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN || op.type == LogicalOperatorType::LOGICAL_ANY_JOIN ||
	           op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT || op.type == LogicalOperatorType::LOGICAL_UNION ||
	           op.type == LogicalOperatorType::LOGICAL_EXCEPT || op.type == LogicalOperatorType::LOGICAL_INTERSECT) {
		// These operator types are disallowed for PAC compilation
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

static bool ContainsDistinct(const LogicalOperator &op) {
	// If there's an explicit DISTINCT operator in the logical plan, detect it
	if (op.type == LogicalOperatorType::LOGICAL_DISTINCT) {
		return true;
	}

	// Inspect expressions attached to this operator: COUNT(DISTINCT ...), etc.
	for (auto &expr : op.expressions) {
		if (!expr) {
			continue;
		}
		if (expr->IsAggregate()) {
			auto &aggr = expr->Cast<BoundAggregateExpression>();
			if (aggr.IsDistinct()) {
				return true;
			}
		}
	}

	// Also check LogicalAggregate nodes' expressions (GROUP BY select list)
	if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = op.Cast<LogicalAggregate>();
		for (auto &expr : aggr.expressions) {
			if (!expr) {
				continue;
			}
			if (expr->IsAggregate()) {
				auto &a = expr->Cast<BoundAggregateExpression>();
				if (a.IsDistinct()) {
					return true;
				}
			}
		}
	}

	for (auto &child : op.children) {
		if (ContainsDistinct(*child)) {
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

// helper: traverse the plan and count how many times each table/CTE name is scanned
void CountScans(const LogicalOperator &op, std::unordered_map<std::string, idx_t> &counts) {
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

PACCompatibilityResult PACRewriteQueryCheck(LogicalOperator &plan, ClientContext &context,
                                            const std::vector<std::string> &pac_tables, bool replan_in_progress) {
	PACCompatibilityResult result;

	// If a replan/compilation is already in progress by the optimizer extension, skip compatibility checks
	// to avoid re-entrant behavior and infinite loops.
	if (replan_in_progress) {
		return result;
	}

	// Case 1: contains PAC tables?
	// Then, we check further conditions (aggregation, joins, etc.)
	// If there are as many sample-CTE scans as PAC table scans, then nothing to do (return empty result)
	// If there are only PAC table scans, check the other conditions and return info accordingly
	// If some conditions fail, throw an error in the caller
	// Case 2: no PAC tables? (return empty result, nothing to do)

	// count all scanned tables/CTEs in the plan
	std::unordered_map<std::string, idx_t> scan_counts;
	CountScans(plan, scan_counts);

	// Record which configured PAC tables were scanned in this plan. The optimizer
	// rule will still want to compile queries that directly read from a privacy
	// unit even when no FK paths or PKs were discovered.
	for (auto &t : pac_tables) {
		if (scan_counts[t] > 0) {
			result.scanned_pac_tables.push_back(t);
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
		std::string sample_name = std::string("_pac_internal_sample_") + t;
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

	if (all_matched && result.scanned_pac_tables.size() > 0) {
		// All PAC tables already have corresponding sample CTE scans the same number of times.
		// Nothing for the PAC rewriter to do.
		return result;
	}

	// If we reach here, there is at least one PAC table that has PAC scans without matching sample-CTE scans.
	// Now validate plan structure: ensure there is an aggregation using allowed aggregates; disallow window, distinct,
	// non-inner joins.

	// Only allow CROSS JOIN with GENERATE_SERIES (for random sample expansion)
	if (ContainsCrossJoinWithGenerateSeries(plan)) {
		return result;
	}

	// Build a vector of scanned table names to check FK links
	std::vector<std::string> scanned_tables;
	for (auto &kv : scan_counts) {
		scanned_tables.push_back(kv.first);
	}

	// Populate scanned_non_pac_tables: scanned tables that are not in the configured pac_tables
	// and are not internal sample tables (named _pac_internal_sample_<table>). This is useful
	// to know which external/non-PAC tables were read by the query.
	std::unordered_set<std::string> pac_set(pac_tables.begin(), pac_tables.end());
	for (auto &name : scanned_tables) {
		if (name.rfind("_pac_internal_sample_", 0) == 0) {
			// internal sample table, skip
			continue;
		}
		if (pac_set.find(name) == pac_set.end()) {
			result.scanned_non_pac_tables.push_back(name);
		}
	}

	// Compute FK paths from scanned tables to any privacy unit (transitive)
	auto fk_paths = FindForeignKeyBetween(context, pac_tables, scanned_tables);

	// Populate privacy unit PKs for any discovered privacy units
	for (auto &kv : fk_paths) {
		auto &path = kv.second;
		if (path.empty()) {
			continue;
		}
		std::string target = path.back();
		if (result.privacy_unit_pks.find(target) == result.privacy_unit_pks.end()) {
			auto pk = FindPrimaryKey(context, target);
			if (!pk.empty()) {
				result.privacy_unit_pks[target] = pk;
			}
		}
	}

	// If the privacy unit table itself is scanned directly (no FK path), we still want to
	// populate its primary key information. This can happen when the query scans the PU
	// table directly and there are no FK paths from other scanned tables. Populate any
	// scanned PAC table entries that do not yet have PK information.
	for (auto &t : result.scanned_pac_tables) {
		if (result.privacy_unit_pks.find(t) == result.privacy_unit_pks.end()) {
			auto pk = FindPrimaryKey(context, t);
			if (!pk.empty()) {
				result.privacy_unit_pks[t] = pk;
			}
		}
	}

	// Attach discovered fk_paths to the result
	result.fk_paths = std::move(fk_paths);

	// If any scanned table is linked to a privacy unit via FKs, trigger PAC compilation.
	// This should not raise errors — we accept the plan and let the rewriter handle it.
	for (auto &kv : result.fk_paths) {
		if (!kv.second.empty()) {
			result.eligible_for_rewrite = true;
			return result;
		}
	}

	if (result.fk_paths.empty() && result.scanned_pac_tables.empty()) {
		// No FK paths and no scanned PAC tables: nothing to do
		return result;
	}

	// Structural checks (throw when invalid)
	if (ContainsWindowFunction(plan)) {
		throw InvalidInputException("PAC rewrite: window functions are not supported for PAC compilation");
	}
	if (!ContainsAggregation(plan)) {
		throw InvalidInputException("Query does not contain any allowed aggregation (sum, count, avg)!");
	}
	if (ContainsDistinct(plan)) {
		throw InvalidInputException("PAC rewrite: DISTINCT is not supported for PAC compilation");
	}
	if (ContainsDisallowedJoin(plan)) {
		throw InvalidInputException("PAC rewrite: only INNER JOINs are supported for PAC compilation");
	}

	// If we reach here, the plan is eligible for rewrite/compilation
	result.eligible_for_rewrite = true;
	return result;
}

} // namespace duckdb
