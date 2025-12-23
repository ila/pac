//
// Created by ila on 12/23/25.
//

#include "include/pac_compiler_helpers.hpp"

#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/common/constants.hpp"

#include <vector>

namespace duckdb {

void ReplanWithoutOptimizers(ClientContext &context, const std::string &query, unique_ptr<LogicalOperator> &plan) {
	// Begin a transaction and disable a set of optimizers to simplify the generated plan
	Connection con(*context.db);
	con.BeginTransaction();
	con.Query("SET disabled_optimizers='compressed_materialization, column_lifetime, statistics_propagation, "
	          "expression_rewriter, filter_pushdown';");
	con.Commit();

	Parser parser;
	Planner planner(context);

	parser.ParseQuery(query);
	if (parser.statements.empty()) {
		return;
	}
	auto statement = parser.statements[0].get();
	planner.CreatePlan(statement->Copy());

	Optimizer optimizer(*planner.binder, context);
	plan = optimizer.Optimize(std::move(planner.plan));
	if (plan) {
		plan->Print();
	}
}

unique_ptr<LogicalOperator> *FindPrivacyUnitGetNode(unique_ptr<LogicalOperator> &plan) {
	unique_ptr<LogicalOperator> *found_ptr = nullptr;
	if (!plan) {
		return nullptr;
	}

	std::vector<unique_ptr<LogicalOperator> *> stack;
	stack.push_back(&plan);
	while (!stack.empty()) {
		auto cur_ptr = stack.back();
		stack.pop_back();
		auto &cur = *cur_ptr;
		if (!cur) {
			continue;
		}
		if (cur->type == LogicalOperatorType::LOGICAL_GET) {
			found_ptr = cur_ptr;
			break;
		}
		for (auto &c : cur->children) {
			stack.push_back(&c);
		}
	}

	if (!found_ptr) {
		throw InternalException("PAC Compiler: could not find LogicalGet node in plan");
	}

	return found_ptr;
}

} // namespace duckdb
