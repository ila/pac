#pragma once

#include "duckdb.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

// ---------------------------------------------------------------------------
// Logical operator: pass-through that marks where the physical summary goes
// ---------------------------------------------------------------------------
struct LogicalPacUtilitySummary : public LogicalExtensionOperator {
	idx_t num_key_cols;
	string output_path; // empty = stdout

	LogicalPacUtilitySummary(idx_t num_key_cols, string output_path);

	vector<ColumnBinding> GetColumnBindings() override;
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;

protected:
	void ResolveTypes() override;
};

// ---------------------------------------------------------------------------
// Physical operator: streaming pass-through that accumulates recall/precision
// ---------------------------------------------------------------------------
class PhysicalPacUtilitySummary : public PhysicalOperator {
public:
	PhysicalPacUtilitySummary(PhysicalPlan &plan, vector<LogicalType> types, idx_t num_key_cols, string output_path,
	                          idx_t estimated_cardinality);

	idx_t num_key_cols;
	string output_path;

	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorFinalResultType OperatorFinalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                                         OperatorFinalizeInput &input) const override;

	bool ParallelOperator() const override {
		return false;
	}
	bool RequiresOperatorFinalize() const override {
		return true;
	}

	string GetName() const override {
		return "PAC_UTILITY_SUMMARY";
	}
};

} // namespace duckdb
