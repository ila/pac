//
// PAC Utility Summary - Streaming pass-through operator
//
// Computes recall (row matching rate) and precision (value accuracy) from the
// diff projection output.  All rows pass through unchanged; metrics are
// printed/appended at operator finalize time.
//

#include "diff/pac_utility_summary.hpp"

#include "duckdb/common/printer.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"

#include <cmath>
#include <fstream>

namespace duckdb {

// ============================================================================
// Logical operator
// ============================================================================

LogicalPacUtilitySummary::LogicalPacUtilitySummary(idx_t num_key_cols, string output_path)
    : LogicalExtensionOperator(), num_key_cols(num_key_cols), output_path(std::move(output_path)) {
}

vector<ColumnBinding> LogicalPacUtilitySummary::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

void LogicalPacUtilitySummary::ResolveTypes() {
	types = children[0]->types;
}

PhysicalOperator &LogicalPacUtilitySummary::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	// First, create the physical plan for the child
	auto &child = planner.CreatePlan(*children[0]);

	auto &op = planner.Make<PhysicalPacUtilitySummary>(types, num_key_cols, output_path, estimated_cardinality);
	op.children.push_back(child);
	return op;
}

// ============================================================================
// Physical operator - global state
// ============================================================================

struct PacUtilitySummaryGlobalState : public GlobalOperatorState {
	idx_t total_rows = 0;
	idx_t matched_rows = 0; // "=" rows: all columns non-NULL
	idx_t missing_rows = 0; // "-" rows: any key column NULL

	// Per measure column (columns after key columns): accumulate relative error %
	vector<double> utility_sum;
	vector<idx_t> utility_count;
};

// ============================================================================
// Physical operator
// ============================================================================

PhysicalPacUtilitySummary::PhysicalPacUtilitySummary(PhysicalPlan &plan, vector<LogicalType> types, idx_t num_key_cols,
                                                     string output_path, idx_t estimated_cardinality)
    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
      num_key_cols(num_key_cols), output_path(std::move(output_path)) {
}

unique_ptr<GlobalOperatorState> PhysicalPacUtilitySummary::GetGlobalOperatorState(ClientContext &context) const {
	auto state = make_uniq<PacUtilitySummaryGlobalState>();
	idx_t num_measure_cols = types.size() > num_key_cols ? types.size() - num_key_cols : 0;
	state->utility_sum.resize(num_measure_cols, 0.0);
	state->utility_count.resize(num_measure_cols, 0);
	return std::move(state);
}

OperatorResultType PhysicalPacUtilitySummary::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                      GlobalOperatorState &gstate_p, OperatorState &state) const {
	auto &gstate = gstate_p.Cast<PacUtilitySummaryGlobalState>();

	idx_t num_cols = types.size();
	idx_t count = input.size();

	// Row semantics from diff projection:
	//   "=" matched:  all columns non-NULL (keys = ref, measures = 100*|ref-pac|/max(0.00001,|ref|))
	//   "+" pac-only: keys non-NULL, some measure NULL
	//   "-" missing:  keys NULL, measures = 0

	for (idx_t row = 0; row < count; row++) {
		gstate.total_rows++;

		// Check key columns for NULLs: all-NULL means "-" row, partial NULL is an error
		idx_t null_keys = 0;
		for (idx_t k = 0; k < num_key_cols; k++) {
			if (input.data[k].GetValue(row).IsNull()) {
				null_keys++;
			}
		}
		if (null_keys > 0) {
			if (null_keys < num_key_cols) {
				throw InternalException("pac_diffcols: unexpected NULL in key column; "
				                        "key columns must not contain NULL values");
			}
			gstate.missing_rows++;
			continue;
		}
		// Check if any measure column is NULL → "+" row (pac-only, skip)
		bool any_measure_null = false;
		for (idx_t c = num_key_cols; c < num_cols; c++) {
			if (input.data[c].GetValue(row).IsNull()) {
				any_measure_null = true;
				break;
			}
		}
		if (any_measure_null) {
			continue; // "+" row — no ref to compare against
		}
		// "=" matched row — accumulate utility (cell values are relative error %)
		gstate.matched_rows++;

		for (idx_t m = num_key_cols; m < num_cols; m++) {
			auto val = input.data[m].GetValue(row);
			double dbl_val;
			try {
				dbl_val = val.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
			} catch (...) {
				continue;
			}
			idx_t idx = m - num_key_cols;
			gstate.utility_sum[idx] += dbl_val;
			gstate.utility_count[idx]++;
		}
	}
	// Pass through unchanged
	chunk.Reference(input);
	return OperatorResultType::NEED_MORE_INPUT;
}

OperatorFinalResultType PhysicalPacUtilitySummary::OperatorFinalize(Pipeline &pipeline, Event &event,
                                                                    ClientContext &context,
                                                                    OperatorFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<PacUtilitySummaryGlobalState>();

	// Compute recall: matched / (matched + missing)
	idx_t ref_rows = gstate.matched_rows + gstate.missing_rows;
	double recall = (ref_rows > 0) ? static_cast<double>(gstate.matched_rows) / static_cast<double>(ref_rows) : 1.0;

	// Compute utility: average relative error % across all reference rows
	// (matched "=" rows contribute their actual error; missing "-" rows contribute 100%)
	double utility = 0.0;
	idx_t num_measure_cols = gstate.utility_sum.size();
	idx_t cols_with_data = 0;
	for (idx_t i = 0; i < num_measure_cols; i++) {
		if (gstate.utility_count[i] > 0) {
			double col_avg = gstate.utility_sum[i] / static_cast<double>(gstate.utility_count[i]);
			utility += col_avg;
			cols_with_data++;
		}
	}
	if (cols_with_data > 0) {
		utility /= static_cast<double>(cols_with_data);
	}
	// Output results
	if (!output_path.empty()) {
		// Append to CSV file
		std::ofstream out(output_path, std::ios::app);
		if (out.is_open()) {
			out << utility << "," << recall << "\n";
			out.close();
		}
	}
	// Print to stderr
	string msg = "utility=" + std::to_string(utility) + " recall=" + std::to_string(recall);
	Printer::Print(msg);
	return OperatorFinalResultType::FINISHED;
}

} // namespace duckdb
