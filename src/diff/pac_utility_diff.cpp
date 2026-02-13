//
// PAC Utility Diff - Plan-level rewrite
//
// When pac_diffcols is set, this wraps the complete PAC plan and the
// deep-copied reference plan in a FULL OUTER JOIN and builds a diff projection
// on top that encodes utility % for numeric columns and NULL patterns for
// extra/missing rows.  Both plans (including any LIMIT/ORDER BY) go under the
// join unchanged, so each side independently limits its own output.
//

#include "diff/pac_utility_diff.hpp"
#include "diff/pac_utility_summary.hpp"
#include "pac_debug.hpp"
#include "utils/pac_helpers.hpp"

#include "duckdb/planner/binder.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/common/types.hpp"

namespace duckdb {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Check whether a logical type is numeric (for utility % computation).
static bool IsNumericType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return true;
	default:
		return false;
	}
}

// Create a ROW_NUMBER() OVER () window expression for positional matching.
static unique_ptr<LogicalOperator> AddRowNumber(unique_ptr<LogicalOperator> child, idx_t table_index,
                                                ColumnBinding &rn_binding_out) {
	auto window = make_uniq<LogicalWindow>(table_index);

	auto row_number =
	    make_uniq<BoundWindowExpression>(ExpressionType::WINDOW_ROW_NUMBER, LogicalType::BIGINT, nullptr, nullptr);

	rn_binding_out = ColumnBinding(table_index, 0);

	window->expressions.push_back(std::move(row_number));
	window->children.push_back(std::move(child));
	return std::move(window);
}

// Create an IS NULL check expression.
static unique_ptr<Expression> MakeIsNull(unique_ptr<Expression> child) {
	auto op = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL, LogicalType::BOOLEAN);
	op->children.push_back(std::move(child));
	return std::move(op);
}

// Walk past LIMIT/ORDER BY/TOP_N (read-only) to find the projection and extract column names.
static vector<string> ExtractColumnNames(LogicalOperator *node) {
	while (node &&
	       (node->type == LogicalOperatorType::LOGICAL_LIMIT || node->type == LogicalOperatorType::LOGICAL_ORDER_BY ||
	        node->type == LogicalOperatorType::LOGICAL_TOP_N)) {
		if (node->children.empty()) {
			break;
		}
		node = node->children[0].get();
	}
	vector<string> names;
	if (node && node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = node->Cast<LogicalProjection>();
		for (auto &expr : proj.expressions) {
			names.push_back(expr->alias.empty() ? expr->GetName() : expr->alias);
		}
	}
	return names;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

void ApplyUtilityDiff(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                      unique_ptr<LogicalOperator> ref_plan, idx_t num_key_cols, const string &output_path) {

#if PAC_DEBUG
	PAC_DEBUG_PRINT("ApplyUtilityDiff: num_key_cols=" + std::to_string(num_key_cols));
#endif

	plan->ResolveOperatorTypes();
	ref_plan->ResolveOperatorTypes();

	// Get column bindings and types from the complete plans
	auto pac_bindings = plan->GetColumnBindings();
	auto ref_bindings = ref_plan->GetColumnBindings();
	auto col_types = plan->types; // copy — plan will be moved

	// Extract column names by reading (not modifying) the projection inside the plan
	auto col_names = ExtractColumnNames(plan.get());
	if (col_names.empty()) {
		for (idx_t i = 0; i < col_types.size(); i++) {
			col_names.push_back("col" + std::to_string(i));
		}
	}

	idx_t num_cols = col_types.size();
	if (num_key_cols >= num_cols) {
		throw InvalidInputException("pac_diffcols: num_key_cols (" + std::to_string(num_key_cols) +
		                            ") must be less than number of columns (" + std::to_string(num_cols) +
		                            "); at least one measure column is required");
	}

#if PAC_DEBUG
	PAC_DEBUG_PRINT("ApplyUtilityDiff: " + std::to_string(num_cols) + " columns:");
	for (idx_t i = 0; i < num_cols; i++) {
		PAC_DEBUG_PRINT("  [" + std::to_string(i) + "] " + col_names[i] + " : " + col_types[i].ToString() +
		                (i < num_key_cols ? " (KEY)" : (IsNumericType(col_types[i]) ? " (NUMERIC)" : "")));
	}
#endif

	auto &binder = input.optimizer.binder;
	auto &optimizer = input.optimizer;

	// ---- Step 1: Wrap each side in a pass-through projection ----
	// This ensures each side has unique table indices for the FULL OUTER JOIN,
	// so that pac-side and ref-side column bindings are unambiguous.
	ColumnBinding pac_rn_binding, ref_rn_binding;
	unique_ptr<LogicalOperator> pac_side = std::move(plan);
	unique_ptr<LogicalOperator> ref_side = std::move(ref_plan);

	auto wrap_passthrough = [&](unique_ptr<LogicalOperator> child,
	                            const vector<ColumnBinding> &bindings) -> unique_ptr<LogicalOperator> {
		vector<unique_ptr<Expression>> exprs;
		for (idx_t i = 0; i < col_types.size(); i++) {
			exprs.push_back(make_uniq<BoundColumnRefExpression>(col_types[i], bindings[i]));
		}
		auto proj = make_uniq<LogicalProjection>(binder.GenerateTableIndex(), std::move(exprs));
		proj->children.push_back(std::move(child));
		return std::move(proj);
	};

	pac_side = wrap_passthrough(std::move(pac_side), pac_bindings);
	pac_side->ResolveOperatorTypes();
	pac_bindings = pac_side->GetColumnBindings();

	ref_side = wrap_passthrough(std::move(ref_side), ref_bindings);
	ref_side->ResolveOperatorTypes();
	ref_bindings = ref_side->GetColumnBindings();

	if (num_key_cols == 0) {
		pac_side = AddRowNumber(std::move(pac_side), binder.GenerateTableIndex(), pac_rn_binding);
		ref_side = AddRowNumber(std::move(ref_side), binder.GenerateTableIndex(), ref_rn_binding);
	}

	// ---- Step 2: Create the FULL OUTER JOIN ----
	auto join = make_uniq<LogicalComparisonJoin>(JoinType::OUTER);

	if (num_key_cols == 0) {
		JoinCondition cond;
		cond.comparison = ExpressionType::COMPARE_EQUAL;
		cond.left = make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, pac_rn_binding);
		cond.right = make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, ref_rn_binding);
		join->conditions.push_back(std::move(cond));
	} else {
		for (idx_t i = 0; i < num_key_cols; i++) {
			JoinCondition cond;
			cond.comparison = ExpressionType::COMPARE_EQUAL;
			cond.left = make_uniq<BoundColumnRefExpression>(col_types[i], pac_bindings[i]);
			cond.right = make_uniq<BoundColumnRefExpression>(col_types[i], ref_bindings[i]);
			join->conditions.push_back(std::move(cond));
		}
	}

	join->children.push_back(std::move(pac_side));
	join->children.push_back(std::move(ref_side));

	// ---- Step 3: Build diff-encoded projection expressions ----
	ColumnBinding pac_probe_binding = (num_key_cols > 0) ? pac_bindings[0] : pac_rn_binding;
	ColumnBinding ref_probe_binding = (num_key_cols > 0) ? ref_bindings[0] : ref_rn_binding;
	LogicalType probe_type = (num_key_cols > 0) ? col_types[0] : LogicalType::BIGINT;

	vector<unique_ptr<Expression>> new_expressions;

	// Row semantics:
	//   "=" matched:  keys = ref values (non-null), measures = 100*pac/ref (non-null)
	//   "+" pac-only: keys = pac values (non-null), measures = NULL
	//   "-" missing:  keys = NULL,                  measures = 0

	for (idx_t i = 0; i < num_cols; i++) {
		bool is_key = (i < num_key_cols);
		bool is_numeric = IsNumericType(col_types[i]);

		if (is_key) {
			// Key columns:
			//   "-" row (pac_probe IS NULL) → NULL
			//   "+" row (ref_probe IS NULL) → pac.key
			//   "=" row                     → ref.key
			auto case_expr = make_uniq<BoundCaseExpression>(col_types[i]);
			{
				BoundCaseCheck check;
				check.when_expr = MakeIsNull(make_uniq<BoundColumnRefExpression>(probe_type, pac_probe_binding));
				check.then_expr = make_uniq<BoundConstantExpression>(Value(col_types[i]));
				case_expr->case_checks.push_back(std::move(check));
			}
			{
				BoundCaseCheck check;
				check.when_expr = MakeIsNull(make_uniq<BoundColumnRefExpression>(probe_type, ref_probe_binding));
				check.then_expr = make_uniq<BoundColumnRefExpression>(col_types[i], pac_bindings[i]);
				case_expr->case_checks.push_back(std::move(check));
			}
			case_expr->else_expr = make_uniq<BoundColumnRefExpression>(col_types[i], ref_bindings[i]);
			case_expr->alias = col_names[i];
			new_expressions.push_back(std::move(case_expr));
		} else if (is_numeric) {
			// Numeric measure columns:
			//   "+" row (ref_probe IS NULL) → NULL
			//   "-" row (pac_probe IS NULL) → 0 (missing = zero utility)
			//   "=" row, both NULL          → 0  (perfect match)
			//   "=" row, one NULL           → 50
			//   "=" row, both non-null      → 100 * |ref - pac| / max(0.00001, |ref|)
			auto case_expr = make_uniq<BoundCaseExpression>(col_types[i]);

			// Helpers: create fresh DOUBLE-cast column references
			auto make_ref_dbl = [&]() {
				return BoundCastExpression::AddDefaultCastToType(
				    make_uniq<BoundColumnRefExpression>(col_types[i], ref_bindings[i]), LogicalType::DOUBLE);
			};
			auto make_pac_dbl = [&]() {
				return BoundCastExpression::AddDefaultCastToType(
				    make_uniq<BoundColumnRefExpression>(col_types[i], pac_bindings[i]), LogicalType::DOUBLE);
			};

			// "+" row → NULL
			{
				BoundCaseCheck check;
				check.when_expr = MakeIsNull(make_uniq<BoundColumnRefExpression>(probe_type, ref_probe_binding));
				check.then_expr = make_uniq<BoundConstantExpression>(Value(col_types[i]));
				case_expr->case_checks.push_back(std::move(check));
			}

			// "-" row → 0 (missing from PAC result = zero utility)
			{
				BoundCaseCheck check;
				check.when_expr = MakeIsNull(make_uniq<BoundColumnRefExpression>(probe_type, pac_probe_binding));
				check.then_expr = BoundCastExpression::AddDefaultCastToType(
				    make_uniq<BoundConstantExpression>(Value::DOUBLE(0.0)), col_types[i]);
				case_expr->case_checks.push_back(std::move(check));
			}

			// Both NULL → 0 (perfect match)
			{
				BoundCaseCheck check;
				auto pac_null = MakeIsNull(make_uniq<BoundColumnRefExpression>(col_types[i], pac_bindings[i]));
				auto ref_null = MakeIsNull(make_uniq<BoundColumnRefExpression>(col_types[i], ref_bindings[i]));
				check.when_expr = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
				                                                        std::move(pac_null), std::move(ref_null));
				check.then_expr = BoundCastExpression::AddDefaultCastToType(
				    make_uniq<BoundConstantExpression>(Value::DOUBLE(0.0)), col_types[i]);
				case_expr->case_checks.push_back(std::move(check));
			}

			// One NULL → 50
			{
				BoundCaseCheck check;
				auto pac_null = MakeIsNull(make_uniq<BoundColumnRefExpression>(col_types[i], pac_bindings[i]));
				auto ref_null = MakeIsNull(make_uniq<BoundColumnRefExpression>(col_types[i], ref_bindings[i]));
				check.when_expr = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_OR,
				                                                        std::move(pac_null), std::move(ref_null));
				check.then_expr = BoundCastExpression::AddDefaultCastToType(
				    make_uniq<BoundConstantExpression>(Value::DOUBLE(50.0)), col_types[i]);
				case_expr->case_checks.push_back(std::move(check));
			}

			// ELSE: CAST(100 * |ref - pac| / max(0.00001, |ref|) AS original_type)
			{
				// |ref - pac|: CASE WHEN ref >= pac THEN ref-pac ELSE pac-ref END
				auto abs_diff = make_uniq<BoundCaseExpression>(LogicalType::DOUBLE);
				{
					BoundCaseCheck check;
					check.when_expr = make_uniq<BoundComparisonExpression>(ExpressionType::COMPARE_GREATERTHANOREQUALTO,
					                                                       make_ref_dbl(), make_pac_dbl());
					check.then_expr = optimizer.BindScalarFunction("-", make_ref_dbl(), make_pac_dbl());
					abs_diff->case_checks.push_back(std::move(check));
				}
				abs_diff->else_expr = optimizer.BindScalarFunction("-", make_pac_dbl(), make_ref_dbl());

				// max(0.00001, |ref|):
				// CASE WHEN ref >= 0.00001 THEN ref
				//      WHEN ref <= -0.00001 THEN 0-ref
				//      ELSE 0.00001 END
				auto denom = make_uniq<BoundCaseExpression>(LogicalType::DOUBLE);
				{
					BoundCaseCheck check;
					check.when_expr = make_uniq<BoundComparisonExpression>(
					    ExpressionType::COMPARE_GREATERTHANOREQUALTO, make_ref_dbl(),
					    make_uniq<BoundConstantExpression>(Value::DOUBLE(0.00001)));
					check.then_expr = make_ref_dbl();
					denom->case_checks.push_back(std::move(check));
				}
				{
					BoundCaseCheck check;
					check.when_expr = make_uniq<BoundComparisonExpression>(
					    ExpressionType::COMPARE_LESSTHANOREQUALTO, make_ref_dbl(),
					    make_uniq<BoundConstantExpression>(Value::DOUBLE(-0.00001)));
					check.then_expr = optimizer.BindScalarFunction(
					    "-", make_uniq<BoundConstantExpression>(Value::DOUBLE(0.0)), make_ref_dbl());
					denom->case_checks.push_back(std::move(check));
				}
				denom->else_expr = make_uniq<BoundConstantExpression>(Value::DOUBLE(0.00001));

				// 100 * abs_diff / denom
				auto hundred = make_uniq<BoundConstantExpression>(Value::DOUBLE(100.0));
				auto numerator = optimizer.BindScalarFunction("*", std::move(hundred), std::move(abs_diff));
				auto result = optimizer.BindScalarFunction("/", std::move(numerator), std::move(denom));

				case_expr->else_expr = BoundCastExpression::AddDefaultCastToType(std::move(result), col_types[i]);
			}

			case_expr->alias = col_names[i];
			new_expressions.push_back(std::move(case_expr));
		} else {
			// Non-numeric, non-key columns:
			//   "+" row → NULL
			//   "-" row → NULL
			//   "=" row → COALESCE(pac, ref)
			auto case_expr = make_uniq<BoundCaseExpression>(col_types[i]);
			{
				BoundCaseCheck check;
				check.when_expr = MakeIsNull(make_uniq<BoundColumnRefExpression>(probe_type, ref_probe_binding));
				check.then_expr = make_uniq<BoundConstantExpression>(Value(col_types[i]));
				case_expr->case_checks.push_back(std::move(check));
			}
			{
				BoundCaseCheck check;
				check.when_expr = MakeIsNull(make_uniq<BoundColumnRefExpression>(probe_type, pac_probe_binding));
				check.then_expr = make_uniq<BoundConstantExpression>(Value(col_types[i]));
				case_expr->case_checks.push_back(std::move(check));
			}
			auto coalesce = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_COALESCE, col_types[i]);
			coalesce->children.push_back(make_uniq<BoundColumnRefExpression>(col_types[i], pac_bindings[i]));
			coalesce->children.push_back(make_uniq<BoundColumnRefExpression>(col_types[i], ref_bindings[i]));
			case_expr->else_expr = std::move(coalesce);
			case_expr->alias = col_names[i];
			new_expressions.push_back(std::move(case_expr));
		}
	}

	// ---- Step 4: Create a new top-level diff projection ----
	idx_t diff_proj_idx = binder.GenerateTableIndex();
	plan = make_uniq<LogicalProjection>(diff_proj_idx, std::move(new_expressions));
	plan->children.push_back(std::move(join));

	// ---- Step 5: Wrap in utility summary operator ----
	auto summary = make_uniq<LogicalPacUtilitySummary>(num_key_cols, output_path);
	summary->children.push_back(std::move(plan));
	plan = std::move(summary);

#if PAC_DEBUG
	PAC_DEBUG_PRINT("ApplyUtilityDiff: plan rewrite complete");
#endif
}

} // namespace duckdb
