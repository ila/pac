//
// PAC Categorical Query Rewriter - Implementation
//
// See pac_categorical_rewriter.hpp for design documentation.
//
// Created by ila on 1/23/26.
//
#include "categorical/pac_categorical_rewriter.hpp"
#include "aggregates/pac_aggregate.hpp"
#include "pac_debug.hpp"

namespace duckdb {

static string FindPacAggregateInExpression(Expression *expr, LogicalOperator *plan_root); // forward declaration

// Trace a column binding through the plan to find if it comes from a PAC aggregate
// Returns the PAC aggregate name if found (base name without _counters), empty string otherwise
static string TracePacAggregateFromBinding(const ColumnBinding &binding, LogicalOperator *plan_root) {
	auto *source_op = FindOperatorByTableIndex(plan_root, binding.table_index);
	if (!source_op) { // source_op os the operator that produces this binding
		return "";
	}
	if (source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto *unwrapped = RecognizeDuckDBScalarWrapper(source_op);
		if (unwrapped) { // We see a DuckDB scalar subquery wrapper: skip it & search the actual scalar subquery source
			return FindPacAggregateInOperator(unwrapped);
		}
		auto &proj = source_op->Cast<LogicalProjection>();
		if (binding.column_index < proj.expressions.size()) { // Recursively search this expression for PAC aggregates
			return FindPacAggregateInExpression(proj.expressions[binding.column_index].get(), plan_root);
		}
	} else if (source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = source_op->Cast<LogicalAggregate>();
		if (binding.table_index == aggr.aggregate_index) {
			// This binding comes from an aggregate expression
			if (binding.column_index < aggr.expressions.size()) {
				auto &agg_expr = aggr.expressions[binding.column_index];
				if (agg_expr->type == ExpressionType::BOUND_AGGREGATE) {
					auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
					if (IsAnyPacAggregate(bound_agg.function.name)) {
						return GetBasePacAggregateName(bound_agg.function.name);
					}
				}
			}
		}
	}
	return "";
}

// Helper to collect ALL distinct PAC aggregate bindings in an expression
// Returns the bindings in the order they were discovered
static void CollectPacBindingsInExpression(Expression *expr, LogicalOperator *root, vector<PacBindingInfo> &bindings,
                                           unordered_map<uint64_t, idx_t> &binding_hash_to_index) {
	// Check if this is a column reference that traces back to a PAC aggregate
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		string pac_name = TracePacAggregateFromBinding(col_ref.binding, root);
		if (!pac_name.empty()) {                                  // yes!
			uint64_t binding_hash = HashBinding(col_ref.binding); // Hash the binding for uniqueness check
			if (binding_hash_to_index.find(binding_hash) == binding_hash_to_index.end()) {
				PacBindingInfo info;
				info.binding = col_ref.binding;
				info.aggregate_name = pac_name;
				info.original_type = col_ref.return_type; // Capture before counters conversion
				info.index = bindings.size();
				binding_hash_to_index[binding_hash] = info.index;
				bindings.push_back(info);
			}
		}
	}
	ExpressionIterator::EnumerateChildren(*expr, [&](Expression &child) {
		CollectPacBindingsInExpression(&child, root, bindings, binding_hash_to_index);
	});
}

// Find all PAC aggregate bindings in an expression
static vector<PacBindingInfo> FindAllPacBindingsInExpression(Expression *expr, LogicalOperator *plan_root) {
	vector<PacBindingInfo> bindings;
	unordered_map<uint64_t, idx_t> binding_hash_to_index;
	CollectPacBindingsInExpression(expr, plan_root, bindings, binding_hash_to_index);
	return bindings;
}

// Recursively search for PAC aggregate in an expression tree, with plan context for tracing column refs
// Returns the base aggregate name (without _counters suffix)
static string FindPacAggregateInExpression(Expression *expr, LogicalOperator *plan_root) {
	// examine specific expression types where the PAC aggregate could be in
	if (expr->type == ExpressionType::BOUND_AGGREGATE) { // Base case: direct PAC aggregate
		auto &agg_expr = expr->Cast<BoundAggregateExpression>();
		if (IsAnyPacAggregate(agg_expr.function.name)) {
			return GetBasePacAggregateName(agg_expr.function.name);
		}
	} else if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		return TracePacAggregateFromBinding(col_ref.binding, plan_root);
	} else if (expr->type == ExpressionType::SUBQUERY) {
		auto &subquery_expr = expr->Cast<BoundSubqueryExpression>();
		for (auto &child : subquery_expr.children) {
			string result = FindPacAggregateInExpression(child.get(), plan_root);
			if (!result.empty()) {
				return result; // Found PAC aggregate in the subquery's children (for IN, ANY, ALL operators)
			}
		}
		if (subquery_expr.subquery_type == SubqueryType::SCALAR) {
			string result = FindPacAggregateInOperator(plan_root);
			if (!result.empty()) {
				return result; // found PAC aggregate in scalar subquery
			}
		}
		return "";
	}
	// Generic traversal for all other expression types (comparisons, operators, casts,
	// functions, constants, CASE, BETWEEN, conjunctions, window functions, etc.)
	string result;
	ExpressionIterator::EnumerateChildren(*expr, [&](Expression &child) {
		if (result.empty()) {
			result = FindPacAggregateInExpression(&child, plan_root);
		}
	});
	return result;
}

// Helper to check if a filter's child is an aggregate that produces the given binding
// This detects HAVING clauses where the comparison references the immediate child aggregate
static bool IsHavingClausePattern(LogicalOperator *filter_op, const ColumnBinding &binding,
                                  LogicalOperator *plan_root) {
	if (!filter_op || filter_op->children.empty()) {
		return false;
	}
	// Check if the immediate child (or through projections) is an aggregate that produces this binding
	LogicalOperator *child = filter_op->children[0].get();
	while (child && child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		if (child->children.empty()) {
			break;
		}
		child = child->children[0].get(); // Skip projections to find the aggregate
	}
	if (child && child->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = child->Cast<LogicalAggregate>();
		if (binding.table_index == agg.aggregate_index || binding.table_index == agg.group_index) {
			return true; // this aggregate produces the binding we're comparing against
		}
	}
	return false;
}

// Resolve a column binding to its source by following through projection operators
// (including functions within projections like `0.5 * agg_result` or `pac_scale_counters(col)`)
static ColumnBinding ResolveBindingSource(const ColumnBinding &binding, LogicalOperator *plan_root) {
	auto *source_op = FindOperatorByTableIndex(plan_root, binding.table_index);
	if (!source_op || source_op->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return binding;
	}
	auto &proj = source_op->Cast<LogicalProjection>();
	if (binding.column_index < proj.expressions.size()) {
		auto &expr = proj.expressions[binding.column_index];
		if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &col_ref = expr->Cast<BoundColumnRefExpression>();
			return ResolveBindingSource(col_ref.binding, plan_root);
		}
		// For functions like 0.5 * agg_result, trace through the function's children
		if (expr->type == ExpressionType::BOUND_FUNCTION) {
			auto &func_expr = expr->Cast<BoundFunctionExpression>();
			for (auto &child : func_expr.children) {
				if (child->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = child->Cast<BoundColumnRefExpression>();
					auto traced = ResolveBindingSource(col_ref.binding, plan_root);
					// If we found a different binding, return it
					if (traced.table_index != col_ref.binding.table_index) {
						return traced;
					}
					return col_ref.binding;
				}
			}
		}
	}
	return binding;
}

// Find scalar wrapper for a binding (if any)
// Returns the outer Projection of the wrapper pattern, or nullptr
static LogicalOperator *FindScalarWrapperForBinding(const ColumnBinding &binding, LogicalOperator *plan_root) {
	auto *source_op = FindOperatorByTableIndex(plan_root, binding.table_index);
	if (!source_op || source_op->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return nullptr;
	}
	// Check if this projection is the outer part of a scalar wrapper
	return RecognizeDuckDBScalarWrapper(source_op) ? source_op : nullptr;
}

// Strip casts from an expression to find the underlying expression
static Expression *StripCasts(Expression *expr) {
	while (expr->type == ExpressionType::OPERATOR_CAST) {
		expr = expr->Cast<BoundCastExpression>().child.get();
	}
	return expr;
}

// Recursively search the plan for categorical patterns (plan-aware version)
// Now detects ANY filter expression containing a PAC aggregate, not just comparisons
void FindCategoricalPatternsInOperator(LogicalOperator *op, LogicalOperator *plan_root,
                                       vector<CategoricalPatternInfo> &patterns, bool inside_aggregate) {
	// Track if we're entering an aggregate
	bool now_inside_aggregate = inside_aggregate || (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);

	// Track patterns count before checking this operator AND its children.
	// Used at the end to strip scalar wrappers when any new patterns were found.
	size_t patterns_before_all = patterns.size();

	// Check filter expressions - detect ANY boolean expression containing a PAC aggregate
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op->Cast<LogicalFilter>();
		for (idx_t i = 0; i < filter.expressions.size(); i++) {
			auto &filter_expr = filter.expressions[i];
			// Find ALL PAC aggregate bindings in this expression (not just single)
			auto pac_bindings = FindAllPacBindingsInExpression(filter_expr.get(), plan_root);
			if (pac_bindings.empty()) {
				continue;
			}
			// Check if ANY of the bindings is NOT a HAVING clause aggregate
			// If at least one binding is from a subquery (not HAVING), this is categorical
			bool has_non_having_binding = false;
			ColumnBinding first_non_having_binding;
			string first_aggregate_name;

			for (auto &binding_info : pac_bindings) {
				ColumnBinding traced_binding = ResolveBindingSource(binding_info.binding, plan_root);
				bool is_having = IsHavingClausePattern(op, traced_binding, plan_root);
				if (!is_having) {
					has_non_having_binding = true;
					if (first_aggregate_name.empty()) {
						first_non_having_binding = binding_info.binding;
						first_aggregate_name = binding_info.aggregate_name;
					}
				}
			}
			if (has_non_having_binding) {
				CategoricalPatternInfo info;
				info.parent_op = op;
				info.expr_index = i;
				info.pac_binding = first_non_having_binding;
				info.has_pac_binding = true;
				info.aggregate_name = first_aggregate_name;
				info.pac_bindings = std::move(pac_bindings);

				// Check if this binding goes through a scalar subquery wrapper
				info.scalar_wrapper_op = FindScalarWrapperForBinding(first_non_having_binding, plan_root);

				for (auto &bi : info.pac_bindings) {
					if (bi.binding == first_non_having_binding) {
						info.original_return_type = bi.original_type;
						break; // Captured original return type from the first non-having PAC binding
					}
				}
				patterns.push_back(info);
			}
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	           op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		auto &join = op->Cast<LogicalComparisonJoin>();
		for (idx_t i = 0; i < join.conditions.size(); i++) {
			auto &cond = join.conditions[i];
			// Check if comparison involves PAC aggregate (use plan-aware version)
			CategoricalPatternInfo info;
			string left_pac = FindPacAggregateInExpression(cond.left.get(), plan_root);
			string right_pac = FindPacAggregateInExpression(cond.right.get(), plan_root);

			// NOTE: Unlike FILTER expressions, COMPARISON_JOIN conditions cannot be HAVING clauses.
			// HAVING filters are always FILTER operators, not join conditions.
			// So we don't need the now_inside_aggregate check here - any PAC aggregate
			// in a join condition is a categorical pattern (correlated subquery).
			if (!left_pac.empty()) {
				info.parent_op = op;
				info.expr_index = i;
				info.aggregate_name = left_pac;
				if (cond.left->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = cond.left->Cast<BoundColumnRefExpression>();
					info.scalar_wrapper_op = FindScalarWrapperForBinding(col_ref.binding, plan_root);
				}
				patterns.push_back(info);
			} else if (!right_pac.empty()) {
				info.parent_op = op;
				info.expr_index = i;
				info.aggregate_name = right_pac;
				if (cond.right->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = cond.right->Cast<BoundColumnRefExpression>();
					info.scalar_wrapper_op = FindScalarWrapperForBinding(col_ref.binding, plan_root);
				}
				patterns.push_back(info);
			}
		}
	} else if (!inside_aggregate && patterns.empty() && op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		// Check projection expressions for arithmetic involving multiple PAC aggregates
		// This handles cases like Q08: sum(CASE...)/sum(volume) in SELECT list
		// NOTE: Only check projections if we haven't already found filter/join patterns,
		// because those patterns will handle the projections via RewriteProjectionsWithCounters.
		// We only want standalone projection patterns (no filter/join categorical patterns).
		auto &proj = op->Cast<LogicalProjection>();
		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			auto &expr = proj.expressions[i];
			if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
				continue; // Skip simple column references - they don't need rewriting
			}
			// Find ALL PAC aggregate bindings in this expression
			auto pac_bindings = FindAllPacBindingsInExpression(expr.get(), plan_root);
			if (pac_bindings.empty()) {
				continue;
			}
			if (!IsNumericalType(expr->return_type)) {
				continue; // Only create pattern if result is numerical (pac_noised only works on numbers)
			}
			CategoricalPatternInfo info;
			info.parent_op = op;
			info.expr_index = i;
			info.pac_binding = pac_bindings[0].binding;
			info.has_pac_binding = true;
			info.aggregate_name = pac_bindings[0].aggregate_name;
			info.original_return_type = expr->return_type;
			info.pac_bindings = std::move(pac_bindings);
			patterns.push_back(std::move(info));
		}
	}
	for (auto &child : op->children) {
		FindCategoricalPatternsInOperator(child.get(), plan_root, patterns, now_inside_aggregate);
	}
	// On the way back up: if patterns were found in this subtree, strip scalar wrappers in direct children.
	if (patterns.size() > patterns_before_all) {
		for (auto &child : op->children) {
			if (child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto *unwrapped = RecognizeDuckDBScalarWrapper(child.get());
				if (unwrapped) {
					StripScalarWrapperInPlace(child, true);
				}
			}
		}
	}
}

// Check if an expression traces back to a PAC _counters aggregate
static bool TracesPacCountersAggregate(Expression *expr, LogicalOperator *plan_root) {
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		string pac_name = TracePacAggregateFromBinding(col_ref.binding, plan_root);
		return !pac_name.empty(); // TracePacAggregateFromBinding uses IsAnyPacAggregate, so it will find _counters too
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(*expr, [&](Expression &child) {
		if (TracesPacCountersAggregate(&child, plan_root)) {
			found = true;
		}
	});
	return found;
}

// Replace standard aggregates that operate on PAC counter results
// Returns the new BoundAggregateExpression, or nullptr if binding fails.
static unique_ptr<Expression> RebindAggregate(ClientContext &context, const string &func_name,
                                              vector<unique_ptr<Expression>> children, bool is_distinct) {
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto &func_entry = catalog.GetEntry<AggregateFunctionCatalogEntry>(context, DEFAULT_SCHEMA, func_name);
	vector<LogicalType> arg_types;
	for (auto &child : children) {
		arg_types.push_back(child->return_type);
	}
	ErrorData error;
	FunctionBinder function_binder(context);
	auto best_function = function_binder.BindFunction(func_name, func_entry.functions, arg_types, error);
	if (!best_function.IsValid()) {
		return nullptr;
	}
	AggregateFunction func = func_entry.functions.GetFunctionByOffset(best_function.GetIndex());
	return function_binder.BindAggregateFunction(func, std::move(children), nullptr,
	                                             is_distinct ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT);
}

// with pac_*_list variants that aggregate element-wise
static void ReplaceAggregatesOverCounters(LogicalOperator *op, ClientContext &context, LogicalOperator *plan_root) {
	if (op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return;
	}
	auto &agg = op->Cast<LogicalAggregate>();
	for (idx_t i = 0; i < agg.expressions.size(); i++) {
		auto &agg_expr = agg.expressions[i];
		if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
			continue;
		}
		auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
		if (bound_agg.children.empty()) {
			continue;
		}
		// Check if input traces to a PAC _counters aggregate
		bool traces_counters = TracesPacCountersAggregate(bound_agg.children[0].get(), plan_root);
		if (!traces_counters) {
			continue;
		}
		string list_variant = GetListAggregateVariant(bound_agg.function.name);
		if (list_variant.empty()) {
			continue;
		}
		// Rebind to the list variant
		vector<unique_ptr<Expression>> children;
		for (auto &child : bound_agg.children) {
			auto child_copy = child->Copy();
			if (child_copy->type == ExpressionType::BOUND_COLUMN_REF) {
				child_copy->return_type = LogicalType::LIST(PacFloatLogicalType());
			}
			children.push_back(std::move(child_copy));
		}
		auto new_aggr = RebindAggregate(context, list_variant, std::move(children), bound_agg.IsDistinct());
		if (new_aggr) {
			agg.expressions[i] = std::move(new_aggr);
			idx_t types_index = agg.groups.size() + i;
			if (types_index < agg.types.size()) {
				agg.types[types_index] = LogicalType::LIST(PacFloatLogicalType());
			}
		}
	}
}

// Build an operator expression from cloned children, handling COALESCE type coercion
// For COALESCE, all children must have compatible types. If the first child changed to DOUBLE
// (typical when PAC binding becomes a lambda element), cast other children to match.
static unique_ptr<Expression> BuildClonedOperatorExpression(Expression *original_expr,
                                                            vector<unique_ptr<Expression>> new_children) {
	auto &op = original_expr->Cast<BoundOperatorExpression>();
	LogicalType result_type = op.return_type;

	// For COALESCE with mismatched child types, cast all to the first child's type
	if (original_expr->type == ExpressionType::OPERATOR_COALESCE && new_children.size() > 1) {
		LogicalType first_type = new_children[0]->return_type;
		bool types_mismatch = false;
		for (idx_t i = 1; i < new_children.size(); i++) {
			if (new_children[i]->return_type != first_type) {
				types_mismatch = true;
				break;
			}
		}
		if (types_mismatch) {
			for (auto &child : new_children) {
				if (child->return_type != first_type) {
					child = BoundCastExpression::AddDefaultCastToType(std::move(child), first_type);
				}
			}
			result_type = first_type;
		}
	}
	auto result = make_uniq<BoundOperatorExpression>(original_expr->type, result_type);
	for (auto &child : new_children) {
		result->children.push_back(std::move(child));
	}
	return result;
}

// Capture a non-PAC column reference for use in a lambda
// Returns the BoundReferenceExpression index (1 + capture_idx since index 0 is the element)
static unique_ptr<Expression> CaptureColumnRef(const BoundColumnRefExpression &col_ref,
                                               vector<unique_ptr<Expression>> &captures,
                                               unordered_map<uint64_t, idx_t> &capture_map) {
	uint64_t hash = HashBinding(col_ref.binding);
	idx_t capture_idx;
	auto it = capture_map.find(hash);
	if (it != capture_map.end()) {
		capture_idx = it->second;
	} else {
		capture_idx = captures.size();
		capture_map[hash] = capture_idx;
		captures.push_back(col_ref.Copy());
	}
	return make_uniq<BoundReferenceExpression>(col_ref.alias, col_ref.return_type, 1 + capture_idx);
}

// Build a struct_extract_at expression to extract a field from a struct element
// field_idx is 0-based internally, but struct_extract_at needs 1-based argument
static unique_ptr<Expression> BuildStructFieldExtract(const LogicalType &struct_type, idx_t field_idx,
                                                      const string &field_name) {
	auto elem_ref = make_uniq<BoundReferenceExpression>("elem", struct_type, idx_t(0));
	auto child_types = StructType::GetChildTypes(struct_type);
	LogicalType extract_return_type = PacFloatLogicalType();
	for (idx_t j = 0; j < child_types.size(); j++) {
		if (child_types[j].first == field_name) {
			extract_return_type = child_types[j].second; // Get the field type from the struct
			break;
		}
	}
	auto extract_func = StructExtractAtFun::GetFunction();
	auto bind_data = StructExtractAtFun::GetBindData(field_idx);
	vector<unique_ptr<Expression>> extract_children;
	extract_children.push_back(std::move(elem_ref));
	extract_children.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(static_cast<int64_t>(field_idx + 1))));
	return make_uniq<BoundFunctionExpression>(extract_return_type, extract_func, std::move(extract_children),
	                                          std::move(bind_data));
}

// Clone an expression tree for use as a lambda body (unified single/multi binding version).
// PAC aggregate column refs are replaced with lambda element references.
// Other column refs are captured and become BoundReferenceExpression(1+i).
//
// pac_binding_map: maps binding hash -> struct field index (single: one entry mapping to 0)
// struct_type: nullptr = single binding (elem ref), non-null = multi binding (struct field extract)
static unique_ptr<Expression> CloneForLambdaBody(Expression *expr,
                                                 const unordered_map<uint64_t, idx_t> &pac_binding_map,
                                                 vector<unique_ptr<Expression>> &captures,
                                                 unordered_map<uint64_t, idx_t> &capture_map,
                                                 LogicalOperator *plan_root, const LogicalType *struct_type) {
	// Helper to build the replacement expression for a matched PAC binding
	auto make_pac_replacement = [&](const unordered_map<uint64_t, idx_t>::const_iterator &it,
	                                const string &alias) -> unique_ptr<Expression> {
		if (struct_type) {
			return BuildStructFieldExtract(*struct_type, it->second, GetStructFieldName(it->second));
		} else {
			return make_uniq<BoundReferenceExpression>(alias, PacFloatLogicalType(), idx_t(0));
		}
	};
	// rewrite all kinds of expressions:
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		uint64_t binding_hash = HashBinding(col_ref.binding);
		// Direct match
		auto it = pac_binding_map.find(binding_hash);
		if (it != pac_binding_map.end()) {
			return make_pac_replacement(it, "elem");
		}
		// Trace through projections (and functions like pac_scale_counters) to find PAC binding
		ColumnBinding traced = ResolveBindingSource(col_ref.binding, plan_root);
		if (!(traced == col_ref.binding)) {
			auto traced_it = pac_binding_map.find(HashBinding(traced));
			if (traced_it != pac_binding_map.end()) {
				return make_pac_replacement(traced_it, col_ref.alias);
			}
		}
		return CaptureColumnRef(col_ref, captures, capture_map); // Other column ref - needs to be captured
	} else if (expr->type == ExpressionType::VALUE_CONSTANT) {
		return expr->Copy();
	} else if (expr->type == ExpressionType::OPERATOR_CAST) {
		auto &cast = expr->Cast<BoundCastExpression>();
		auto child_clone =
		    CloneForLambdaBody(cast.child.get(), pac_binding_map, captures, capture_map, plan_root, struct_type);
		if (child_clone->return_type == cast.return_type) {
			return child_clone; // If the child's type already matches the target type, skip the cast
		}
		// Otherwise, create a new cast with the correct function for the new child type
		return BoundCastExpression::AddDefaultCastToType(std::move(child_clone), cast.return_type);
	} else if (expr->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
		auto &comp = expr->Cast<BoundComparisonExpression>();
		auto left_clone =
		    CloneForLambdaBody(comp.left.get(), pac_binding_map, captures, capture_map, plan_root, struct_type);
		auto right_clone =
		    CloneForLambdaBody(comp.right.get(), pac_binding_map, captures, capture_map, plan_root, struct_type);
		// Reconcile types if they differ (needed for multi where struct fields are DOUBLE
		// but original CASTs may introduce DECIMAL)
		if (left_clone->return_type != right_clone->return_type) {
			if (left_clone->return_type != PacFloatLogicalType()) {
				left_clone = BoundCastExpression::AddDefaultCastToType(std::move(left_clone), PacFloatLogicalType());
			}
			if (right_clone->return_type != PacFloatLogicalType()) {
				right_clone = BoundCastExpression::AddDefaultCastToType(std::move(right_clone), PacFloatLogicalType());
			}
		}
		return make_uniq<BoundComparisonExpression>(expr->type, std::move(left_clone), std::move(right_clone));
	} else if (expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = expr->Cast<BoundFunctionExpression>();
		vector<unique_ptr<Expression>> new_children;
		bool any_cast_needed = false;
		for (idx_t i = 0; i < func.children.size(); i++) {
			auto child_clone = CloneForLambdaBody(func.children[i].get(), pac_binding_map, captures, capture_map,
			                                      plan_root, struct_type);
			// If a child's type changed (e.g., DECIMAL->DOUBLE from PAC counter conversion),
			// cast it to the type the bound function expects, so the function binding stays valid.
			if (i < func.function.arguments.size() && child_clone->return_type != func.function.arguments[i]) {
				child_clone =
				    BoundCastExpression::AddDefaultCastToType(std::move(child_clone), func.function.arguments[i]);
				any_cast_needed = true;
			}
			new_children.push_back(std::move(child_clone));
		}
		unique_ptr<Expression> result =
		    make_uniq<BoundFunctionExpression>(func.return_type, func.function, std::move(new_children),
		                                       func.bind_info ? func.bind_info->Copy() : nullptr);
		// Children were cast from PAC_FLOAT to the function's bound types (e.g., DECIMAL).
		// Cast the result back to PAC_FLOAT so the list_transform output stays LIST<PAC_FLOAT>.
		if (any_cast_needed && result->return_type != PacFloatLogicalType()) {
			result = BoundCastExpression::AddDefaultCastToType(std::move(result), PacFloatLogicalType());
		}
		return result;
	} else if (expr->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) { // AND, OR, NOT, arithmetic, COALESCE..
		auto &op = expr->Cast<BoundOperatorExpression>();
		vector<unique_ptr<Expression>> new_children;
		for (auto &child : op.children) {
			new_children.push_back(
			    CloneForLambdaBody(child.get(), pac_binding_map, captures, capture_map, plan_root, struct_type));
		}
		return BuildClonedOperatorExpression(expr, std::move(new_children));
	} else if (expr->type == ExpressionType::CASE_EXPR) {
		auto &case_expr = expr->Cast<BoundCaseExpression>();
		if (IsScalarSubqueryWrapper(case_expr)) {
			return CloneForLambdaBody(case_expr.else_expr.get(), pac_binding_map, captures, capture_map, plan_root,
			                          struct_type);
		}
		// Regular CASE - recurse into all branches
		// Start with original return type, will update based on cloned branches
		auto result = make_uniq<BoundCaseExpression>(case_expr.return_type);
		for (auto &check : case_expr.case_checks) {
			BoundCaseCheck new_check;
			new_check.when_expr = CloneForLambdaBody(check.when_expr.get(), pac_binding_map, captures, capture_map,
			                                         plan_root, struct_type);
			new_check.then_expr = CloneForLambdaBody(check.then_expr.get(), pac_binding_map, captures, capture_map,
			                                         plan_root, struct_type);
			result->case_checks.push_back(std::move(new_check));
		}
		if (case_expr.else_expr) {
			result->else_expr = CloneForLambdaBody(case_expr.else_expr.get(), pac_binding_map, captures, capture_map,
			                                       plan_root, struct_type);
			// Update return type to match ELSE branch (the PAC element type)
			result->return_type = result->else_expr->return_type;
		}
		// Cast THEN branches to match the return type if needed
		for (auto &check : result->case_checks) {
			if (check.then_expr && check.then_expr->return_type != result->return_type) {
				check.then_expr =
				    BoundCastExpression::AddDefaultCastToType(std::move(check.then_expr), result->return_type);
			}
		}
		return result;
	}
	return expr->Copy();
}

// Build a BoundLambdaExpression from a lambda body and captures
static unique_ptr<Expression> BuildPacLambda(unique_ptr<Expression> lambda_body,
                                             vector<unique_ptr<Expression>> captures) {
	auto lambda =
	    make_uniq<BoundLambdaExpression>(ExpressionType::LAMBDA, LogicalType::LAMBDA, std::move(lambda_body), idx_t(1));
	lambda->captures = std::move(captures);
	return lambda;
}

// Build a list_transform function call with proper binding
// element_return_type: the type each element maps to (e.g., BOOLEAN for predicates, some other type for casts)
static unique_ptr<Expression> BuildListTransformCall(OptimizerExtensionInput &input,
                                                     unique_ptr<Expression> counters_list,
                                                     unique_ptr<Expression> lambda_expr,
                                                     const LogicalType &element_return_type = LogicalType::BOOLEAN) {
	// Get the lambda body for ListLambdaBindData
	auto &bound_lambda = lambda_expr->Cast<BoundLambdaExpression>();

	// Get list_transform function from catalog
	auto &catalog = Catalog::GetSystemCatalog(input.context);
	auto &func_entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, "list_transform");

	// Get the first function overload (list_transform has only one signature pattern)
	auto &scalar_func = func_entry.functions.functions[0];

	// Create the ListLambdaBindData with the lambda body
	auto list_return_type = LogicalType::LIST(element_return_type);
	auto bind_data = make_uniq<ListLambdaBindData>(list_return_type, std::move(bound_lambda.lambda_expr), false, false);

	// Build children: [list, captures...] Note: The lambda itself is NOT a child after binding - only its captures are
	vector<unique_ptr<Expression>> children;
	children.push_back(std::move(counters_list));

	for (auto &capture : bound_lambda.captures) {
		children.push_back(std::move(capture)); // Add captures as children
	}
	// Create the bound function expression
	return make_uniq<BoundFunctionExpression>(list_return_type, scalar_func, std::move(children), std::move(bind_data));
}

// Build a list_zip function call combining multiple counter lists
// Returns LIST<STRUCT<a T1, b T2, ...>> where each field corresponds to one PAC binding
static unique_ptr<Expression> BuildListZipCall(OptimizerExtensionInput &input,
                                               vector<unique_ptr<Expression>> counter_lists,
                                               LogicalType &out_struct_type) {
	// Build the struct type for list_zip result -- list_zip returns LIST<STRUCT<a T1, b T2, ...>>
	child_list_t<LogicalType> struct_children;
	for (idx_t i = 0; i < counter_lists.size(); i++) {
		string field_name = GetStructFieldName(i);
		struct_children.push_back(
		    make_pair(field_name, PacFloatLogicalType())); // All counter lists use PAC_FLOAT element type
	}
	out_struct_type = LogicalType::STRUCT(struct_children);
	auto list_struct_type = LogicalType::LIST(out_struct_type);

	// Get list_zip function from catalog
	auto &catalog = Catalog::GetSystemCatalog(input.context);
	auto &func_entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, "list_zip");

	// Find the appropriate overload (list_zip is variadic)
	vector<LogicalType> arg_types;
	for (auto &list : counter_lists) {
		arg_types.push_back(list->return_type);
	}
	ErrorData error;
	FunctionBinder function_binder(input.context);
	auto best_function = function_binder.BindFunction(func_entry.name, func_entry.functions, arg_types, error);
	if (best_function.IsValid()) {
		auto scalar_func = func_entry.functions.GetFunctionByOffset(best_function.GetIndex());
		return make_uniq<BoundFunctionExpression>(list_struct_type, scalar_func, std::move(counter_lists), nullptr);
	}
	return nullptr;
}

// Helper to check if an expression contains a column ref with a specific table_index
static bool ExpressionContainsColumnRefToTable(Expression *expr, idx_t table_index) {
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		if (col_ref.binding.table_index == table_index) {
			return true;
		}
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		if (ExpressionContainsColumnRefToTable(child.get(), table_index)) {
			found = true;
		}
	});
	return found;
}

// Check if this projection's output is referenced by a categorical filter pattern
// If so, we should NOT wrap it with pac_noised - the filter will handle the rewrite
static bool IsProjectionReferencedByFilterPattern(LogicalProjection &proj,
                                                  const vector<CategoricalPatternInfo> &patterns,
                                                  LogicalOperator *plan_root) {
	// For each pattern, check if the filter expression references this projection's output
	for (auto &pattern : patterns) {
		if (!pattern.parent_op) {
			continue;
		}
		if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = pattern.parent_op->Cast<LogicalFilter>();
			if (pattern.expr_index >= filter.expressions.size()) {
				continue;
			}
			auto &filter_expr = filter.expressions[pattern.expr_index];
			if (ExpressionContainsColumnRefToTable(filter_expr.get(), proj.table_index)) {
				return true; // Check for direct column refs to this projection
			}
			auto pac_bindings = FindAllPacBindingsInExpression(filter_expr.get(), plan_root);
			for (auto &binding_info : pac_bindings) {
				if (binding_info.binding.table_index == proj.table_index) {
					return true; // Also check traced PAC bindings
				}
			}
		} else if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		           pattern.parent_op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &join = pattern.parent_op->Cast<LogicalComparisonJoin>();
			if (pattern.expr_index >= join.conditions.size()) {
				continue;
			}
			// Check for direct column refs to this projection on either side
			auto &cond = join.conditions[pattern.expr_index];
			bool left_has = ExpressionContainsColumnRefToTable(cond.left.get(), proj.table_index);
			bool right_has = ExpressionContainsColumnRefToTable(cond.right.get(), proj.table_index);
			if (left_has || right_has) {
				return true;
			}
			// Also check traced PAC bindings
			auto left_bindings = FindAllPacBindingsInExpression(cond.left.get(), plan_root);
			auto right_bindings = FindAllPacBindingsInExpression(cond.right.get(), plan_root);
			for (auto &binding_info : left_bindings) {
				if (binding_info.binding.table_index == proj.table_index) {
					return true;
				}
			}
			for (auto &binding_info : right_bindings) {
				if (binding_info.binding.table_index == proj.table_index) {
					return true;
				}
			}
		}
	}
	return false;
}

// Build a list_transform expression over PAC counter bindings.
// Check whether an expression tree contains any null-handling operators
// (COALESCE, IS NULL, IS NOT NULL). When absent, pac_coalesce is unnecessary
// because NULL propagates identically through arithmetic.
static bool ExpressionContainsNullHandling(Expression *expr) {
	if (expr->type == ExpressionType::OPERATOR_COALESCE || expr->type == ExpressionType::OPERATOR_IS_NULL ||
	    expr->type == ExpressionType::OPERATOR_IS_NOT_NULL) {
		return true;
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(*expr, [&](Expression &child) {
		if (!found) {
			found = ExpressionContainsNullHandling(&child);
		}
	});
	return found;
}

// For single binding: list_transform(counters, elem -> body(elem))
// For multiple bindings: list_transform(list_zip(c1, c2, ...), elem -> body(elem.a, elem.b, ...))
//   CloneForLambdaBody (multi mode) handles per-field type casting internally.
//
// Returns the list_transform expression, or nullptr on failure.
// The caller wraps with pac_noised as needed.
static unique_ptr<Expression> BuildCounterListTransform(OptimizerExtensionInput &input,
                                                        const vector<PacBindingInfo> &pac_bindings,
                                                        Expression *expr_to_transform, LogicalOperator *plan_root,
                                                        const LogicalType &result_element_type) {
	if (pac_bindings.size() == 1) { // --- SINGLE AGGREGATE ---
		auto &binding_info = pac_bindings[0];
		ColumnBinding pac_binding = binding_info.binding;
		auto counters_ref =
		    make_uniq<BoundColumnRefExpression>("pac_var", LogicalType::LIST(PacFloatLogicalType()), pac_binding);
		unique_ptr<Expression> input_list;
		if (ExpressionContainsNullHandling(expr_to_transform)) {
			input_list = input.optimizer.BindScalarFunction("pac_coalesce", std::move(counters_ref));
		} else {
			input_list = std::move(counters_ref);
		}
		// Outer lambda: clone the expression replacing PAC binding with lambda element
		unordered_map<uint64_t, idx_t> pac_binding_map;
		pac_binding_map[HashBinding(pac_binding)] = 0;
		vector<unique_ptr<Expression>> captures;
		unordered_map<uint64_t, idx_t> capture_map;
		auto lambda_body =
		    CloneForLambdaBody(expr_to_transform, pac_binding_map, captures, capture_map, plan_root, nullptr);
		// Ensure body returns result_element_type if needed
		if (result_element_type == PacFloatLogicalType() && lambda_body->return_type != PacFloatLogicalType()) {
			lambda_body = BoundCastExpression::AddDefaultCastToType(std::move(lambda_body), PacFloatLogicalType());
		}
		auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));
		return BuildListTransformCall(input, std::move(input_list), std::move(lambda), result_element_type);
	} else { // --- MULTIPLE AGGREGATES ---
		vector<unique_ptr<Expression>> counter_lists;
		unordered_map<uint64_t, idx_t> binding_to_index;
		bool needs_coalesce = ExpressionContainsNullHandling(expr_to_transform);
		for (auto &bi : pac_bindings) {
			auto ref =
			    make_uniq<BoundColumnRefExpression>("pac_var", LogicalType::LIST(PacFloatLogicalType()), bi.binding);
			if (needs_coalesce) {
				counter_lists.push_back(input.optimizer.BindScalarFunction("pac_coalesce", std::move(ref)));
			} else {
				counter_lists.push_back(std::move(ref));
			}
			binding_to_index[HashBinding(bi.binding)] = bi.index;
		}
		LogicalType struct_type;
		auto zipped_list = BuildListZipCall(input, std::move(counter_lists), struct_type);
		if (zipped_list) {
			vector<unique_ptr<Expression>> captures;
			unordered_map<uint64_t, idx_t> map;
			auto lambda_body =
			    CloneForLambdaBody(expr_to_transform, binding_to_index, captures, map, plan_root, &struct_type);
			if (result_element_type == PacFloatLogicalType() && lambda_body->return_type != PacFloatLogicalType()) {
				lambda_body = BoundCastExpression::AddDefaultCastToType(std::move(lambda_body), PacFloatLogicalType());
			}
			auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));
			return BuildListTransformCall(input, std::move(zipped_list), std::move(lambda), result_element_type);
		}
	}
	return nullptr;
}

// Check which child of a binary function contains a PAC binding
// Returns 0 if first child has PAC, 1 if second child has PAC, -1 if neither or both
static int FindPacChildIndex(Expression *expr, LogicalOperator *plan_root) {
	if (expr->type != ExpressionType::BOUND_FUNCTION) {
		return -1;
	}
	auto &func = expr->Cast<BoundFunctionExpression>();
	if (func.children.size() != 2) {
		return -1;
	}
	auto left_bindings = FindAllPacBindingsInExpression(func.children[0].get(), plan_root);
	auto right_bindings = FindAllPacBindingsInExpression(func.children[1].get(), plan_root);
	bool left_has = !left_bindings.empty();
	bool right_has = !right_bindings.empty();
	if (left_has && !right_has) {
		return 0;
	}
	if (!left_has && right_has) {
		return 1;
	}
	return -1; // both or neither
}

/// Try to algebraically simplify a filter comparison involving a PAC aggregate.
// Moves arithmetic from the PAC (list) side to the scalar side so the PAC side
// becomes a bare column ref. Always returns a BoundComparisonExpression on success.
// Returns nullptr only if the expression is not a comparison or doesn't have
// exactly one PAC binding (i.e., no work was possible).
static unique_ptr<Expression> TryRewriteFilterComparison(OptimizerExtensionInput &input,
                                                         const vector<PacBindingInfo> &pac_bindings, Expression *expr,
                                                         LogicalOperator *plan_root) {
	if (pac_bindings.size() != 1 || expr->GetExpressionClass() != ExpressionClass::BOUND_COMPARISON) {
		return nullptr; // Only works with a comparison against a single PAC binding
	}
	auto &comp = expr->Cast<BoundComparisonExpression>();
	ExpressionType cmp_type = comp.type;

	// Determine which side has the PAC binding
	auto left_bindings = FindAllPacBindingsInExpression(comp.left.get(), plan_root);
	auto right_bindings = FindAllPacBindingsInExpression(comp.right.get(), plan_root);
	bool left_has_pac = !left_bindings.empty();
	bool right_has_pac = !right_bindings.empty();

	if (left_has_pac == right_has_pac) {
		return nullptr; // both sides or neither — can't optimize
	}
	// Normalize: scalar_side CMP list_side (PAC on right)
	unique_ptr<Expression> scalar_side;
	unique_ptr<Expression> list_side;
	if (left_has_pac) { // PAC on left: flip comparison
		scalar_side = comp.right->Copy();
		list_side = comp.left->Copy();
		cmp_type = FlipComparison(cmp_type);
	} else {
		scalar_side = comp.left->Copy();
		list_side = comp.right->Copy();
	}
	// Iteratively simplify: move arithmetic from list_side to scalar_side
	while (list_side->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = list_side->Cast<BoundFunctionExpression>();
		if (func.children.size() != 2) {
			break;
		}
		int pac_child = FindPacChildIndex(list_side.get(), plan_root);
		if (pac_child < 0) {
			break;
		}
		bool needs_positive_check = false;
		const char *inverse_op = GetInverseArithmeticOp(func.function.name, pac_child, needs_positive_check);
		if (!inverse_op) {
			break;
		}
		auto &scalar_operand = func.children[1 - pac_child];
		if (needs_positive_check && !IsPositiveConstant(scalar_operand.get())) {
			break;
		}
		// When the inverse is division, multiply by the reciprocal instead (cheaper at runtime).
		// We know scalar_operand is a positive constant, so we can compute 1/value at plan time.
		double const_val;
		if (inverse_op[0] == '/' && TryGetConstantDouble(scalar_operand.get(), const_val)) {
			auto reciprocal = make_uniq<BoundConstantExpression>(Value::DOUBLE(1.0 / const_val));
			scalar_side = input.optimizer.BindScalarFunction("*", std::move(scalar_side), std::move(reciprocal));
		} else {
			scalar_side =
			    input.optimizer.BindScalarFunction(inverse_op, std::move(scalar_side), scalar_operand->Copy());
		}
		list_side = func.children[pac_child]->Copy();
	}
	// Return the (possibly simplified) comparison
	return make_uniq<BoundComparisonExpression>(cmp_type, std::move(scalar_side), std::move(list_side));
}

// Forward declarations for filter/join rewriting
static unique_ptr<Expression> NoisePacNumericExpr(OptimizerExtensionInput &input,
                                                  const vector<PacBindingInfo> &pac_bindings, Expression *expr,
                                                  LogicalOperator *plan_root);
static unique_ptr<Expression> RewriteExpressionWithCounters(OptimizerExtensionInput &input,
                                                            const vector<PacBindingInfo> &pac_bindings,
                                                            Expression *expr, LogicalOperator *plan_root,
                                                            const LogicalType &target_type = PacFloatLogicalType());

// Rewrite a filter expression tree to noise PAC sides, preserving normal comparison operators.
// For comparisons: algebraically simplify, then noise the PAC side with pac_noised.
// For numeric sub-expressions: noise directly via NoisePacNumericExpr.
// For non-numeric (AND/OR/NOT): recurse into children.
static unique_ptr<Expression> RewriteFilterWithNoised(OptimizerExtensionInput &input,
                                                      const vector<PacBindingInfo> &pac_bindings, Expression *expr,
                                                      LogicalOperator *plan_root) {
	// Comparison: algebraic simplification then noise PAC side(s)
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
		auto simplified = TryRewriteFilterComparison(input, pac_bindings, expr, plan_root);
		if (!simplified) {
			simplified = expr->Copy();
		}
		auto &comp = simplified->Cast<BoundComparisonExpression>();
		// Noise each side that has PAC bindings independently
		auto left_bindings = FindAllPacBindingsInExpression(comp.left.get(), plan_root);
		auto right_bindings = FindAllPacBindingsInExpression(comp.right.get(), plan_root);
		if (!left_bindings.empty()) {
			auto noised = NoisePacNumericExpr(input, left_bindings, comp.left.get(), plan_root);
			if (noised) {
				comp.left = std::move(noised);
			}
		}
		if (!right_bindings.empty()) {
			auto noised = NoisePacNumericExpr(input, right_bindings, comp.right.get(), plan_root);
			if (noised) {
				comp.right = std::move(noised);
			}
		}
		// Reconcile types: after noising, PAC sides are DOUBLE but other side may be DECIMAL/INTEGER.
		// Cast both to DOUBLE for a valid comparison.
		if (comp.left->return_type != comp.right->return_type) {
			if (comp.left->return_type != PacFloatLogicalType()) {
				comp.left = BoundCastExpression::AddDefaultCastToType(std::move(comp.left), PacFloatLogicalType());
			}
			if (comp.right->return_type != PacFloatLogicalType()) {
				comp.right = BoundCastExpression::AddDefaultCastToType(std::move(comp.right), PacFloatLogicalType());
			}
		}
		return simplified;
	}
	// Numeric: noise directly
	if (IsNumericalType(expr->return_type)) {
		return NoisePacNumericExpr(input, pac_bindings, expr, plan_root);
	}
	// Non-numeric (AND/OR/etc): recurse into children
	auto clone = expr->Copy();
	bool changed = false;
	ExpressionIterator::EnumerateChildren(*clone, [&](unique_ptr<Expression> &child) {
		auto cb = FindAllPacBindingsInExpression(child.get(), plan_root);
		if (!cb.empty()) {
			auto r = RewriteFilterWithNoised(input, cb, child.get(), plan_root);
			if (r) {
				child = std::move(r);
				changed = true;
			}
		}
	});
	return changed ? std::move(clone) : nullptr;
}

// Convert a _counters aggregate back to its plain pac_* variant in-place.
// Traces the binding through projections to the source aggregate, rebinds it,
// and fixes types along the way. Returns the new scalar type on success.
static bool UndoCountersConversion(const ColumnBinding &binding, LogicalOperator *plan_root, ClientContext &context,
                                   LogicalType &out_type) {
	auto *source = FindOperatorByTableIndex(plan_root, binding.table_index);
	if (!source) {
		return false;
	}
	if (source->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = source->Cast<LogicalProjection>();
		if (binding.column_index >= proj.expressions.size()) {
			return false;
		}
		auto &expr = proj.expressions[binding.column_index];
		if (expr->type != ExpressionType::BOUND_COLUMN_REF) {
			return false;
		}
		auto &inner = expr->Cast<BoundColumnRefExpression>();
		if (!UndoCountersConversion(inner.binding, plan_root, context, out_type)) {
			return false;
		}
		inner.return_type = out_type;
		return true;
	}
	if (source->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = source->Cast<LogicalAggregate>();
		if (binding.table_index != agg.aggregate_index) {
			return false;
		}
		if (binding.column_index >= agg.expressions.size()) {
			return false;
		}
		auto &agg_expr = agg.expressions[binding.column_index];
		if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
			return false;
		}
		auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
		if (!IsPacCountersAggregate(bound_agg.function.name)) {
			return false;
		}
		string plain_name = GetBasePacAggregateName(bound_agg.function.name);
		vector<unique_ptr<Expression>> children;
		for (auto &child : bound_agg.children) {
			children.push_back(child->Copy());
		}
		auto new_agg = RebindAggregate(context, plain_name, std::move(children), bound_agg.IsDistinct());
		if (!new_agg) {
			return false;
		}
		out_type = new_agg->return_type;
		agg.expressions[binding.column_index] = std::move(new_agg);
		return true;
	}
	return false;
}

// Noise a numeric expression containing PAC bindings.
// Bare col_ref: undo the _counters conversion back to plain pac_* (equivalent to pac_noised but cheaper).
// Arithmetic: list_transform + pac_noised via existing BuildCounterListTransform path.
static unique_ptr<Expression> NoisePacNumericExpr(OptimizerExtensionInput &input,
                                                  const vector<PacBindingInfo> &pac_bindings, Expression *expr,
                                                  LogicalOperator *plan_root) {
	Expression *stripped = StripCasts(expr);
	if (pac_bindings.size() == 1 && stripped->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = stripped->Cast<BoundColumnRefExpression>();
		// Bare col_ref to counters: convert back to plain pac_* (no lambda needed)
		LogicalType plain_type;
		if (UndoCountersConversion(col_ref.binding, plan_root, input.context, plain_type)) {
			auto result = make_uniq<BoundColumnRefExpression>(col_ref.alias, plain_type, col_ref.binding);
			if (expr->return_type != plain_type && expr->return_type.id() != LogicalTypeId::LIST) {
				return BoundCastExpression::AddDefaultCastToType(std::move(result), expr->return_type);
			}
			return result;
		}
		// Fallback: wrap with pac_noised
		auto counters =
		    make_uniq<BoundColumnRefExpression>("pac_var", LogicalType::LIST(PacFloatLogicalType()), col_ref.binding);
		auto noised = input.optimizer.BindScalarFunction("pac_noised", std::move(counters));
		if (expr->return_type != PacFloatLogicalType() && expr->return_type.id() != LogicalTypeId::LIST) {
			noised = BoundCastExpression::AddDefaultCastToType(std::move(noised), expr->return_type);
		}
		return noised;
	}
	// Arithmetic: list_transform + pac_noised via existing path
	return RewriteExpressionWithCounters(input, pac_bindings, expr, plan_root);
}

// Build list_transform over PAC counters and wrap with pac_noised.
// pac_bindings: all PAC aggregate bindings in the expression
// expr: the expression to transform (numeric for projection)
// target_type: cast result to this type if different from PAC_FLOAT
static unique_ptr<Expression> RewriteExpressionWithCounters(OptimizerExtensionInput &input,
                                                            const vector<PacBindingInfo> &pac_bindings,
                                                            Expression *expr, LogicalOperator *plan_root,
                                                            const LogicalType &target_type) {
	if (pac_bindings.empty()) {
		return nullptr;
	}
	auto list_expr = BuildCounterListTransform(input, pac_bindings, expr, plan_root, PacFloatLogicalType());
	if (list_expr) {
		auto noised = input.optimizer.BindScalarFunction("pac_noised", std::move(list_expr));
		if (target_type != PacFloatLogicalType()) {
			noised = BoundCastExpression::AddDefaultCastToType(std::move(noised), target_type);
		}
		return noised;
	}
	return nullptr;
}

// Replace column bindings in-place throughout an expression tree
static void ReplaceBindingInExpression(Expression &expr, const ColumnBinding &old_binding,
                                       const ColumnBinding &new_binding) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr.Cast<BoundColumnRefExpression>();
		if (col_ref.binding == old_binding) {
			col_ref.binding = new_binding;
		}
	} else {
		ExpressionIterator::EnumerateChildren(
		    expr, [&](Expression &child) { ReplaceBindingInExpression(child, old_binding, new_binding); });
	}
}

// Rewrite a single projection expression: update col_ref types, build list_transform + terminal.
static void RewriteProjectionExpression(OptimizerExtensionInput &input, LogicalProjection &proj, idx_t i,
                                        LogicalOperator *plan_root, bool is_filter_pattern, bool is_terminal,
                                        unordered_map<uint64_t, unique_ptr<Expression>> &saved_filter_pattern_exprs) {
	auto &expr = proj.expressions[i];
	if (IsAlreadyWrappedInPacNoised(expr.get())) {
		return;
	}
	auto pac_bindings = FindAllPacBindingsInExpression(expr.get(), plan_root);
	if (pac_bindings.empty()) {
		return;
	}
	// Simple column reference to a PAC counter list.
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		// Save original type before overwriting (needed for terminal cast).
		auto original_type = expr->return_type;
		// Update type to match the rewritten aggregate output.
		expr->return_type = LogicalType::LIST(PacFloatLogicalType());
		if (is_terminal) {
			// Terminal projection: wrap with pac_noised to produce scalar output.
			// A bare col_ref has no null-handling operators, so pac_coalesce is unnecessary.
			unique_ptr<Expression> result = input.optimizer.BindScalarFunction("pac_noised", expr->Copy());
			// Cast back to original type if not PAC_FLOAT (e.g., BIGINT for count, DECIMAL for sum).
			if (original_type != PacFloatLogicalType()) {
				result = BoundCastExpression::AddDefaultCastToType(std::move(result), original_type);
			}
			proj.expressions[i] = std::move(result);
			proj.types[i] = original_type;
		} else if (i < proj.types.size()) {
			// Intermediate: pass through as LIST<PAC_FLOAT> for downstream operators.
			proj.types[i] = LogicalType::LIST(PacFloatLogicalType());
		}
		return;
	}
	if (!IsNumericalType(expr->return_type)) {
		return; // we currently only support numeric PAC computations (noising..)
	}
	// Filter pattern simple cast (single aggregate): replace with direct counters ref
	if (is_filter_pattern && pac_bindings.size() == 1) {
		bool is_simple_cast = expr->type == ExpressionType::OPERATOR_CAST &&
		                      expr->Cast<BoundCastExpression>().child->type == ExpressionType::BOUND_COLUMN_REF;
		if (is_simple_cast) {
			proj.expressions[i] = make_uniq<BoundColumnRefExpression>(
			    "pac_var", LogicalType::LIST(PacFloatLogicalType()), pac_bindings[0].binding);
			proj.types[i] = LogicalType::LIST(PacFloatLogicalType());
			return;
		}
	}
	// Determine expression to clone (strip outer CAST if needed)
	Expression *expr_to_clone = expr.get();
	if (expr->type == ExpressionType::OPERATOR_CAST &&
	    expr->Cast<BoundCastExpression>().return_type != PacFloatLogicalType()) {
		expr_to_clone = expr->Cast<BoundCastExpression>().child.get();
	}
	if (is_filter_pattern) { // Intermediate: produce LIST<DOUBLE> for downstream filter (no terminal wrapping)
		if (pac_bindings.size() == 1) {
			// Save the original arithmetic expression with PAC col_ref rebased to the projection output binding.
			// The filter will inline this so that e.g. `col_ref(proj, i) > value` becomes
			// `multiply(0.5, col_ref(proj, i)) > value`, and FindAllPacBindingsInExpression
			// traces col_ref(proj, i) through the raw counter pass-through to the aggregate.
			auto saved = expr->Copy();
			ColumnBinding proj_binding(proj.table_index, i);
			ReplaceBindingInExpression(*saved, pac_bindings[0].binding, proj_binding);
			saved_filter_pattern_exprs[HashBinding(proj_binding)] = std::move(saved);

			// Replace projection expression with raw counter pass-through
			proj.expressions[i] = make_uniq<BoundColumnRefExpression>(
			    "pac_var", LogicalType::LIST(PacFloatLogicalType()), pac_bindings[0].binding);
			proj.types[i] = LogicalType::LIST(PacFloatLogicalType());
		} else {
			// Multiple bindings: keep existing list_transform (fallback)
			auto list_expr =
			    BuildCounterListTransform(input, pac_bindings, expr_to_clone, plan_root, PacFloatLogicalType());
			if (list_expr) {
				proj.expressions[i] = std::move(list_expr);
				proj.types[i] = LogicalType::LIST(PacFloatLogicalType());
			}
		}
	} else {
		auto result = RewriteExpressionWithCounters(input, pac_bindings, expr_to_clone, plan_root, expr->return_type);
		if (result) {
			proj.expressions[i] = std::move(result);
			proj.types[i] = expr->return_type;
		}
	}
}

// Inline saved projection arithmetic expressions into an expression tree.
// Replaces col_refs matching saved bindings with the saved expression.
// Avoids recursing into replaced nodes (which would cause infinite recursion since
// the saved expression itself contains the same binding as an intermediate col_ref).
static void InlineSavedExpressionsIntoExpr(unique_ptr<Expression> &expr,
                                           const unordered_map<uint64_t, unique_ptr<Expression>> &saved_exprs) {
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		uint64_t key = HashBinding(expr->Cast<BoundColumnRefExpression>().binding);
		auto it = saved_exprs.find(key);
		if (it != saved_exprs.end()) {
			expr = it->second->Copy();
			return; // Don't recurse into the replacement
		}
	}
	ExpressionIterator::EnumerateChildren( // Recurse into children, but stop when a replacement is made
	    *expr, [&](unique_ptr<Expression> &child) { InlineSavedExpressionsIntoExpr(child, saved_exprs); });
}

// Single bottom-up rewrite pass.
// Processes children first, then current operator. Handles:
// - AGGREGATE: convert pac_sum → pac_sum_counters, then aggregate-over-counters → _list
// - PROJECTION: update simple col_ref types, build list_transform + pac_noised for arithmetic
// - FILTER (in rewrite_map): noise PAC sides of comparisons with pac_noised
// - JOIN (in rewrite_map): noise PAC sides of join conditions with pac_noised
static void RewriteBottomUp(unique_ptr<LogicalOperator> &op_ptr, OptimizerExtensionInput &input,
                            unique_ptr<LogicalOperator> &plan,
                            const unordered_map<LogicalOperator *, unordered_set<idx_t>> &pattern_lookup,
                            vector<CategoricalPatternInfo> &patterns,
                            unordered_map<uint64_t, unique_ptr<Expression>> &saved_filter_pattern_exprs) {
	auto *op = op_ptr.get();
	// Strip scalar wrappers (Projection→first()→Projection) over PAC aggregates before recursing.
	// This removes the first() aggregate that can't handle LIST<DOUBLE>, and lets the inner
	// projection be processed naturally with the outer's table_index.
	if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto *unwrapped = RecognizeDuckDBScalarWrapper(op);
		if (unwrapped && !FindPacAggregateInOperator(unwrapped).empty()) {
			StripScalarWrapperInPlace(op_ptr, true);
			op = op_ptr.get();
		}
	}
	for (auto &child : op->children) { // Recurse into children first (bottom-up)
		RewriteBottomUp(child, input, plan, pattern_lookup, patterns, saved_filter_pattern_exprs);
	}
	LogicalOperator *plan_root = plan.get();
	if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		// === AGGREGATE: convert PAC aggregates to _counters, then check aggregates-over-counters ===
		auto &agg = op->Cast<LogicalAggregate>();

		// Convert PAC aggregates to _counters variants
		for (idx_t i = 0; i < agg.expressions.size(); i++) {
			auto &agg_expr = agg.expressions[i];
			if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
				continue;
			}
			auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
			if (!IsPacAggregate(bound_agg.function.name)) {
				continue;
			}
			string counters_name = GetCountersVariant(bound_agg.function.name);
			vector<unique_ptr<Expression>> children;
			for (auto &child_expr : bound_agg.children) {
				children.push_back(child_expr->Copy());
			}
			auto new_aggr = RebindAggregate(input.context, counters_name, std::move(children), bound_agg.IsDistinct());
			if (!new_aggr) { // Fallback: just rename the function in place
				bound_agg.function.name = counters_name;
				bound_agg.function.return_type = LogicalType::LIST(PacFloatLogicalType());
				agg_expr->return_type = LogicalType::LIST(PacFloatLogicalType());
			} else {
				agg.expressions[i] = std::move(new_aggr);
			}
			idx_t types_index = agg.groups.size() + i;
			if (types_index < agg.types.size()) {
				agg.types[types_index] = LogicalType::LIST(PacFloatLogicalType());
			}
		}
		// Check for standard aggregates over counters (e.g., sum(LIST<DOUBLE>) → pac_sum_list)
		// Children already converted (bottom-up), so their types are LIST<DOUBLE>
		ReplaceAggregatesOverCounters(op, input.context, plan_root);
	} else if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) { // === PROJECTION: rewrite PAC expressions ===
		auto &proj = op->Cast<LogicalProjection>();
		bool is_filter_pattern = IsProjectionReferencedByFilterPattern(proj, patterns, plan_root);
		// A projection is terminal if it's the top-level output projection:
		// either it IS the plan root, or the plan root is ORDER_BY/TOP_N/LIMIT whose child is this projection.
		bool is_terminal = (op == plan_root);
		if (!is_terminal) {
			auto *root = plan_root;
			while (root &&
			       (root->type == LogicalOperatorType::LOGICAL_ORDER_BY ||
			        root->type == LogicalOperatorType::LOGICAL_TOP_N ||
			        root->type == LogicalOperatorType::LOGICAL_LIMIT) &&
			       !root->children.empty()) {
				root = root->children[0].get();
			}
			is_terminal = (root == op);
		}
		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			RewriteProjectionExpression(input, proj, i, plan_root, is_filter_pattern, is_terminal,
			                            saved_filter_pattern_exprs);
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_FILTER) { // === FILTER: noise PAC sides of comparisons ===
		// Inline saved projection arithmetic expressions into filter expressions.
		// This fuses the projection's list_transform into the filter's lambda for a single pass.
		if (!saved_filter_pattern_exprs.empty()) {
			auto &filter = op->Cast<LogicalFilter>();
			for (auto &fexpr : filter.expressions) {
				InlineSavedExpressionsIntoExpr(fexpr, saved_filter_pattern_exprs);
			}
		}
		auto it = pattern_lookup.find(op);
		if (it != pattern_lookup.end()) {
			auto &filter = op->Cast<LogicalFilter>();
			for (auto expr_idx : it->second) {
				if (expr_idx >= filter.expressions.size()) {
					continue;
				}
				auto &filter_expr = filter.expressions[expr_idx];
				auto pac_bindings = FindAllPacBindingsInExpression(filter_expr.get(), plan_root);
				if (pac_bindings.empty()) {
					continue;
				}
				auto result = RewriteFilterWithNoised(input, pac_bindings, filter_expr.get(), plan_root);
				if (result) {
					filter.expressions[expr_idx] = std::move(result);
				}
			}
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	           op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) { // === JOIN: noise PAC sides of conditions ===
		// Inline saved projection arithmetic into join conditions
		if (!saved_filter_pattern_exprs.empty()) {
			auto &join = op->Cast<LogicalComparisonJoin>();
			for (auto &cond : join.conditions) {
				InlineSavedExpressionsIntoExpr(cond.left, saved_filter_pattern_exprs);
				InlineSavedExpressionsIntoExpr(cond.right, saved_filter_pattern_exprs);
			}
		}
		auto it = pattern_lookup.find(op);
		if (it == pattern_lookup.end()) {
			return;
		}
		auto &join = op->Cast<LogicalComparisonJoin>();
		for (auto expr_idx : it->second) {
			if (expr_idx >= join.conditions.size()) {
				continue;
			}
			auto &cond = join.conditions[expr_idx];
			auto lb = FindAllPacBindingsInExpression(cond.left.get(), plan_root);
			auto rb = FindAllPacBindingsInExpression(cond.right.get(), plan_root);
			if (!lb.empty()) {
				auto r = NoisePacNumericExpr(input, lb, cond.left.get(), plan_root);
				if (r) {
					cond.left = std::move(r);
				}
			}
			if (!rb.empty()) {
				auto r = NoisePacNumericExpr(input, rb, cond.right.get(), plan_root);
				if (r) {
					cond.right = std::move(r);
				}
			}
		}
	}
}

void RewriteCategoricalQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	// Detect categorical patterns
	vector<CategoricalPatternInfo> patterns;
	FindCategoricalPatternsInOperator(plan.get(), plan.get(), patterns, false);
	if (patterns.empty()) {
		return;
	}
	// Build lightweight lookup: operator → set of expression indices
	unordered_map<LogicalOperator *, unordered_set<idx_t>> pattern_lookup;
	for (auto &p : patterns) {
		if (p.parent_op) {
			pattern_lookup[p.parent_op].insert(p.expr_index);
		}
	}
	// Bottom-up rewrite pass
	// - Aggregates: pac_sum → pac_sum_counters, then aggregate-over-counters → _list
	// - Projections: update col_ref types, build list_transform + pac_noised/pass-through
	// - Filters: noise PAC sides of comparisons with pac_noised
	// - Joins: noise PAC sides of join conditions with pac_noised
	unordered_map<uint64_t, unique_ptr<Expression>> saved_filter_pattern_exprs;
	RewriteBottomUp(plan, input, plan, pattern_lookup, patterns, saved_filter_pattern_exprs);
	plan->ResolveOperatorTypes();
}

} // namespace duckdb
