//
// PAC Categorical Query Rewriter - Implementation
//
// See pac_categorical_rewriter.hpp for design documentation.
//
// Created by ila on 1/23/26.
//
#include "categorical/pac_categorical_rewriter.hpp"

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
	}
	// Check join conditions (for semi/anti/mark joins with subqueries)
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
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
			// Check if this expression has arithmetic with PAC aggregates
			// - Multiple PAC bindings (e.g., pac_sum(...) / pac_sum(...))
			// - Or single PAC binding with arithmetic (e.g., pac_sum(...) * 0.5)
			bool is_arithmetic_with_pac = false;
			if (pac_bindings.size() >= 2) { // Multiple PAC aggregates - definitely needs lambda rewrite
				is_arithmetic_with_pac = true;
			} else if (pac_bindings.size() == 1) {
				// Single aggregate - check if it's in an arithmetic expression (not just a column ref or simple cast)
				if (expr->type != ExpressionType::BOUND_COLUMN_REF && expr->type != ExpressionType::OPERATOR_CAST) {
					is_arithmetic_with_pac = true;
				} else if (expr->type == ExpressionType::OPERATOR_CAST) {
					// Check if cast contains arithmetic
					auto &cast_expr = expr->Cast<BoundCastExpression>();
					if (cast_expr.child->type != ExpressionType::BOUND_COLUMN_REF) {
						is_arithmetic_with_pac = true;
					}
				}
			}
			if (is_arithmetic_with_pac) {
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
				child_copy->return_type = LogicalType::LIST(LogicalType::DOUBLE);
			}
			children.push_back(std::move(child_copy));
		}
		auto new_aggr = RebindAggregate(context, list_variant, std::move(children), bound_agg.IsDistinct());
		if (new_aggr) {
			agg.expressions[i] = std::move(new_aggr);
			idx_t types_index = agg.groups.size() + i;
			if (types_index < agg.types.size()) {
				agg.types[types_index] = LogicalType::LIST(LogicalType::DOUBLE);
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
	auto elem_ref = make_uniq<BoundReferenceExpression>("elem", struct_type, 0);
	auto child_types = StructType::GetChildTypes(struct_type);
	LogicalType extract_return_type = LogicalType::DOUBLE;
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
// element_type: the type of the lambda parameter. For single binding, this is the aggregate's
//               original type (or DOUBLE for raw counters). For multi binding, always DOUBLE.
// struct_type: nullptr = single binding (elem ref), non-null = multi binding (struct field extract)
static unique_ptr<Expression>
CloneForLambdaBody(Expression *expr, const unordered_map<uint64_t, idx_t> &pac_binding_map,
                   vector<unique_ptr<Expression>> &captures, unordered_map<uint64_t, idx_t> &capture_map,
                   LogicalOperator *plan_root, const LogicalType &element_type, const LogicalType *struct_type) {
	// Helper to build the replacement expression for a matched PAC binding
	auto make_pac_replacement = [&](const unordered_map<uint64_t, idx_t>::const_iterator &it,
	                                const string &alias) -> unique_ptr<Expression> {
		if (struct_type) {
			return BuildStructFieldExtract(*struct_type, it->second, GetStructFieldName(it->second));
		} else {
			return make_uniq<BoundReferenceExpression>(alias, element_type, 0);
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
		auto child_clone = CloneForLambdaBody(cast.child.get(), pac_binding_map, captures, capture_map, plan_root,
		                                      element_type, struct_type);
		if (child_clone->return_type == cast.return_type) {
			return child_clone; // If the child's type already matches the target type, skip the cast
		}
		// Otherwise, create a new cast with the correct function for the new child type
		return BoundCastExpression::AddDefaultCastToType(std::move(child_clone), cast.return_type);
	} else if (expr->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
		auto &comp = expr->Cast<BoundComparisonExpression>();
		auto left_clone = CloneForLambdaBody(comp.left.get(), pac_binding_map, captures, capture_map, plan_root,
		                                     element_type, struct_type);
		auto right_clone = CloneForLambdaBody(comp.right.get(), pac_binding_map, captures, capture_map, plan_root,
		                                      element_type, struct_type);
		// Reconcile types if they differ (needed for multi where struct fields are DOUBLE
		// but original CASTs may introduce DECIMAL)
		if (left_clone->return_type != right_clone->return_type) {
			if (left_clone->return_type != LogicalType::DOUBLE) {
				left_clone = BoundCastExpression::AddDefaultCastToType(std::move(left_clone), LogicalType::DOUBLE);
			}
			if (right_clone->return_type != LogicalType::DOUBLE) {
				right_clone = BoundCastExpression::AddDefaultCastToType(std::move(right_clone), LogicalType::DOUBLE);
			}
		}
		return make_uniq<BoundComparisonExpression>(expr->type, std::move(left_clone), std::move(right_clone));
	} else if (expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = expr->Cast<BoundFunctionExpression>();
		vector<unique_ptr<Expression>> new_children;
		bool any_cast_needed = false;
		for (idx_t i = 0; i < func.children.size(); i++) {
			auto child_clone = CloneForLambdaBody(func.children[i].get(), pac_binding_map, captures, capture_map,
			                                      plan_root, element_type, struct_type);
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
		// When element_type is DOUBLE (raw counters context or multi binding), children were cast
		// from DOUBLE to the function's bound types (e.g., DECIMAL). Cast the result back to DOUBLE
		// so the list_transform output stays LIST<DOUBLE>.
		if (any_cast_needed && element_type == LogicalType::DOUBLE && result->return_type != LogicalType::DOUBLE) {
			result = BoundCastExpression::AddDefaultCastToType(std::move(result), LogicalType::DOUBLE);
		}
		return result;
	} else if (expr->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) { // AND, OR, NOT, arithmetic, COALESCE..
		auto &op = expr->Cast<BoundOperatorExpression>();
		vector<unique_ptr<Expression>> new_children;
		for (auto &child : op.children) {
			new_children.push_back(CloneForLambdaBody(child.get(), pac_binding_map, captures, capture_map, plan_root,
			                                          element_type, struct_type));
		}
		return BuildClonedOperatorExpression(expr, std::move(new_children));
	} else if (expr->type == ExpressionType::CASE_EXPR) {
		auto &case_expr = expr->Cast<BoundCaseExpression>();
		if (IsScalarSubqueryWrapper(case_expr)) {
			return CloneForLambdaBody(case_expr.else_expr.get(), pac_binding_map, captures, capture_map, plan_root,
			                          element_type, struct_type);
		}
		// Regular CASE - recurse into all branches
		// Start with original return type, will update based on cloned branches
		auto result = make_uniq<BoundCaseExpression>(case_expr.return_type);
		for (auto &check : case_expr.case_checks) {
			BoundCaseCheck new_check;
			new_check.when_expr = CloneForLambdaBody(check.when_expr.get(), pac_binding_map, captures, capture_map,
			                                         plan_root, element_type, struct_type);
			new_check.then_expr = CloneForLambdaBody(check.then_expr.get(), pac_binding_map, captures, capture_map,
			                                         plan_root, element_type, struct_type);
			result->case_checks.push_back(std::move(new_check));
		}
		if (case_expr.else_expr) {
			result->else_expr = CloneForLambdaBody(case_expr.else_expr.get(), pac_binding_map, captures, capture_map,
			                                       plan_root, element_type, struct_type);
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
	    make_uniq<BoundLambdaExpression>(ExpressionType::LAMBDA, LogicalType::LAMBDA, std::move(lambda_body), 1);
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
		struct_children.push_back(make_pair(field_name, LogicalType::DOUBLE)); // All counter lists are LIST<DOUBLE>
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
//
// For single binding: list_transform(pac_coalesce(counters), elem -> body(elem))
//   If element_type != DOUBLE, inserts an inner cast lambda first (double-lambda approach)
//   so the outer lambda body sees elements of element_type rather than raw DOUBLE.
//
// For multiple bindings: list_transform(list_zip(c1, c2, ...), elem -> body(elem.a, elem.b, ...))
//   CloneForLambdaBody (multi mode) handles per-field type casting internally.
//
// Returns the list_transform expression, or nullptr on failure.
// The caller wraps with pac_noised or pac_filter as needed.
static unique_ptr<Expression> BuildCounterListTransform(OptimizerExtensionInput &input,
                                                        const vector<PacBindingInfo> &pac_bindings,
                                                        Expression *expr_to_transform, LogicalOperator *plan_root,
                                                        const LogicalType &element_type,
                                                        const LogicalType &result_element_type) {
	if (pac_bindings.size() == 1) { // --- SINGLE AGGREGATE ---
		auto &binding_info = pac_bindings[0];
		ColumnBinding pac_binding = binding_info.binding;
		auto counters_ref =
		    make_uniq<BoundColumnRefExpression>("pac_var", LogicalType::LIST(LogicalType::DOUBLE), pac_binding);
		auto safe_counters = input.optimizer.BindScalarFunction("pac_coalesce", std::move(counters_ref));

		unique_ptr<Expression> input_list;
		if (element_type != LogicalType::DOUBLE) {
			// Double-lambda: first transform DOUBLE -> element_type
			auto inner_elem = make_uniq<BoundReferenceExpression>("elem", LogicalType::DOUBLE, 0);
			unique_ptr<Expression> inner_body =
			    BoundCastExpression::AddDefaultCastToType(std::move(inner_elem), element_type);
			auto inner_lambda = BuildPacLambda(std::move(inner_body), {});
			input_list = BuildListTransformCall(input, std::move(safe_counters), std::move(inner_lambda), element_type);
		} else {
			input_list = std::move(safe_counters);
		}
		// Outer lambda: clone the expression replacing PAC binding with lambda element
		unordered_map<uint64_t, idx_t> pac_binding_map;
		pac_binding_map[HashBinding(pac_binding)] = 0;
		vector<unique_ptr<Expression>> captures;
		unordered_map<uint64_t, idx_t> capture_map;
		auto lambda_body = CloneForLambdaBody(expr_to_transform, pac_binding_map, captures, capture_map, plan_root,
		                                      element_type, nullptr);
		// Ensure body returns result_element_type if needed
		if (result_element_type == LogicalType::DOUBLE && lambda_body->return_type != LogicalType::DOUBLE) {
			lambda_body = BoundCastExpression::AddDefaultCastToType(std::move(lambda_body), LogicalType::DOUBLE);
		}
		auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));
		return BuildListTransformCall(input, std::move(input_list), std::move(lambda), result_element_type);
	} else { // --- MULTIPLE AGGREGATES ---
		vector<unique_ptr<Expression>> counter_lists;
		unordered_map<uint64_t, idx_t> binding_to_index;
		for (auto &bi : pac_bindings) {
			counter_lists.push_back(input.optimizer.BindScalarFunction(
			    "pac_coalesce",
			    make_uniq<BoundColumnRefExpression>("pac_var", LogicalType::LIST(LogicalType::DOUBLE), bi.binding)));
			binding_to_index[HashBinding(bi.binding)] = bi.index;
		}
		LogicalType struct_type;
		auto zipped_list = BuildListZipCall(input, std::move(counter_lists), struct_type);
		if (zipped_list) {
			vector<unique_ptr<Expression>> captures;
			unordered_map<uint64_t, idx_t> map;
			auto lambda_body = CloneForLambdaBody(expr_to_transform, binding_to_index, captures, map, plan_root,
			                                      LogicalType::DOUBLE, &struct_type);
			if (result_element_type == LogicalType::DOUBLE && lambda_body->return_type != LogicalType::DOUBLE) {
				lambda_body = BoundCastExpression::AddDefaultCastToType(std::move(lambda_body), LogicalType::DOUBLE);
			}
			auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));
			return BuildListTransformCall(input, std::move(zipped_list), std::move(lambda), result_element_type);
		}
	}
	return nullptr;
}

// Unified expression rewrite: build list_transform over PAC counters and wrap.
// pac_bindings: all PAC aggregate bindings in the expression
// expr: the expression to transform (boolean for filter, numeric for projection)
// wrap_kind: determines terminal function and type parameters
// target_type: for PAC_NOISED, cast result to this type (ignored for PAC_FILTER)
static unique_ptr<Expression> RewriteExpressionWithCounters(OptimizerExtensionInput &input,
                                                            const vector<PacBindingInfo> &pac_bindings,
                                                            Expression *expr, LogicalOperator *plan_root,
                                                            PacWrapKind wrap_kind,
                                                            const LogicalType &target_type = LogicalType::DOUBLE) {
	LogicalType element_type, result_element_type;
	if (pac_bindings.empty()) {
		return nullptr;
	}
	if (wrap_kind == PacWrapKind::PAC_NOISED) {
		element_type = LogicalType::DOUBLE;
		result_element_type = LogicalType::DOUBLE;
	} else {
		// Filter/Join: use original type for single binding (enables double-lambda),
		// DOUBLE for multi-binding (list_zip always produces DOUBLE)
		element_type = (pac_bindings.size() == 1) ? pac_bindings[0].original_type : LogicalType::DOUBLE;
		result_element_type = LogicalType::BOOLEAN;
	}
	auto list_expr = BuildCounterListTransform(input, pac_bindings, expr, plan_root, element_type, result_element_type);
	if (list_expr) {
		if (wrap_kind == PacWrapKind::PAC_NOISED) {
			auto noised = input.optimizer.BindScalarFunction("pac_noised", std::move(list_expr));
			if (target_type != LogicalType::DOUBLE) {
				noised = BoundCastExpression::AddDefaultCastToType(std::move(noised), target_type);
			}
			return noised;
		} else {
			return input.optimizer.BindScalarFunction("pac_filter", std::move(list_expr));
		}
	}
	return nullptr;
}

// Build a map from parent operators to their ExpressionRewriteInfo entries.
// Groups patterns by (parent_op, expr_index) and pre-collects PAC bindings per expression.
static unordered_map<LogicalOperator *, vector<ExpressionRewriteInfo>>
BuildRewriteMap(const vector<CategoricalPatternInfo> &patterns, LogicalOperator *plan_root) {
	unordered_map<LogicalOperator *, vector<ExpressionRewriteInfo>> result;
	unordered_set<uint64_t> seen;

	for (auto &pattern : patterns) {
		ExpressionRewriteInfo info;
		info.original_return_type = pattern.original_return_type;
		info.parent_op = pattern.parent_op;
		info.expr_index = pattern.expr_index;

		// Key: combine pointer and expr_index for uniqueness
		uint64_t key = reinterpret_cast<uint64_t>(pattern.parent_op) ^ (uint64_t(pattern.expr_index) << 48);
		if (seen.count(key)) {
			continue;
		}
		seen.insert(key);

		// Use pre-collected bindings from detection when available, otherwise compute
		if (!pattern.pac_bindings.empty()) {
			info.pac_bindings = pattern.pac_bindings;
		} else if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = pattern.parent_op->Cast<LogicalFilter>();
			if (pattern.expr_index < filter.expressions.size()) {
				info.pac_bindings =
				    FindAllPacBindingsInExpression(filter.expressions[pattern.expr_index].get(), plan_root);
			}
		} else if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = pattern.parent_op->Cast<LogicalProjection>();
			if (pattern.expr_index < proj.expressions.size()) {
				info.pac_bindings =
				    FindAllPacBindingsInExpression(proj.expressions[pattern.expr_index].get(), plan_root);
			}
		}
		if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		    pattern.parent_op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &join = pattern.parent_op->Cast<LogicalComparisonJoin>();
			if (pattern.expr_index < join.conditions.size()) {
				info.left_pac_bindings =
				    FindAllPacBindingsInExpression(join.conditions[pattern.expr_index].left.get(), plan_root);
				info.right_pac_bindings =
				    FindAllPacBindingsInExpression(join.conditions[pattern.expr_index].right.get(), plan_root);
				for (auto &b : info.left_pac_bindings) {
					info.pac_bindings.push_back(b);
				}
				for (auto &b : info.right_pac_bindings) {
					info.pac_bindings.push_back(b);
				}
			}
		}
		result[pattern.parent_op].push_back(std::move(info));
	}
	return result;
}

// Rewrite a single projection expression: update col_ref types, build list_transform + terminal.
static void RewriteProjectionExpression(OptimizerExtensionInput &input, LogicalProjection &proj, idx_t i,
                                        LogicalOperator *plan_root, bool is_filter_pattern) {
	auto &expr = proj.expressions[i];
	if (IsAlreadyWrappedInPacNoised(expr.get())) {
		return;
	}
	auto pac_bindings = FindAllPacBindingsInExpression(expr.get(), plan_root);
	if (pac_bindings.empty()) {
		return;
	}
	// Simple column reference to a PAC counter list: always pass through as LIST<DOUBLE>.
	// These are always intermediates consumed by a higher operator (filter, join, aggregate, or
	// arithmetic projection) that will apply its own terminal wrapping (pac_filter or pac_noised).
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);
		if (i < proj.types.size()) {
			proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
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
			proj.expressions[i] = make_uniq<BoundColumnRefExpression>("pac_var", LogicalType::LIST(LogicalType::DOUBLE),
			                                                          pac_bindings[0].binding);
			proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
			return;
		}
	}
	// Determine expression to clone (strip outer CAST if needed)
	Expression *expr_to_clone = expr.get();
	if (expr->type == ExpressionType::OPERATOR_CAST &&
	    expr->Cast<BoundCastExpression>().return_type != LogicalType::DOUBLE) {
		expr_to_clone = expr->Cast<BoundCastExpression>().child.get();
	}
	if (is_filter_pattern) { // Intermediate: produce LIST<DOUBLE> for downstream filter (no terminal wrapping)
		auto list_expr = BuildCounterListTransform(input, pac_bindings, expr_to_clone, plan_root, LogicalType::DOUBLE,
		                                           LogicalType::DOUBLE);
		if (list_expr) {
			proj.expressions[i] = std::move(list_expr);
			proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
		}
	} else {
		auto result = RewriteExpressionWithCounters(input, pac_bindings, expr_to_clone, plan_root,
		                                            PacWrapKind::PAC_NOISED, expr->return_type);
		if (result) {
			proj.expressions[i] = std::move(result);
			proj.types[i] = expr->return_type;
		}
	}
}

// Single bottom-up rewrite pass.
// Processes children first, then current operator. Handles:
// - AGGREGATE: convert pac_sum → pac_sum_counters, then aggregate-over-counters → _list
// - PROJECTION: update simple col_ref types, build list_transform + pac_noised for arithmetic
// - FILTER (in rewrite_map): build list_transform + pac_filter
// - JOIN (in rewrite_map): rewrite conditions (two-list → CROSS_PRODUCT+FILTER, single-list → double-lambda)
static void RewriteBottomUp(unique_ptr<LogicalOperator> &op_ptr, OptimizerExtensionInput &input,
                            unique_ptr<LogicalOperator> &plan,
                            const unordered_map<LogicalOperator *, vector<ExpressionRewriteInfo>> &rewrite_map,
                            vector<CategoricalPatternInfo> &patterns) {
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
		RewriteBottomUp(child, input, plan, rewrite_map, patterns);
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
				bound_agg.function.return_type = LogicalType::LIST(LogicalType::DOUBLE);
				agg_expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);
			} else {
				agg.expressions[i] = std::move(new_aggr);
			}
			idx_t types_index = agg.groups.size() + i;
			if (types_index < agg.types.size()) {
				agg.types[types_index] = LogicalType::LIST(LogicalType::DOUBLE);
			}
		}
		// Check for standard aggregates over counters (e.g., sum(LIST<DOUBLE>) → pac_sum_list)
		// Children already converted (bottom-up), so their types are LIST<DOUBLE>
		ReplaceAggregatesOverCounters(op, input.context, plan_root);
	} else if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) { // === PROJECTION: rewrite PAC expressions ===
		auto &proj = op->Cast<LogicalProjection>();
		bool is_filter_pattern = IsProjectionReferencedByFilterPattern(proj, patterns, plan_root);
		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			RewriteProjectionExpression(input, proj, i, plan_root, is_filter_pattern);
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_FILTER) { // === FILTER: rewrite expressions with pac_filter ===
		auto it = rewrite_map.find(op);
		if (it != rewrite_map.end()) {
			auto &filter = op->Cast<LogicalFilter>();
			unordered_set<idx_t> processed;

			for (auto &info : it->second) {
				if (processed.count(info.expr_index)) {
					continue;
				}
				processed.insert(info.expr_index);
				if (info.expr_index >= filter.expressions.size()) {
					continue;
				}
				auto &filter_expr = filter.expressions[info.expr_index];
				auto pac_bindings = FindAllPacBindingsInExpression(filter_expr.get(), plan_root);
				if (pac_bindings.empty()) {
					continue;
				}
				auto result = RewriteExpressionWithCounters(input, pac_bindings, filter_expr.get(), plan_root,
				                                            PacWrapKind::PAC_FILTER);
				if (result) {
					filter.expressions[info.expr_index] = std::move(result);
				}
			}
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	           op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) { // === JOIN: rewrite comparison conditions ===
		auto it = rewrite_map.find(op);
		if (it != rewrite_map.end()) {
			auto &join = op->Cast<LogicalComparisonJoin>();
			for (auto &info : it->second) {
				if (info.expr_index >= join.conditions.size()) {
					continue;
				}
				auto &cond = join.conditions[info.expr_index];
				auto left_bindings = FindAllPacBindingsInExpression(cond.left.get(), plan_root);
				auto right_bindings = FindAllPacBindingsInExpression(cond.right.get(), plan_root);
				bool left_is_list = !left_bindings.empty();
				bool right_is_list = !right_bindings.empty();
				if (!left_is_list && !right_is_list) {
					continue;
				}
				if (left_is_list && right_is_list) { // Both sides are lists: CROSS_PRODUCT + FILTER with list_zip
					// Combine all PAC bindings, deduplicating by hash
					vector<PacBindingInfo> all_bindings;
					unordered_set<uint64_t> seen_hashes;
					for (auto &b : left_bindings) {
						uint64_t h = HashBinding(b.binding);
						if (seen_hashes.insert(h).second) {
							all_bindings.push_back(b);
						}
					}
					for (auto &b : right_bindings) {
						uint64_t h = HashBinding(b.binding);
						if (seen_hashes.insert(h).second) {
							all_bindings.push_back(b);
						}
					}
					for (idx_t j = 0; j < all_bindings.size(); j++) {
						all_bindings[j].index = j;
					}
					auto comparison = // Build comparison expression from the join condition
					    make_uniq<BoundComparisonExpression>(cond.comparison, cond.left->Copy(), cond.right->Copy());
					auto pac_filter_expr = RewriteExpressionWithCounters(input, all_bindings, comparison.get(),
					                                                     plan_root, PacWrapKind::PAC_FILTER);
					if (pac_filter_expr) {
						auto cross_product = // Convert COMPARISON_JOIN to CROSS_PRODUCT + FILTER
						    LogicalCrossProduct::Create(std::move(join.children[0]), std::move(join.children[1]));
						auto filter_op = make_uniq<LogicalFilter>();
						filter_op->expressions.push_back(std::move(pac_filter_expr));
						filter_op->children.push_back(std::move(cross_product));

						// Invalidate all patterns that pointed to the old join (now destroyed)
						for (auto &p : patterns) {
							if (p.parent_op == op) {
								p.parent_op = nullptr;
							}
						}
						op_ptr = std::move(filter_op);
						break; // Replaced operator, can't process more conditions
					}
				} else {
					// One side is list: rewrite the comparison with pac_filter
					auto &pac_bindings = left_is_list ? left_bindings : right_bindings;
					if (pac_bindings.empty()) {
						continue;
					}
					auto comparison =
					    make_uniq<BoundComparisonExpression>(cond.comparison, cond.left->Copy(), cond.right->Copy());
					auto result = RewriteExpressionWithCounters(input, pac_bindings, comparison.get(), plan_root,
					                                            PacWrapKind::PAC_FILTER);
					if (result) {
						cond.left = std::move(result);
						cond.right = make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));
						cond.comparison = ExpressionType::COMPARE_EQUAL;
					}
				}
			}
		}
	}
}

void RewriteCategoricalQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                             vector<CategoricalPatternInfo> &patterns) {
	// Find patterns in first pass and create a rewrite map from patterns (grouped by expression)
	auto rewrite_map = BuildRewriteMap(patterns, plan.get());

	// Second bottom-up pass doing ALL rewrites
	// - Aggregates: pac_sum → pac_sum_counters, then aggregate-over-counters → _list
	// - Projections: update col_ref types, build list_transform + pac_noised/pass-through
	// - Filters: build list_transform + pac_filter
	// - Joins: rewrite conditions (two-list → CROSS_PRODUCT+FILTER, single-list → double-lambda)
	RewriteBottomUp(plan, input, plan, rewrite_map, patterns);
	plan->ResolveOperatorTypes();
}

} // namespace duckdb
