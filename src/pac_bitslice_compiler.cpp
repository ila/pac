//
// Created by ila on 12/21/25.
//

#include "include/pac_bitslice_compiler.hpp"
#include "include/pac_helpers.hpp"
#include "include/pac_compatibility_check.hpp"
#include "include/pac_compiler_helpers.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog.hpp"

namespace duckdb {

// Forward-declare helper so it can be used by functions defined earlier in this file.
static idx_t EnsureProjectedColumn(LogicalGet &g, const std::string &col_name);

// Helper: ensure PK columns are present in a LogicalGet's column_ids and projection_ids.
// If a PK column is not present in the scan, this function adds its column id to the LogicalGet
// and ensures it is projected and that column bindings are regenerated.
void AddPKColumns(LogicalGet &get, const std::vector<std::string> &pks) {
	// We'll look up the table to map column names to column indexes
	auto table_entry_ptr = get.GetTable();
	if (!table_entry_ptr) {
		throw InternalException("PAC compiler: expected LogicalGet to be bound to a table when PKs are present");
	}
	// For each PK name, ensure it is present in the LogicalGet's column_ids and projection
	for (auto &pk : pks) {
		idx_t proj_idx = EnsureProjectedColumn(get, pk);
		if (proj_idx == DConstants::INVALID_INDEX) {
			// If the helper couldn't find the column in the table, this is an error
			throw InternalException("PAC compiler: could not find PK column " + pk + " in table");
		}
	}
}

// New helper: ensure a column is projected in a LogicalGet and return its projection index
// If the column is not present, attempt to add it from the underlying table schema. Returns
// DConstants::INVALID_INDEX if the table or column cannot be found (caller can decide how to handle).
static idx_t EnsureProjectedColumn(LogicalGet &g, const std::string &col_name) {
	// try existing projected columns by matching returned names via ColumnIndex primary
	for (idx_t cid = 0; cid < g.GetColumnIds().size(); ++cid) {
		auto col_idx = g.GetColumnIds()[cid];
		if (!col_idx.IsVirtualColumn()) {
			idx_t primary = col_idx.GetPrimaryIndex();
			if (primary < g.names.size() && g.names[primary] == col_name) {
				return cid;
			}
		}
	}
	// otherwise add column from table schema
	auto table_entry = g.GetTable();
	if (!table_entry) {
		return DConstants::INVALID_INDEX;
	}
	idx_t logical_idx = DConstants::INVALID_INDEX;
	idx_t ti = 0;
	for (auto &col : table_entry->GetColumns().Logical()) {
		if (col.Name() == col_name) {
			logical_idx = ti;
			break;
		}
		ti++;
	}
	if (logical_idx == DConstants::INVALID_INDEX) {
		return DConstants::INVALID_INDEX;
	}
	g.AddColumnId(logical_idx);
	g.projection_ids.push_back(g.GetColumnIds().size() - 1);
	g.GenerateColumnBindings(g.table_index, g.GetColumnIds().size());
	return g.GetColumnIds().size() - 1;
}

// Helper: build XOR(pk1, pk2, ...) then hash(...) bound expression for the given LogicalGet's PKs.
static unique_ptr<Expression> BuildXorHashFromPKs(OptimizerExtensionInput &input, LogicalGet &get,
                                                  const std::vector<std::string> &pks) {
	vector<unique_ptr<Expression>> pk_cols;
	for (auto &pk : pks) {
		// find the binding for this pk in the LogicalGet projection (use centralized helper)
		idx_t proj_idx = EnsureProjectedColumn(get, pk);
		if (proj_idx == DConstants::INVALID_INDEX) {
			throw InternalException("PAC compiler: failed to find PK column " + pk);
		}
		// create BoundColumnRefExpression referencing table_index and proj_idx
		auto col_binding = ColumnBinding(get.table_index, proj_idx);
		// determine the column's logical type from the LogicalGet's column index
		auto col_index_obj = get.GetColumnIds()[proj_idx];
		auto &col_type = get.GetColumnType(col_index_obj);
		pk_cols.push_back(make_uniq<BoundColumnRefExpression>(col_type, col_binding));
	}

	// If there is only one PK, no XOR needed
	unique_ptr<Expression> xor_expr;
	if (pk_cols.size() == 1) {
		xor_expr = std::move(pk_cols[0]);
	} else {
		// Build chain using optimizer.BindScalarFunction with operator "^" (bitwise XOR)
		auto left = std::move(pk_cols[0]);
		for (size_t i = 1; i < pk_cols.size(); ++i) {
			auto right = std::move(pk_cols[i]);
			// use the public two-arg BindScalarFunction
			left = input.optimizer.BindScalarFunction("^", std::move(left), std::move(right));
		}
		xor_expr = std::move(left);
	}

	// Finally bind the hash over the xor_expr
	return input.optimizer.BindScalarFunction("hash", std::move(xor_expr));
}

// New helper: find the unique_ptr reference to a LogicalGet node by table name.
// Returns a pointer to the owning unique_ptr so callers can replace/mutate it in-place.
// Also optionally returns information about the parent join if the node is part of a join.
static unique_ptr<LogicalOperator> *FindNodeRefByTable(unique_ptr<LogicalOperator> *root, const std::string &table_name,
                                                       LogicalOperator **parent_out = nullptr,
                                                       idx_t *child_idx_out = nullptr) {
	if (!root || !root->get()) {
		return nullptr;
	}
	LogicalOperator *node = root->get();
	if (node->type == LogicalOperatorType::LOGICAL_GET) {
		auto &g = node->Cast<LogicalGet>();
		auto tblptr = g.GetTable();
		if (tblptr && tblptr->name == table_name) {
			return root;
		}
	}
	for (idx_t i = 0; i < node->children.size(); i++) {
		auto res = FindNodeRefByTable(&node->children[i], table_name, parent_out, child_idx_out);
		if (res) {
			if (parent_out && !*parent_out) {
				*parent_out = node;
				if (child_idx_out) {
					*child_idx_out = i;
				}
			}
			return res;
		}
	}
	return nullptr;
}

unique_ptr<LogicalOperator> CreateLogicalJoin(const PACCompatibilityResult &check, ClientContext &context,
                                              unique_ptr<LogicalOperator> left_operator, unique_ptr<LogicalGet> right) {
	// Simpler join builder: use precomputed metadata only. Find FK(s) on one side that reference
	// the other's table; pair FK columns to PK columns and produce equality JoinCondition(s).

	// The left logical operator can be a GET or another JOIN
	LogicalGet *left = nullptr;
	if (left_operator->type == LogicalOperatorType::LOGICAL_GET) {
		left = &left_operator->Cast<LogicalGet>();
	} else if (left_operator->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		// We extract the table from the right side of the join
		if (left_operator->children.size() != 2 ||
		    left_operator->children[1]->type != LogicalOperatorType::LOGICAL_GET) {
			throw InternalException("PAC compiler: expected right child of left join to be LogicalGet");
		}
		left = &left_operator->children[1]->Cast<LogicalGet>();
	} else {
		throw InternalException("PAC compiler: expected left node to be LogicalGet or LogicalComparisonJoin");
	}

	auto left_table_ptr = left->GetTable();
	auto right_table_ptr = right->GetTable();
	if (!left_table_ptr || !right_table_ptr) {
		throw InternalException("PAC compiler: expected both LogicalGet nodes to be bound to tables for join creation");
	}
	std::string left_table_name = left_table_ptr->name;
	std::string right_table_name = right_table_ptr->name;

	// Require metadata to be present; compatibility check is responsible for populating it.
	auto lit = check.table_metadata.find(left_table_name);
	auto rit = check.table_metadata.find(right_table_name);
	if (lit == check.table_metadata.end() || rit == check.table_metadata.end()) {
		throw InternalException("PAC compiler: missing table metadata for join: " + left_table_name + " <-> " +
		                        right_table_name);
	}
	const auto &left_meta = lit->second;
	const auto &right_meta = rit->second;

	// Helper: ensure a column is projected in a LogicalGet and return its projection index
	auto ensure_proj_idx = [&](LogicalGet *g, const std::string &col_name) -> idx_t {
		if (!g) {
			return DConstants::INVALID_INDEX;
		}
		return EnsureProjectedColumn(*g, col_name);
	};

	vector<JoinCondition> conditions;

	// Try: left has FK referencing right
	for (auto &fk : left_meta.fks) {
		if (fk.first == right_table_name) {
			// fk.second: FK column names on left
			const auto &left_fk_cols = fk.second;
			const auto &right_pks = right_meta.pks;
			if (right_pks.size() != left_fk_cols.size() || right_pks.empty()) {
				throw InvalidInputException("PAC compiler: FK/PK column count mismatch for " + left_table_name +
				                            " -> " + right_table_name);
			}
			for (size_t i = 0; i < left_fk_cols.size(); ++i) {
				idx_t lproj = ensure_proj_idx(left, left_fk_cols[i]);
				idx_t rproj = ensure_proj_idx(right.get(), right_pks[i]);
				if (lproj == DConstants::INVALID_INDEX || rproj == DConstants::INVALID_INDEX) {
					throw InternalException("PAC compiler: failed to project FK/PK columns for join");
				}
				JoinCondition cond;
				cond.comparison = ExpressionType::COMPARE_EQUAL;
				auto left_col_index = left->GetColumnIds()[lproj];
				auto right_col_index = right->GetColumnIds()[rproj];
				auto &left_type = left->GetColumnType(left_col_index);
				auto &right_type = right->GetColumnType(right_col_index);
				cond.left = make_uniq<BoundColumnRefExpression>(left_type, ColumnBinding(left->table_index, lproj));
				cond.right = make_uniq<BoundColumnRefExpression>(right_type, ColumnBinding(right->table_index, rproj));
				conditions.push_back(std::move(cond));
			}
			break;
		}
	}

	// If no condition yet, try: right has FK referencing left
	if (conditions.empty()) {
		for (auto &fk : right_meta.fks) {
			if (fk.first == left_table_name) {
				const auto &right_fk_cols = fk.second;
				const auto &left_pks = left_meta.pks;
				if (left_pks.size() != right_fk_cols.size() || left_pks.empty()) {
					throw InvalidInputException("PAC compiler: FK/PK column count mismatch for " + right_table_name +
					                            " -> " + left_table_name);
				}
				for (size_t i = 0; i < right_fk_cols.size(); ++i) {
					// pair left PK with right FK
					idx_t lproj = ensure_proj_idx(left, left_pks[i]);
					idx_t rproj = ensure_proj_idx(right.get(), right_fk_cols[i]);
					if (lproj == DConstants::INVALID_INDEX || rproj == DConstants::INVALID_INDEX) {
						throw InternalException("PAC compiler: failed to project FK/PK columns for join");
					}
					JoinCondition cond;
					cond.comparison = ExpressionType::COMPARE_EQUAL;
					auto left_col_index = left->GetColumnIds()[lproj];
					auto right_col_index = right->GetColumnIds()[rproj];
					auto &left_type = left->GetColumnType(left_col_index);
					auto &right_type = right->GetColumnType(right_col_index);
					cond.left = make_uniq<BoundColumnRefExpression>(left_type, ColumnBinding(left->table_index, lproj));
					cond.right =
					    make_uniq<BoundColumnRefExpression>(right_type, ColumnBinding(right->table_index, rproj));
					conditions.push_back(std::move(cond));
				}
				break;
			}
		}
	}

	if (conditions.empty()) {
		throw InvalidInputException("PAC compiler: expected FK link between " + left_table_name + " and " +
		                            right_table_name);
	}

	vector<unique_ptr<Expression>> extra;
	return LogicalComparisonJoin::CreateJoin(context, JoinType::INNER, JoinRefType::REGULAR, std::move(left_operator),
	                                         std::move(right), std::move(conditions), std::move(extra));
}

unique_ptr<LogicalGet> CreateLogicalGet(ClientContext &context, unique_ptr<LogicalOperator> &plan, const string &table,
                                        idx_t idx) {

	Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
	CatalogSearchPath path(context);

	for (auto &schema : path.Get()) {
		auto entry =
		    catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema.schema, table, OnEntryNotFound::RETURN_NULL);
		if (!entry) {
			continue;
		}

		auto &table_entry = entry->Cast<TableCatalogEntry>();
		vector<LogicalType> types = table_entry.GetTypes();
		unique_ptr<FunctionData> bind_data;
		auto scan_function = table_entry.GetScanFunction(context, bind_data);
		vector<LogicalType> return_types = {};
		vector<string> return_names = {};
		vector<ColumnIndex> column_ids = {};
		vector<idx_t> projection_ids = {};
		for (auto &col : table_entry.GetColumns().Logical()) {
			return_types.push_back(col.Type());
			return_names.push_back(col.Name());
			column_ids.push_back(ColumnIndex(col.Oid()));
			projection_ids.push_back(column_ids.size() - 1);
		}

		unique_ptr<LogicalGet> get = make_uniq<LogicalGet>(idx, scan_function, std::move(bind_data),
		                                                   std::move(return_types), std::move(return_names));
		get->SetColumnIds(std::move(column_ids));
		get->projection_ids = projection_ids; // we project everything
		get->ResolveOperatorTypes();
		get->Verify(context);

		return get;
	}

	throw ParserException("PAC: missing internal sample table " + table);
}

// Helper: examine PACCompatibilityResult.fk_paths and populate gets_present / gets_missing.
// Also returns the start table and canonical target privacy unit via reference output parameters.
void PopulateGetsFromFKPath(const PACCompatibilityResult &check, std::vector<std::string> &gets_present,
                            std::vector<std::string> &gets_missing, std::string &start_table_out,
                            std::string &target_pu_out) {
	// Expect at least one FK path when this is called
	if (check.fk_paths.empty()) {
		throw InternalException("PAC compiler: no fk_paths available");
	}
	auto it = check.fk_paths.begin();
	start_table_out = it->first; // scanned table name
	auto fk_path = it->second;   // vector from start -> ... -> privacy_unit
	if (fk_path.empty()) {
		throw InternalException("PAC compiler: FK path is empty");
	}
	// canonical target privacy unit (last element)
	target_pu_out = fk_path.back();
	// For each table in the FK path, check whether a GET is already present
	for (auto &table_in_path : fk_path) {
		bool found_get = false;
		for (auto &t : check.scanned_pu_tables) {
			if (t == table_in_path) {
				gets_present.push_back(t);
				found_get = true;
				break;
			}
		}
		if (!found_get) {
			for (auto &t : check.scanned_non_pu_tables) {
				if (t == table_in_path) {
					gets_present.push_back(t);
					found_get = true;
					break;
				}
			}
		}
		if (!found_get) {
			gets_missing.push_back(table_in_path);
		}
	}
}

string GetPacAggregateFunctionName(const string &function_name) {
	string pac_function_name;
	if (function_name == "sum" || function_name == "sum_no_overflow") {
		pac_function_name = "pac_sum";
	} else if (function_name == "count" || function_name == "count_star") {
		pac_function_name = "pac_count";
	} else if (function_name == "avg") {
		pac_function_name = "pac_avg";
	} else if (function_name == "min") {
		pac_function_name = "pac_min";
	} else if (function_name == "max") {
		pac_function_name = "pac_max";
	} else {
		throw NotImplementedException("PAC compiler: unsupported aggregate function " + function_name);
	}
	return pac_function_name;
}

void ModifyPlanWithoutPU(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                         unique_ptr<LogicalOperator> &plan, const std::vector<std::string> &gets_missing,
                         const std::vector<std::string> &gets_present, const std::vector<std::string> &fk_path,
                         const std::string &privacy_unit) {
	// Note: we assume we don't use rowid

	// Check if join elimination is enabled
	Value join_elim_val;
	bool join_elimination = false;
	if (input.context.TryGetCurrentSetting("pac_join_elimination", join_elim_val) && !join_elim_val.IsNull()) {
		join_elimination = join_elim_val.GetValue<bool>();
	}

#ifdef DEBUG
	Printer::Print("ModifyPlanWithoutPU: join_elimination = " + std::to_string(join_elimination));
	Printer::Print("ModifyPlanWithoutPU: privacy_unit = " + privacy_unit);
#endif

	// Create the necessary LogicalGets for missing tables
	std::unordered_map<std::string, unique_ptr<LogicalGet>> get_map;
	// Preserve creation order: store ordered table names (ownership kept in get_map)
	vector<string> ordered_table_names;
	auto idx = GetNextTableIndex(plan);
	for (auto &table : gets_missing) {
		// If join elimination is enabled, skip the PU table
		if (join_elimination && table == privacy_unit) {
#ifdef DEBUG
			Printer::Print("ModifyPlanWithoutPU: Skipping PU table " + table + " due to join elimination");
#endif
			continue;
		}

		auto it = check.table_metadata.find(table);
		if (it == check.table_metadata.end()) {
			throw InternalException("PAC compiler: missing table metadata for missing GET: " + table);
		}
		std::vector<std::string> pks = it->second.pks;
		auto get = CreateLogicalGet(input.context, plan, table, idx);
		// store in map (ownership) and record name to preserve order
		get_map[table] = std::move(get);
		ordered_table_names.push_back(table);
#ifdef DEBUG
		Printer::Print("ModifyPlanWithoutPU: Added table " + table + " to join chain");
#endif
		idx++;
	}

	// Find the unique_ptr reference to the existing table that connects to the missing tables
	// We need to find the last present table in the FK path, which is the connection point
	LogicalOperator *parent_join = nullptr;
	idx_t child_idx_in_parent = 0;
	unique_ptr<LogicalOperator> *target_ref = nullptr;

	// Find the last present table in the FK path (ordered)
	// This is the table that connects to the first missing table
	string connecting_table;
	if (!fk_path.empty()) {
		// Go through the FK path and find the last table that's in gets_present
		for (auto &table_in_path : fk_path) {
			bool is_present = false;
			for (auto &present : gets_present) {
				if (table_in_path == present) {
					is_present = true;
					connecting_table = table_in_path;
					break;
				}
			}
			// Once we hit a missing table, we've found our connection point
			if (!is_present) {
				break;
			}
		}
	}

	// If we couldn't find a connecting table, fall back to the first present table
	if (connecting_table.empty() && !gets_present.empty()) {
		connecting_table = gets_present[0];
	}

	target_ref = FindNodeRefByTable(&plan, connecting_table, &parent_join, &child_idx_in_parent);
	if (!target_ref) {
		throw InternalException("PAC compiler: could not find existing LogicalGet for table " +
		                        (connecting_table.empty() ? std::string("<none>") : connecting_table));
	}

	// Check if the target node is part of a LEFT JOIN
	bool is_left_join = false;
	JoinType original_join_type = JoinType::INNER;
	if (parent_join && parent_join->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &parent_join_op = parent_join->Cast<LogicalComparisonJoin>();
		original_join_type = parent_join_op.join_type;
		is_left_join = (original_join_type == JoinType::LEFT);
	}

	// We create the joins in this order:
	// the existing node joins with ordered_table_names[0], then this join node joins with ordered_table_names[1], ...
	// All intermediate joins are INNER joins to reach the privacy unit
	unique_ptr<LogicalOperator> final_join;
	unique_ptr<LogicalOperator> existing_node = (*target_ref)->Copy(input.context);

	for (size_t i = 0; i < ordered_table_names.size(); ++i) {
		auto &tbl_name = ordered_table_names[i];
		// move the LogicalGet from get_map into a unique_ptr<LogicalGet>
		unique_ptr<LogicalGet> right_op = std::move(get_map[tbl_name]);
		if (!right_op) {
			throw InternalException("PAC compiler: failed to transfer ownership of LogicalGet for " + tbl_name);
		}

		if (i == 0) {
			auto join = CreateLogicalJoin(check, input.context, std::move(existing_node), std::move(right_op));
			final_join = std::move(join);
		} else {
			auto join = CreateLogicalJoin(check, input.context, std::move(final_join), std::move(right_op));
			final_join = std::move(join);
		}
	}

	// Use ReplaceNode to insert the join and handle column binding remapping
	ReplaceNode(plan, *target_ref, final_join);

	// If the original was a LEFT JOIN, we need to restore it at the top level
	// The INNER joins to reach the PU are now inside the LEFT JOIN structure
	if (is_left_join && parent_join && parent_join->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &parent_join_op = parent_join->Cast<LogicalComparisonJoin>();
		// The join type should already be LEFT, but make sure the structure is correct
		// The INNER join chain is now in place of the original child
		parent_join_op.join_type = original_join_type;
	}

#if DEBUG
	plan->Print();

#endif

	// When join elimination is enabled, we don't join to PU table
	// Instead, we use the FK column from the last table in the chain
	unique_ptr<Expression> hash_input_expr;

	if (join_elimination) {
		// Find the last table in the FK chain (the one that would link to PU)
		// This is the last table in ordered_table_names, or if empty, the connecting_table
		string last_table_name;
		if (!ordered_table_names.empty()) {
			last_table_name = ordered_table_names.back();
		} else {
			last_table_name = connecting_table;
		}

		// Find the LogicalGet for this table
		unique_ptr<LogicalOperator> *last_table_ref = FindNodeRefByTable(&plan, last_table_name);
		if (!last_table_ref || !last_table_ref->get()) {
			throw InternalException("PAC compiler: could not find LogicalGet for last table " + last_table_name);
		}
		auto last_get = &last_table_ref->get()->Cast<LogicalGet>();

		// Find the FK column(s) that reference the PU
		auto it = check.table_metadata.find(last_table_name);
		if (it == check.table_metadata.end()) {
			throw InternalException("PAC compiler: missing metadata for table " + last_table_name);
		}

		vector<string> fk_cols;
		for (auto &fk : it->second.fks) {
			if (fk.first == privacy_unit) {
				fk_cols = fk.second;
				break;
			}
		}

		if (fk_cols.empty()) {
			throw InternalException("PAC compiler: no FK found from " + last_table_name + " to " + privacy_unit);
		}

		// Build hash from FK columns
		hash_input_expr = BuildXorHashFromPKs(input, *last_get, fk_cols);
	} else {
		// Original behavior: locate the privacy-unit LogicalGet we just inserted
		unique_ptr<LogicalOperator> *pu_ref = nullptr;
		if (!privacy_unit.empty()) {
			pu_ref = FindNodeRefByTable(&plan, privacy_unit);
		}
		if (!pu_ref || !pu_ref->get()) {
			throw InternalException("PAC compiler: could not find LogicalGet for privacy unit " + privacy_unit);
		}
		auto pu_get = &pu_ref->get()->Cast<LogicalGet>();

		vector<string> pu_pks;
		for (auto &pk : check.table_metadata.at(privacy_unit).pks) {
			pu_pks.push_back(pk);
		}

		// Build the hash expression from PU PKs
		hash_input_expr = BuildXorHashFromPKs(input, *pu_get, pu_pks);
	}

	// Now we need to edit the aggregate node to use pac functions
	auto *agg = FindTopAggregate(plan);

	FunctionBinder function_binder(input.context);

	// Process each aggregate expression
	for (idx_t i = 0; i < agg->expressions.size(); i++) {
		if (agg->expressions[i]->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			throw NotImplementedException("Not found expected aggregate expression in PAC compiler");
		}

		auto &old_aggr = agg->expressions[i]->Cast<BoundAggregateExpression>();
		string function_name = old_aggr.function.name;

		// Extract the original aggregate's value child expression (e.g., the `val` in SUM(val))
		// COUNT(*) has no children, so we create a constant 1 expression when there's no child
		unique_ptr<Expression> value_child;
		if (old_aggr.children.empty()) {
			// COUNT(*) case - create a constant 1
			if (function_name == "count_star" || function_name == "count") {
				value_child = make_uniq_base<Expression, BoundConstantExpression>(Value::BIGINT(1));
			} else {
				throw InternalException("PAC compiler: expected aggregate to have a child expression");
			}
		} else {
			value_child = old_aggr.children[0]->Copy();
		}

		// Get PAC function name
		string pac_function_name = GetPacAggregateFunctionName(function_name);

		// Bind the PAC aggregate function
		ErrorData error;
		vector<LogicalType> arg_types;
		arg_types.push_back(hash_input_expr->return_type);
		arg_types.push_back(value_child->return_type);

		auto &entry = Catalog::GetSystemCatalog(input.context)
		                  .GetEntry<AggregateFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, pac_function_name);
		auto &aggr_catalog = entry.Cast<AggregateFunctionCatalogEntry>();

		auto best = function_binder.BindFunction(aggr_catalog.name, aggr_catalog.functions, arg_types, error);
		if (!best.IsValid()) {
			throw InternalException("PAC compiler: failed to bind pac aggregate for given argument types");
		}
		auto bound_aggr_func = aggr_catalog.functions.GetFunctionByOffset(best.GetIndex());

		// Build arguments for this aggregate (copy hash expression for each aggregate)
		vector<unique_ptr<Expression>> aggr_children;
		aggr_children.push_back(hash_input_expr->Copy());
		aggr_children.push_back(std::move(value_child));

		// Pass through the DISTINCT flag from the original aggregate
		AggregateType agg_type = old_aggr.IsDistinct() ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT;

		auto new_aggr =
		    function_binder.BindAggregateFunction(bound_aggr_func, std::move(aggr_children), nullptr, agg_type);

		agg->expressions[i] = std::move(new_aggr);
	}

	agg->ResolveOperatorTypes();
}

void ModifyPlanWithPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                      const std::vector<std::string> &pks, bool use_rowid, const std::string &pu_table_name) {

	auto pu_scan_ptr = FindPrivacyUnitGetNode(plan, pu_table_name);
	auto &get = pu_scan_ptr->get()->Cast<LogicalGet>();
	if (use_rowid) {
		AddRowIDColumn(get);
	} else {
		// Ensure primary key columns are present in the LogicalGet (add them if necessary)
		AddPKColumns(get, pks);
	}

	// Now we need to edit the aggregate node to use pac functions
	auto *agg = FindTopAggregate(plan);

	// We create a hash expression over either the row id or the XOR of PK columns
	// Build the hash expression once (will be copied/reused for each aggregate)
	unique_ptr<Expression> hash_input_expr;
	if (use_rowid) {
		// rowid is the last column added
		auto rowid_binding = ColumnBinding(get.table_index, get.GetColumnIds().size() - 1);
		auto rowid_col = make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, rowid_binding);
		auto bound_hash = input.optimizer.BindScalarFunction("hash", std::move(rowid_col));
		hash_input_expr = std::move(bound_hash);
	} else {
		// Replaced duplicated inline logic with helper
		hash_input_expr = BuildXorHashFromPKs(input, get, pks);
	}

	FunctionBinder function_binder(input.context);

	// Process each aggregate expression
	for (idx_t i = 0; i < agg->expressions.size(); i++) {
		if (agg->expressions[i]->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			throw NotImplementedException("Not found expected aggregate expression in PAC compiler");
		}

		auto &old_aggr = agg->expressions[i]->Cast<BoundAggregateExpression>();
		string function_name = old_aggr.function.name;

		// Extract the original aggregate's value child expression (e.g., the `val` in SUM(val))
		// COUNT(*) has no children, so we create a constant 1 expression when there's no child
		unique_ptr<Expression> value_child;
		if (old_aggr.children.empty()) {
			// COUNT(*) case - create a constant 1
			if (function_name == "count_star" || function_name == "count") {
				value_child = make_uniq_base<Expression, BoundConstantExpression>(Value::BIGINT(1));
			} else {
				throw InternalException("PAC compiler: expected aggregate to have a child expression");
			}
		} else {
			value_child = old_aggr.children[0]->Copy();
		}

		// Get PAC function name
		string pac_function_name = GetPacAggregateFunctionName(function_name);

		// Bind the PAC aggregate function
		ErrorData error;
		vector<LogicalType> arg_types;
		arg_types.push_back(hash_input_expr->return_type);
		arg_types.push_back(value_child->return_type);

		auto &entry = Catalog::GetSystemCatalog(input.context)
		                  .GetEntry<AggregateFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, pac_function_name);
		auto &aggr_catalog = entry.Cast<AggregateFunctionCatalogEntry>();

		auto best = function_binder.BindFunction(aggr_catalog.name, aggr_catalog.functions, arg_types, error);
		if (!best.IsValid()) {
			throw InternalException("PAC compiler: failed to bind pac aggregate for given argument types");
		}
		auto bound_aggr_func = aggr_catalog.functions.GetFunctionByOffset(best.GetIndex());

		vector<unique_ptr<Expression>> aggr_children;
		aggr_children.push_back(hash_input_expr->Copy());
		aggr_children.push_back(std::move(value_child));

		// Pass through the DISTINCT flag from the original aggregate
		AggregateType agg_type = old_aggr.IsDistinct() ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT;

		auto new_aggr =
		    function_binder.BindAggregateFunction(bound_aggr_func, std::move(aggr_children), nullptr, agg_type);

		agg->expressions[i] = std::move(new_aggr);
	}

	agg->ResolveOperatorTypes();
}

void CompilePacBitsliceQuery(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                             unique_ptr<LogicalOperator> &plan, const std::string &privacy_unit,
                             const std::string &query, const std::string &query_hash) {

#ifdef DEBUG
	Printer::Print("CompilePacBitsliceQuery called for PU=" + privacy_unit + " hash=" + query_hash);
#endif

	string path = GetPacCompiledPath(input.context, ".");
	if (!path.empty() && path.back() != '/') {
		path.push_back('/');
	}
	string filename = path + privacy_unit + "_" + query_hash + "_bitslice.sql";

	// The bitslice compiler works in the following way:
	// a) the query scans PU table:
	// a.1) the PU table has 1 PK: we hash it
	// a.2) the PU table has multiple PKs: we XOR them and hash the result
	// a.3) the PU table has no PK: we hash rowid
	// b) the query does not scan PU table:
	// b.1) we follow the FK path to find the PK(s) of the PU table
	// b.2) we join the chain of tables from the scanned table to the PU table
	// b.3) we hash the PK(s) as in a)

	// Example: SELECT group_key, SUM(val) AS sum_val FROM t_single GROUP BY group_key;
	// Becomes: SELECT group_key, pac_sum(HASH(rowid), val) AS sum_val FROM t_single GROUP BY group_key;
	// todo- what is MI? what is k?

	bool pu_present_in_tree = false;
	bool use_rowid = false;

	if (!check.scanned_pu_tables.empty()) {
		pu_present_in_tree = true;
	}

	// Case a) query scans PU table (we assume only 1 PAC table for now)
	std::vector<string> pks;
	// Build two vectors: present (GETs already in the plan) and missing (GETs to create)
	std::vector<std::string> gets_present;
	std::vector<std::string> gets_missing;
	std::vector<LogicalGet *> new_gets;

	// fk_path_to_use will be populated when we detect an FK path (used for multi-hop joins)
	std::vector<std::string> fk_path_to_use;

	if (pu_present_in_tree) {
		// Look up PKs for the scanned PAC table via table_metadata (filled by the compatibility check)
		if (!check.scanned_pu_tables.empty()) {
			auto tbl = check.scanned_pu_tables[0];
			auto it = check.table_metadata.find(tbl);
			if (it != check.table_metadata.end() && !it->second.pks.empty()) {
				pks = it->second.pks;
			} else {
				// no PKs found -> use rowid
				use_rowid = true;
			}
		} else {
			use_rowid = true;
		}
	} else if (!check.fk_paths.empty()) {
		// The query does not scan the PU table: we need to follow the FK path
		string start_table;
		string target_pu;
		PopulateGetsFromFKPath(check, gets_present, gets_missing, start_table, target_pu);

		// extract the ordered fk_path from the compatibility result
		auto it = check.fk_paths.find(start_table);
		if (it == check.fk_paths.end() || it->second.empty()) {
			throw InternalException("PAC compiler: expected fk_path for start table " + start_table);
		}
		fk_path_to_use = it->second;

#ifdef DEBUG
		Printer::Print("PAC bitslice: FK path detection");
		Printer::Print("start_table: " + start_table);
		Printer::Print("target_pu: " + target_pu);
		Printer::Print("fk_path:");
		for (auto &p : fk_path_to_use) {
			Printer::Print("  " + p);
		}
		Printer::Print("gets_present:");
		for (auto &g : gets_present) {
			Printer::Print("  " + g);
		}
		Printer::Print("gets_missing:");
		for (auto &g : gets_missing) {
			Printer::Print("  " + g);
		}
#endif
		// fk_path_to_use is populated and will be passed into ModifyPlanWithoutPU below
	}

	// Now we need to find the PAC scan node
	// Replan the plan without compressed materialization
	ReplanWithoutOptimizers(input.context, query, plan);

	if (pu_present_in_tree) {
		ModifyPlanWithPU(input, plan, pks, use_rowid, check.scanned_pu_tables[0]);
	} else {
		ModifyPlanWithoutPU(check, input, plan, gets_missing, gets_present, fk_path_to_use, privacy_unit);
	}

	plan->ResolveOperatorTypes();
	plan->Verify(input.context);
#ifdef DEBUG
	plan->Print();
#endif
}

} // namespace duckdb
