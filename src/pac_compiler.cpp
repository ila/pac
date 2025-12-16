//
// Created by ila on 12/12/25.
// Clean PAC compiler implementation
//

#include "include/pac_compiler.hpp"
#include "include/pac_privacy_unit.hpp"
#include "include/pac_helpers.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/common/constants.hpp"

#include <fstream>
#include <cctype>

namespace duckdb {

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

static LogicalAggregate *FindTopAggregate(unique_ptr<LogicalOperator> &op) {
    if (!op) {
        return nullptr;
    }
    if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
        return &op->Cast<LogicalAggregate>();
    }
    for (auto &child : op->children) {
        if (auto *agg = FindTopAggregate(child)) {
            return agg;
        }
    }
    return nullptr;
}

static LogicalProjection *FindTopProjection(unique_ptr<LogicalOperator> &op) {
    if (!op) {
        return nullptr;
    }
    if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
        return &op->Cast<LogicalProjection>();
    }
    for (auto &child : op->children) {
        if (auto *proj = FindTopProjection(child)) {
            return proj;
        }
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// Sample CTE emission
// -----------------------------------------------------------------------------

void CreateSampleCTE(ClientContext &context,
                     const std::string &privacy_unit,
                     const std::string &filename,
                     const std::string &query_normalized) {
    int64_t m_cfg = 128;
    Value m_val;
    if (context.TryGetCurrentSetting("pac_m", m_val) && !m_val.IsNull()) {
        m_cfg = MaxValue<int64_t>(1, m_val.GetValue<int64_t>());
    }

    std::ofstream ofs(filename);
    if (!ofs) {
        throw ParserException("PAC: failed to write " + filename);
    }

    ofs << "-- PAC compiled sample CTE\n";
    ofs << "-- privacy unit: " << privacy_unit << "\n";
    ofs << "-- normalized query: " << query_normalized << "\n\n";

    ofs << "WITH pac_sample AS (\n";
    ofs << "  SELECT src.rowid, s.sample_id\n";
    ofs << "  FROM " << privacy_unit << " AS src\n";
    ofs << "  CROSS JOIN generate_series(1," << m_cfg << ") AS s(sample_id)\n";
    ofs << ")\n";

    ofs.close();
}

// -----------------------------------------------------------------------------
// pac_sample LogicalGet
// -----------------------------------------------------------------------------
unique_ptr<LogicalGet>
CreatePacSampleLogicalGet(ClientContext &context,
                          idx_t table_index,
                          const string &privacy_unit) {
    string name = "_pac_internal_sample_" + Sanitize(privacy_unit);

    Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
    CatalogSearchPath path(context);

    for (auto &schema : path.Get()) {
        auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY,
                                      schema.schema, name,
                                      OnEntryNotFound::RETURN_NULL);
        if (!entry) {
            continue;
        }

    	auto &table_entry = entry->Cast<TableCatalogEntry>();
        vector<LogicalType> types = {LogicalType::BIGINT, LogicalType::BIGINT};
    	unique_ptr<FunctionData> bind_data;
    	auto scan_function = table_entry.GetScanFunction(context, bind_data);
    	vector<LogicalType> return_types = {};
    	vector<string> return_names = {};
    	vector<ColumnIndex> column_ids = {};
    	for (auto &col : table_entry.GetColumns().Logical()) {
    		return_types.push_back(col.Type());
    		return_names.push_back(col.Name());
    		column_ids.push_back(ColumnIndex(col.Oid()));
    	}

		unique_ptr<LogicalGet> sample_get = make_uniq<LogicalGet>(table_index, scan_function, std::move(bind_data),
		                                                          std::move(return_types), std::move(return_names));
    	sample_get->SetColumnIds(std::move(column_ids));
    	sample_get->projection_ids = {0, 1}; // project all columns: rowid, sample_id
    	sample_get->ResolveOperatorTypes();
    	sample_get->Verify(context);

        return sample_get;
    }

    throw ParserException("PAC: missing internal sample table " + name);
}

// -----------------------------------------------------------------------------
// Join construction
// -----------------------------------------------------------------------------

unique_ptr<LogicalOperator>
CreatePacSampleJoinNode(ClientContext &context,
                        unique_ptr<LogicalOperator> base,
                        unique_ptr<LogicalGet> pac_get,
                        idx_t base_idx,
                        idx_t pac_idx) {
    JoinCondition cond;
    cond.comparison = ExpressionType::COMPARE_EQUAL;
    cond.left = make_uniq<BoundColumnRefExpression>(
        LogicalType::BIGINT, ColumnBinding(base_idx, COLUMN_IDENTIFIER_ROW_ID));
    cond.right = make_uniq<BoundColumnRefExpression>(
        LogicalType::BIGINT, ColumnBinding(pac_idx, 0));

    vector<JoinCondition> conditions;
    conditions.push_back(std::move(cond));

    vector<unique_ptr<Expression>> extra;
    return LogicalComparisonJoin::CreateJoin(
        context, JoinType::INNER, JoinRefType::REGULAR,
        std::move(base), std::move(pac_get),
        std::move(conditions), std::move(extra));
}

// -----------------------------------------------------------------------------
// Plan rewrite
// -----------------------------------------------------------------------------

static idx_t
InsertPacJoinBelowAggregate(ClientContext &context,
                            unique_ptr<LogicalOperator> &root,
                            unique_ptr<LogicalGet> pac_get,
                            idx_t pac_idx) {
    auto *agg = FindTopAggregate(root);
    unique_ptr<LogicalOperator> *target = nullptr;

    if (agg) {
        D_ASSERT(!agg->children.empty());
        target = &agg->children[0];
    } else {
        target = &root;
    }

    idx_t base_idx = DConstants::INVALID_INDEX;
    vector<unique_ptr<LogicalOperator> *> stack;
    stack.push_back(target);
    while (!stack.empty() && base_idx == DConstants::INVALID_INDEX) {
        auto cur_ptr = stack.back();
        stack.pop_back();
        auto &cur = *cur_ptr;
        if (!cur) continue;
        if (cur->type == LogicalOperatorType::LOGICAL_GET) {
            base_idx = cur->Cast<LogicalGet>().table_index;
            break;
        }
        for (auto &c : cur->children) {
            stack.push_back(&c);
        }
    }

    if (base_idx == DConstants::INVALID_INDEX) {
        return DConstants::INVALID_INDEX;
    }


    *target = CreatePacSampleJoinNode(
        context, std::move(*target), std::move(pac_get),
        base_idx, pac_idx);

    return pac_idx;
}

// -----------------------------------------------------------------------------
// Aggregate update
// -----------------------------------------------------------------------------

static void UpdateAggregateGroups(LogicalAggregate *agg, idx_t pac_idx) {
    if (!agg) return;
    agg->groups.push_back(
        make_uniq<BoundColumnRefExpression>(
            LogicalType::BIGINT,
            ColumnBinding(pac_idx, 1)));
}

static void UpdateProjection(LogicalProjection *proj, idx_t pac_idx) {
    if (!proj) return;
    proj->expressions.push_back(
        make_uniq<BoundColumnRefExpression>(
            LogicalType::BIGINT,
            ColumnBinding(pac_idx, 1)));
}


// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

void CompilePACQuery(OptimizerExtensionInput &input,
                     unique_ptr<LogicalOperator> &plan,
                     const std::string &privacy_unit) {
    if (privacy_unit.empty()) {
        return;
    }

    string normalized = NormalizeQueryForHash(input.context.GetCurrentQuery());
    string hash = HashStringToHex(normalized);

    string path = ".";
    Value v;
    if (input.context.TryGetCurrentSetting("pac_compiled_path", v) && !v.IsNull()) {
        path = v.ToString();
    }
    if (!path.empty() && path.back() != '/') {
        path.push_back('/');
    }

    string filename = path + privacy_unit + "_" + hash + ".sql";
    CreateSampleCTE(input.context, privacy_unit, filename, normalized);

    auto *agg = FindTopAggregate(plan);
    unique_ptr<LogicalOperator> *target = nullptr;
    if (agg) {
        D_ASSERT(!agg->children.empty());
        target = &agg->children[0];
    } else {
        target = &plan;
    }
    idx_t pac_idx = GetNextTableIndex(*target);
    auto pac_get = CreatePacSampleLogicalGet(input.context, pac_idx, privacy_unit);
	// todo from here
    InsertPacJoinBelowAggregate(input.context, plan, std::move(pac_get), pac_idx);
    UpdateAggregateGroups(agg, pac_idx);
    auto *proj = FindTopProjection(plan);
    UpdateProjection(proj, pac_idx);
    plan->Verify(input.context);
	Printer::Print("done");
}

} // namespace duckdb
