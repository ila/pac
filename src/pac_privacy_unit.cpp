#include "include/pac_privacy_unit.hpp"

#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/types/vector.hpp"

#include <fstream>

using namespace duckdb;

namespace duckdb {

std::unordered_set<std::string> ReadPacTablesFile(const std::string &filename) {
    std::unordered_set<std::string> tables;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) tables.insert(line);
    }
    return tables;
}

void WritePacTablesFile(const std::string &filename, const std::unordered_set<std::string> &tables) {
    std::ofstream file(filename, std::ios::trunc);
    for (const auto &t : tables) file << t << "\n";
}

bool TableExists(ClientContext &context, const std::string &table_name) {
    Catalog &catalog = Catalog::GetSystemCatalog(context);
    CatalogSearchPath search_path(context);
    for (auto &schema_entry : search_path.GetSetPaths()) {
        auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema_entry.schema, table_name,
                                     OnEntryNotFound::RETURN_NULL);
        if (entry) {
            return true;
        }
    }
    return false;
}

// PacPrivacyUnitBindData implementation
PacPrivacyUnitBindData::PacPrivacyUnitBindData(std::string table_name) : table_name(std::move(table_name)) {}

unique_ptr<FunctionData> PacPrivacyUnitBindData::Copy() const {
    return make_uniq<PacPrivacyUnitBindData>(table_name);
}

bool PacPrivacyUnitBindData::Equals(const FunctionData &other) const {
    if (auto other_ptr = dynamic_cast<const PacPrivacyUnitBindData *>(&other)) {
        return table_name == other_ptr->table_name;
    }
    return false;
}

unique_ptr<FunctionData> AddPacPrivacyUnitBind(ClientContext &context, ScalarFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {
    if (arguments.empty() || !arguments[0]->IsFoldable()) {
        throw BinderException("add_pac_privacy_unit: argument must be a constant string");
    }
    auto &expr = (const BoundConstantExpression &)*arguments[0];
    std::string table_name = expr.value.ToString();
    if (!TableExists(context, table_name)) {
        throw BinderException("Table '%s' does not exist!", table_name.c_str());
    }
    return make_uniq<PacPrivacyUnitBindData>(table_name);
}

void AddPacPrivacyUnitFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
    auto &bind_data = func_expr.bind_info->Cast<PacPrivacyUnitBindData>();
    std::string table_name = bind_data.table_name;
    std::string filename = "pac_tables.csv";
    auto tables = ReadPacTablesFile(filename);
    if (tables.count(table_name) == 0) {
        tables.insert(table_name);
        WritePacTablesFile(filename, tables);
        StringVector::AddString(result, "Added PAC privacy unit: " + table_name);
    } else {
        StringVector::AddString(result, "PAC privacy unit already present: " + table_name);
    }
}

unique_ptr<FunctionData> RemovePacPrivacyUnitBind(ClientContext &context, ScalarFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
    if (arguments.empty() || !arguments[0]->IsFoldable()) {
        throw BinderException("remove_pac_privacy_unit: argument must be a constant string");
    }
    auto &expr = (const BoundConstantExpression &)*arguments[0];
    std::string table_name = expr.value.ToString();
    return make_uniq<PacPrivacyUnitBindData>(table_name);
}

void RemovePacPrivacyUnitFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
    auto &bind_data = func_expr.bind_info->Cast<PacPrivacyUnitBindData>();
    std::string table_name = bind_data.table_name;
    std::string filename = "pac_tables.csv";
    auto tables = ReadPacTablesFile(filename);
    if (tables.erase(table_name) > 0) {
        WritePacTablesFile(filename, tables);
        StringVector::AddString(result, "Removed PAC privacy unit: " + table_name);
    } else {
        StringVector::AddString(result, "PAC privacy unit not present: " + table_name);
    }
}

} // namespace duckdb

