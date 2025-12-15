#include "include/pac_privacy_unit.hpp"

#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/string_type.hpp"

#include <fstream>
#include <cstdio>

namespace duckdb {

std::unordered_set<std::string> ReadPacTablesFile(const std::string &filename) {
    std::unordered_set<std::string> tables;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        // StringUtil::Trim mutates the string in-place
        StringUtil::Trim(line);
        if (!line.empty()) {
            tables.insert(line);
        }
    }
    return tables;
}

void WritePacTablesFile(const std::string &filename, const std::unordered_set<std::string> &tables) {
    std::string tmp_filename = filename + ".tmp";
    {
        std::ofstream file(tmp_filename, std::ios::trunc);
        for (const auto &t : tables) {
            std::string s = t;
            StringUtil::Trim(s);
            file << s << "\n";
        }
        file.close();
    }
    // atomically replace
    ::remove(filename.c_str());
    ::rename(tmp_filename.c_str(), filename.c_str());
}

bool TableExists(ClientContext &context, const std::string &table_name) {
    // todo - support custom schemas
    Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
    CatalogSearchPath search_path(context);
    for (auto &schema_entry : search_path.Get()) {
        auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema_entry.schema, table_name,
                                     OnEntryNotFound::RETURN_NULL);
        if (entry) {
            return true;
        }
    }
    return false;
}

// Scalar helper to delete a privacy units file (for tests/cleanup)
void DeletePrivacyUnitFileFun(DataChunk &args, ExpressionState &state, Vector &result) {
    D_ASSERT(args.ColumnCount() == 1);
    if (args.data[0].GetType().id() != LogicalTypeId::VARCHAR) {
        throw BinderException("delete_privacy_unit_file: argument must be a string");
    }
    // extract first (and only) value
    auto val = args.data[0].GetValue(0);
    std::string path = val.ToString();
    int r = ::remove(path.c_str());
    std::string msg;
    if (r == 0) {
        msg = StringUtil::Format("Deleted PAC privacy units file: %s", path);
    } else {
        msg = StringUtil::Format("PAC privacy units file not found: %s", path);
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
    auto result_data = FlatVector::GetData<string_t>(result);
    idx_t count = args.size();
    for (idx_t i = 0; i < count; i++) {
        result_data[i] = StringVector::AddString(result, msg);
    }
    if (args.AllConstant()) {
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
    }
}

// Pragma-style API: PRAGMA add_privacy_unit('table', 'filename?')
void AddPrivacyUnitPragma(ClientContext &context, const FunctionParameters &parameters) {
    if (parameters.values.empty()) {
        throw InvalidInputException("add_privacy_unit pragma requires the table name");
    }
    auto table_name = parameters.values[0].ToString();
	string pac_privacy_file = "pac_tables.csv";
	Value pac_privacy_file_value;
	context.TryGetCurrentSetting("pac_privacy_file", pac_privacy_file_value);
	if (!pac_privacy_file_value.IsNull()) {
		// by default, the ivm files path is the database path
		// however this can be overridden by a setting
		pac_privacy_file = pac_privacy_file_value.ToString();
	}
    // Ensure the referenced table actually exists before touching the file
    if (!TableExists(context, table_name)) {
        throw InvalidInputException(StringUtil::Format("add_privacy_unit: table does not exist: %s", table_name));
    }
    auto tables = ReadPacTablesFile(pac_privacy_file);
    if (tables.count(table_name) == 0) {
        tables.insert(table_name);
        WritePacTablesFile(pac_privacy_file, tables);
    }
}

// Pragma-style API: PRAGMA remove_privacy_unit('table', 'filename?')
void RemovePrivacyUnitPragma(ClientContext &context, const FunctionParameters &parameters) {
    if (parameters.values.empty()) {
        throw InvalidInputException("remove_privacy_unit pragma requires at least the table name");
    }
    auto table_name = parameters.values[0].ToString();
	string pac_privacy_file = "pac_tables.csv";
	Value pac_privacy_file_value;
	context.TryGetCurrentSetting("pac_privacy_file", pac_privacy_file_value);
	if (!pac_privacy_file_value.IsNull()) {
		// by default, the ivm files path is the database path
		// however this can be overridden by a setting
		pac_privacy_file = pac_privacy_file_value.ToString();
	}
    if (parameters.values.size() >= 2) {
        pac_privacy_file = parameters.values[1].ToString();
    }
    // Do not pre-create the file - ReadPacTablesFile will handle a missing file as empty
    auto tables = ReadPacTablesFile(pac_privacy_file);
    if (tables.erase(table_name) > 0) {
        WritePacTablesFile(pac_privacy_file, tables);
    }
}

} // namespace duckdb
