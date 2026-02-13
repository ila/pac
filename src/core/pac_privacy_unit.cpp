#include "core/pac_privacy_unit.hpp"
#include "utils/pac_helpers.hpp"

#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/main/database_manager.hpp"

#include <fstream>
#include <cstdio>

namespace duckdb {

std::unordered_set<string> ReadPacTablesFile(const string &filename) {
	std::unordered_set<string> tables;
	std::ifstream file(filename);
	string line;
	while (std::getline(file, line)) {
		// StringUtil::Trim mutates the string in-place
		StringUtil::Trim(line);
		if (!line.empty()) {
			tables.insert(line);
		}
	}
	return tables;
}

void WritePacTablesFile(const string &filename, const std::unordered_set<string> &tables) {
	string tmp_filename = filename + ".tmp";
	{
		std::ofstream file(tmp_filename, std::ios::trunc);
		for (const auto &t : tables) {
			string s = t;
			StringUtil::Trim(s);
			file << s << "\n";
		}
		file.close();
	}
	// atomically replace
	::remove(filename.c_str());
	::rename(tmp_filename.c_str(), filename.c_str());
}

bool TableExists(ClientContext &context, const string &table_name) {
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
	string path = val.ToString();
	int r = ::remove(path.c_str());
	string msg;
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

} // namespace duckdb
