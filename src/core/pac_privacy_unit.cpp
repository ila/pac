#include "core/pac_privacy_unit.hpp"
#include "utils/pac_helpers.hpp"

#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/column_definition.hpp"
#include "duckdb/main/appender.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
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

// Pragma-style API: PRAGMA add_privacy_unit('table', 'filename?')
void AddPrivacyUnitPragma(ClientContext &context, const FunctionParameters &parameters) {
	if (parameters.values.empty()) {
		throw InvalidInputException("add_privacy_unit pragma requires the table name");
	}
	auto table_name = parameters.values[0].ToString();
	string pac_privacy_file = GetPacPrivacyFile(context);
	// Ensure the referenced table actually exists before touching the file
	if (!TableExists(context, table_name)) {
		throw InvalidInputException(StringUtil::Format("add_privacy_unit: table does not exist: %s", table_name));
	}
	auto tables = ReadPacTablesFile(pac_privacy_file);
	if (tables.count(table_name) == 0) {
		tables.insert(table_name);
		WritePacTablesFile(pac_privacy_file, tables);
	}

	// Create a small internal helper table for plan verification/explain.
	// Sanitize the table name into a safe identifier: replace non-alnum with '_'
	string sanitized = Sanitize(table_name);
	string internal_name = "_pac_internal_sample_" + sanitized;

	// If the internal table does not exist, create it and populate with 100 rows
	if (!TableExists(context, internal_name)) {
		// Create the table using the catalog API instead of context.Query to avoid locking/deadlocks.
		try {
			// Build CreateTableInfo in the current default catalog so the helper table persists across restarts
			string current_catalog = DatabaseManager::GetDefaultDatabase(context);
			auto create_info =
			    unique_ptr<CreateTableInfo>(new CreateTableInfo(current_catalog, DEFAULT_SCHEMA, internal_name));
			// Ignore conflicts if the name already exists
			create_info->on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
			// Define columns: rowid BIGINT, sample_id BIGINT
			create_info->columns.AddColumn(ColumnDefinition("rowid", LogicalType::BIGINT));
			create_info->columns.AddColumn(ColumnDefinition("sample_id", LogicalType::BIGINT));

			// Mark the database as modified on the current meta-transaction so creating a catalog entry is allowed
			Catalog &target_catalog = Catalog::GetCatalog(context, current_catalog);
			MetaTransaction::Get(context).ModifyDatabase(target_catalog.GetAttached());

			// Use the current catalog to create the table in the current context
			auto created_entry = target_catalog.CreateTable(context, std::move(create_info));
			if (!created_entry) {
				throw InvalidInputException(
				    StringUtil::Format("failed to create internal PAC helper table %s: unknown error", internal_name));
			}

			// The Catalog::CreateTable returns an optional_ptr<CatalogEntry>. Retrieve the TableCatalogEntry by casting
			TableCatalogEntry &table_entry = created_entry->Cast<TableCatalogEntry>();

			// Use InternalAppender to populate with 100 rows without issuing SQL queries
			InternalAppender appender(context, table_entry);
			for (int64_t i = 1; i <= 100; ++i) {
				appender.BeginRow();
				appender.Append<int64_t>(i);
				appender.Append<int64_t>(i);
				appender.EndRow();
			}
			appender.Close();
		} catch (std::exception &ex) {
			throw InvalidInputException(
			    StringUtil::Format("failed to create internal PAC helper table %s: %s", internal_name, ex.what()));
		}
	}
}

// Pragma-style API: PRAGMA remove_privacy_unit('table', 'filename?')
void RemovePrivacyUnitPragma(ClientContext &context, const FunctionParameters &parameters) {
	if (parameters.values.empty()) {
		throw InvalidInputException("remove_privacy_unit pragma requires at least the table name");
	}
	auto table_name = parameters.values[0].ToString();
	string pac_privacy_file = GetPacPrivacyFile(context);
	if (parameters.values.size() >= 2) {
		pac_privacy_file = parameters.values[1].ToString();
	}
	// Do not pre-create the file - ReadPacTablesFile will handle a missing file as empty
	auto tables = ReadPacTablesFile(pac_privacy_file);
	if (tables.erase(table_name) > 0) {
		WritePacTablesFile(pac_privacy_file, tables);
	}

	// Also drop the internal helper table if it exists
	string sanitized = Sanitize(table_name);
	string internal_name = "_pac_internal_sample_" + sanitized;
	if (TableExists(context, internal_name)) {
		string drop_sql = "DROP TABLE IF EXISTS " + internal_name + ";";
		try {
			Connection con(*context.db);
			con.BeginTransaction();
			con.Query(drop_sql);
			con.Commit();
		} catch (std::exception &ex) {
			// Non-fatal; surface as InvalidInputException
			throw InvalidInputException(
			    StringUtil::Format("failed to drop internal PAC helper table %s: %s", internal_name, ex.what()));
		}
	}
}

} // namespace duckdb
