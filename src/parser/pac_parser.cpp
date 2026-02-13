//
// PAC Parser Extension
//
// This file implements the main parser extension for PAC (Privacy-Augmented Computation).
// It provides entry points for parsing PAC DDL statements and coordinating with the
// metadata manager, serialization, and parser helper modules.
//
// The heavy lifting has been refactored into separate modules:
// - pac_metadata_manager.cpp: Metadata storage and retrieval
// - pac_metadata_serialization.cpp: JSON serialization/deserialization
// - pac_parser_helpers.cpp: SQL parsing and clause extraction
//
// Created by ila on 12/21/25.
// Refactored on 1/22/26.
//

#include "parser/pac_parser.hpp"
#include "pac_debug.hpp"

#include "duckdb/parser/parser.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"

#include <iostream>

namespace duckdb {

// ============================================================================
// Helper functions for column validation
// ============================================================================

/**
 * FindMissingColumns: Checks which columns from a list don't exist in a table
 */
static vector<string> FindMissingColumns(const TableCatalogEntry &table, const vector<string> &column_names) {
	vector<string> missing;
	for (const auto &col_name : column_names) {
		bool found = false;
		for (auto &col : table.GetColumns().Logical()) {
			if (StringUtil::Lower(col.GetName()) == StringUtil::Lower(col_name)) {
				found = true;
				break;
			}
		}
		if (!found) {
			missing.push_back(col_name);
		}
	}
	return missing;
}

/**
 * FormatColumnList: Formats a list of column names for error messages
 */
static string FormatColumnList(const vector<string> &columns) {
	if (columns.empty()) {
		return "";
	}
	if (columns.size() == 1) {
		return "'" + columns[0] + "'";
	}

	string result;
	for (size_t i = 0; i < columns.size(); i++) {
		if (i > 0) {
			result += ", ";
		}
		result += "'" + columns[i] + "'";
	}
	return result;
}

/**
 * ValidateColumnsExist: Validates that all columns exist in a table, throws exception if not
 */
static void ValidateColumnsExist(const TableCatalogEntry &table, const vector<string> &column_names,
                                 const string &table_name, const string &context_msg = "") {
	auto missing = FindMissingColumns(table, column_names);
	if (!missing.empty()) {
		string error_msg;
		if (missing.size() == 1) {
			error_msg = "Column " + FormatColumnList(missing) + " does not exist in ";
		} else {
			error_msg = "Columns " + FormatColumnList(missing) + " do not exist in ";
		}

		if (context_msg.empty()) {
			error_msg += "table '" + table_name + "'";
		} else {
			error_msg += context_msg;
		}

		throw CatalogException(error_msg);
	}
}

// ============================================================================
// Custom FunctionData for PAC DDL execution
// ============================================================================

/**
 * PACDDLBindData: Stores information about a PAC DDL operation to execute
 *
 * The PAC parser extension needs to execute DDL operations (CREATE TABLE, ALTER TABLE)
 * while also managing metadata. This struct stores the stripped SQL (with PAC clauses removed)
 * and the table name to track which metadata was updated.
 */
struct PACDDLBindData : public TableFunctionData {
	string stripped_sql; // SQL with PAC-specific clauses removed
	bool executed;       // Whether the DDL has been executed
	string table_name;   // Table whose metadata was updated

	explicit PACDDLBindData(string sql, string tbl_name = "")
	    : stripped_sql(std::move(sql)), executed(false), table_name(std::move(tbl_name)) {
	}
};

// Thread-local storage for pending DDL operations
// This is a workaround for passing data between parse and bind phases
// since the bind function can't directly receive custom parameters.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local string g_pac_pending_sql;
static thread_local string g_pac_pending_table_name;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// ============================================================================
// Static bind and execution functions for PAC DDL
// ============================================================================

/**
 * PACDDLBindFunction: Executes the DDL and saves metadata to file
 *
 * This function runs during the bind phase of query execution. It:
 * 1. Retrieves the pending SQL from thread-local storage
 * 2. Executes the DDL using a separate connection (to avoid deadlocks)
 * 3. Saves the updated PAC metadata to the JSON file
 *
 * The metadata is saved after every PAC DDL operation (CREATE/ALTER) to ensure
 * it persists across database sessions. For in-memory databases, no file is saved.
 */
static unique_ptr<FunctionData> PACDDLBindFunction(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	// Get the pending SQL from thread-local storage
	string sql_to_execute = g_pac_pending_sql;
	string table_name = g_pac_pending_table_name;
	g_pac_pending_sql.clear();
	g_pac_pending_table_name.clear();

	if (!sql_to_execute.empty()) {
		// Use a separate connection to execute the DDL to avoid deadlock
		auto &db = DatabaseInstance::GetDatabase(context);
		Connection conn(db);
		auto result = conn.Query(sql_to_execute);
		if (result->HasError()) {
			throw InternalException("Failed to execute DDL: " + result->GetError());
		}
	}

	// Save metadata to JSON file after any PAC DDL operation (CREATE or ALTER)
	// For ALTER PAC TABLE, sql_to_execute is empty but table_name is set
	// Only save to file if NOT an in-memory database
	if (!sql_to_execute.empty() || !table_name.empty()) {
		string metadata_path = PACMetadataManager::GetMetadataFilePath(context);
		// Don't save to file for in-memory databases
		if (!metadata_path.empty()) {
#if PAC_DEBUG
			std::cerr << "[PAC DEBUG] PACDDLBindFunction: Saving metadata to: " << metadata_path << "\n";
			std::cerr << "[PAC DEBUG] PACDDLBindFunction: table_name=" << table_name
			          << ", sql_to_execute.empty()=" << sql_to_execute.empty() << "\n";
#endif
			PACMetadataManager::Get().SaveToFile(metadata_path);
		}
	}

	// Set up return type for empty result
	return_types.push_back(LogicalType::BOOLEAN);
	names.push_back("success");

	return make_uniq<PACDDLBindData>(sql_to_execute, table_name);
}

/**
 * PACDDLExecuteFunction: Returns empty result set (DDL was already executed in bind)
 */
static void PACDDLExecuteFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	// Nothing to output - DDL was already executed in bind
	output.SetCardinality(0);
}

// ============================================================================
// PACParserExtension Main Entry Points
// ============================================================================

/**
 * PACParseFunction: Main entry point for parsing PAC statements
 *
 * This function is called by DuckDB's parser extension mechanism for every query.
 * It:
 * 1. Cleans the query (removes semicolons, normalizes whitespace)
 * 2. Attempts to parse as CREATE PAC TABLE
 * 3. Attempts to parse as ALTER PAC TABLE DROP (checked before ADD)
 * 4. Attempts to parse as ALTER PAC TABLE ADD
 * 5. Returns empty result if no PAC syntax found (let normal parser handle it)
 *
 * The order matters: DROP must be checked before ADD because DROP statements
 * also contain keywords like "protected" that might match ADD patterns.
 */
ParserExtensionParseResult PACParserExtension::PACParseFunction(ParserExtensionInfo *info, const string &query) {
	// Clean up query - preserve spaces but remove semicolons and newlines
	string clean_query = query;
	// Remove semicolons
	clean_query.erase(std::remove(clean_query.begin(), clean_query.end(), ';'), clean_query.end());
	// Replace newlines with spaces
	std::replace(clean_query.begin(), clean_query.end(), '\n', ' ');
	std::replace(clean_query.begin(), clean_query.end(), '\r', ' ');
	std::replace(clean_query.begin(), clean_query.end(), '\t', ' ');
	// Trim whitespace
	StringUtil::Trim(clean_query);

	PACTableMetadata metadata;
	string stripped_sql;
	bool is_pac_ddl = false;

	// Try to parse as CREATE PAC TABLE
	if (ParseCreatePACTable(clean_query, stripped_sql, metadata)) {
		is_pac_ddl = true;
	}
	// Try to parse as ALTER TABLE DROP PAC (must be checked BEFORE ADD since DROP commands also contain keywords like
	// "protected")
	else if (ParseAlterTableDropPAC(clean_query, stripped_sql, metadata)) {
		is_pac_ddl = true;
	}
	// Try to parse as ALTER TABLE ADD PAC
	else if (ParseAlterTableAddPAC(clean_query, stripped_sql, metadata)) {
		is_pac_ddl = true;
	}

	// If no PAC syntax found, return empty result (let normal parser handle it)
	if (!is_pac_ddl) {
		return ParserExtensionParseResult();
	}

	// Return the parse data
	return ParserExtensionParseResult(make_uniq<PACParseData>(stripped_sql, metadata, is_pac_ddl));
}

/**
 * PACPlanFunction: Converts parsed PAC statement into execution plan
 *
 * This function:
 * 1. Validates metadata (columns exist, tables exist)
 * 2. Stores metadata in PACMetadataManager
 * 3. Sets up a table function to execute the DDL
 *
 * For CREATE PAC TABLE:
 *   - Executes the stripped SQL (CREATE TABLE without PAC clauses)
 *   - Saves metadata to file
 *
 * For ALTER PAC TABLE:
 *   - No DDL executed (metadata-only operation)
 *   - Validates columns/tables exist
 *   - Saves updated metadata to file
 */
ParserExtensionPlanResult PACParserExtension::PACPlanFunction(ParserExtensionInfo *info, ClientContext &context,
                                                              unique_ptr<ParserExtensionParseData> parse_data) {
	auto &pac_data = dynamic_cast<PACParseData &>(*parse_data);

	// Validate metadata before storing it
	if (pac_data.is_pac_ddl && !pac_data.metadata.table_name.empty()) {
		// For ALTER PAC TABLE operations, validate that the table exists
		if (pac_data.stripped_sql.empty()) {
			// This is an ALTER PAC TABLE operation (stripped_sql is empty for ALTER PAC TABLE)
			// Check if table exists in the catalog
			try {
				auto &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
				auto table_entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA,
				                                    pac_data.metadata.table_name, OnEntryNotFound::RETURN_NULL);
				if (!table_entry) {
					throw CatalogException("Table '" + pac_data.metadata.table_name + "' does not exist");
				}

				auto &table = table_entry->Cast<TableCatalogEntry>();

				// Validate ALL protected columns exist before adding any (atomic operation)
				if (!pac_data.metadata.protected_columns.empty()) {
					ValidateColumnsExist(table, pac_data.metadata.protected_columns, pac_data.metadata.table_name,
					                     "table '" + pac_data.metadata.table_name +
					                         "'. No protected columns were added.");
				}

				// Validate ALL columns in PAC LINKs exist before adding any (atomic operation)
				for (const auto &link : pac_data.metadata.links) {
					// Check local columns
					ValidateColumnsExist(table, link.local_columns, pac_data.metadata.table_name,
					                     "table '" + pac_data.metadata.table_name + "'. PAC LINK was not added.");

					// Validate that referenced table exists
					auto ref_table_entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA,
					                                        link.referenced_table, OnEntryNotFound::RETURN_NULL);
					if (!ref_table_entry) {
						throw CatalogException("Referenced table '" + link.referenced_table + "' does not exist");
					}

					// Check for cycles: a PAC table (privacy unit) cannot link to another PAC table
					// Regular tables can link to PAC tables, but PAC tables cannot link to other PAC tables
					if (pac_data.metadata.is_privacy_unit) {
						auto ref_metadata = PACMetadataManager::Get().GetTableMetadata(link.referenced_table);
						if (ref_metadata && ref_metadata->is_privacy_unit) {
							throw CatalogException(
							    "Cannot create PAC LINK from PAC table '" + pac_data.metadata.table_name +
							    "' to PAC table '" + link.referenced_table +
							    "'. PAC tables cannot link to other PAC tables (cycles not supported).");
						}
					}

					// Check referenced columns
					auto &ref_table = ref_table_entry->Cast<TableCatalogEntry>();
					ValidateColumnsExist(ref_table, link.referenced_columns, link.referenced_table,
					                     "referenced table '" + link.referenced_table + "'. PAC LINK was not added.");
				}
			} catch (const CatalogException &e) {
				throw;
			}
		} else {
			// This is a CREATE PAC TABLE operation
			// Validate cycle detection: PAC tables cannot link to other PAC tables
			if (pac_data.metadata.is_privacy_unit && !pac_data.metadata.links.empty()) {
				auto &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
				for (const auto &link : pac_data.metadata.links) {
					// Check if referenced table exists and is a PAC table
					auto ref_table_entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA,
					                                        link.referenced_table, OnEntryNotFound::RETURN_NULL);
					if (ref_table_entry) {
						auto ref_metadata = PACMetadataManager::Get().GetTableMetadata(link.referenced_table);
						if (ref_metadata && ref_metadata->is_privacy_unit) {
							throw CatalogException(
							    "Cannot create PAC LINK from PAC table '" + pac_data.metadata.table_name +
							    "' to PAC table '" + link.referenced_table +
							    "'. PAC tables cannot link to other PAC tables (cycles not supported).");
						}
					}
				}
			}
		}

		// Store metadata in global manager after validation
		PACMetadataManager::Get().AddOrUpdateTable(pac_data.metadata.table_name, pac_data.metadata);
	}

	// Store the SQL in thread-local storage for the bind function to execute
	g_pac_pending_sql = pac_data.stripped_sql;
	g_pac_pending_table_name = pac_data.metadata.table_name;

	// Return a table function that will execute the DDL in its bind phase
	ParserExtensionPlanResult plan_result;
	plan_result.function = TableFunction("pac_ddl_executor", {}, PACDDLExecuteFunction, PACDDLBindFunction);
	plan_result.requires_valid_transaction = true;
	plan_result.return_type = StatementReturnType::QUERY_RESULT;

	return plan_result;
}

} // namespace duckdb
