#include "pac_parser.hpp"

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
#include <fstream>
#include <regex>
#include <sstream>

namespace duckdb {

// ============================================================================
// Helper functions for column validation
// ============================================================================

/**
 * FindMissingColumns: Checks which columns from a list don't exist in a table
 *
 * @param table - The table catalog entry to search
 * @param column_names - List of column names to validate
 * @return Vector of column names that were NOT found in the table
 *
 * This function performs case-insensitive column name matching.
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
 *
 * Examples:
 *   []           -> ""
 *   ["col1"]     -> "'col1'"
 *   ["a", "b"]   -> "'a', 'b'"
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
 *
 * @param table - The table catalog entry to search
 * @param column_names - List of column names to validate
 * @param table_name - Name of the table (for error messages)
 * @param context_msg - Optional context for error message
 *
 * Throws CatalogException with detailed error message if any columns are missing.
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
	string stripped_sql;    // SQL with PAC-specific clauses removed
	bool executed;          // Whether the DDL has been executed
	string table_name;      // Table whose metadata was updated

	explicit PACDDLBindData(string sql, string tbl_name = "")
	    : stripped_sql(std::move(sql)), executed(false), table_name(std::move(tbl_name)) {
	}
};

// Thread-local storage for pending DDL operations
// This is a workaround for passing data between parse and bind phases
// since the bind function can't directly receive custom parameters.
static thread_local string g_pac_pending_sql;
static thread_local string g_pac_pending_table_name;

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
#ifdef DEBUG
			std::cerr << "[PAC DEBUG] PACDDLBindFunction: Saving metadata to: " << metadata_path << std::endl;
			std::cerr << "[PAC DEBUG] PACDDLBindFunction: table_name=" << table_name
			          << ", sql_to_execute.empty()=" << sql_to_execute.empty() << std::endl;
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
// PACMetadataManager Implementation
// ============================================================================

/**
 * PACMetadataManager: Singleton that manages PAC table metadata in memory
 *
 * This class stores metadata for all PAC-protected tables:
 * - Primary keys (PAC KEY)
 * - Foreign key links (PAC LINK)
 * - Protected columns (PROTECTED)
 *
 * The metadata is stored in memory and can be serialized to/from JSON files
 * for persistence across database sessions.
 */
PACMetadataManager &PACMetadataManager::Get() {
	static PACMetadataManager instance;
	return instance;
}

/**
 * AddOrUpdateTable: Adds a new table's metadata or updates existing metadata
 *
 * @param table_name - Name of the table
 * @param metadata - Metadata for the table
 *
 * This function is thread-safe and locks the metadata map during the operation.
 */
void PACMetadataManager::AddOrUpdateTable(const string &table_name, const PACTableMetadata &metadata) {
	std::lock_guard<std::mutex> lock(metadata_mutex);
	table_metadata[table_name] = metadata;
}

/**
 * GetTableMetadata: Retrieves metadata for a table
 *
 * @param table_name - Name of the table
 * @return Pointer to the table metadata, or nullptr if not found
 */
const PACTableMetadata *PACMetadataManager::GetTableMetadata(const string &table_name) const {
	std::lock_guard<std::mutex> lock(metadata_mutex);
	auto it = table_metadata.find(table_name);
	if (it != table_metadata.end()) {
		return &it->second;
	}
	return nullptr;
}

/**
 * HasMetadata: Checks if metadata exists for a table
 *
 * @param table_name - Name of the table
 * @return True if metadata exists, false otherwise
 */
bool PACMetadataManager::HasMetadata(const string &table_name) const {
	std::lock_guard<std::mutex> lock(metadata_mutex);
	return table_metadata.find(table_name) != table_metadata.end();
}

/**
 * GetAllTableNames: Retrieves a list of all table names with metadata
 *
 * @return Vector of table names
 */
vector<string> PACMetadataManager::GetAllTableNames() const {
	std::lock_guard<std::mutex> lock(metadata_mutex);
	vector<string> names;
	names.reserve(table_metadata.size());
	for (const auto &entry : table_metadata) {
		names.push_back(entry.first);
	}
	return names;
}

/**
 * RemoveTable: Removes a table's metadata
 *
 * @param table_name - Name of the table
 *
 * This function is thread-safe and locks the metadata map during the operation.
 */
void PACMetadataManager::RemoveTable(const string &table_name) {
	std::lock_guard<std::mutex> lock(metadata_mutex);
	table_metadata.erase(table_name);
}

/**
 * SerializeToJSON: Converts a PACTableMetadata struct to JSON format
 *
 * JSON Structure:
 * {
 *   "table_name": "orders",
 *   "primary_keys": ["o_orderkey"],
 *   "links": [
 *     {
 *       "local_columns": ["o_custkey"],
 *       "referenced_table": "customer",
 *       "referenced_columns": ["c_custkey"]
 *     }
 *   ],
 *   "protected_columns": ["o_totalprice", "o_orderdate"]
 * }
 */
string PACMetadataManager::SerializeToJSON(const PACTableMetadata &metadata) const {
	std::stringstream ss;
	ss << "{\n";
	ss << "  \"table_name\": \"" << metadata.table_name << "\",\n";

	// Serialize primary keys
	ss << "  \"primary_keys\": [";
	for (size_t i = 0; i < metadata.primary_key_columns.size(); i++) {
		if (i > 0)
			ss << ", ";
		ss << "\"" << metadata.primary_key_columns[i] << "\"";
	}
	ss << "],\n";

	// Serialize links (now with support for composite keys)
	ss << "  \"links\": [\n";
	for (size_t i = 0; i < metadata.links.size(); i++) {
		if (i > 0)
			ss << ",\n";
		const auto &link = metadata.links[i];
		ss << "    {\n";

		// Serialize local_columns array
		ss << "      \"local_columns\": [";
		for (size_t j = 0; j < link.local_columns.size(); j++) {
			if (j > 0)
				ss << ", ";
			ss << "\"" << link.local_columns[j] << "\"";
		}
		ss << "],\n";

		ss << "      \"referenced_table\": \"" << link.referenced_table << "\",\n";

		// Serialize referenced_columns array
		ss << "      \"referenced_columns\": [";
		for (size_t j = 0; j < link.referenced_columns.size(); j++) {
			if (j > 0)
				ss << ", ";
			ss << "\"" << link.referenced_columns[j] << "\"";
		}
		ss << "]\n";

		ss << "    }";
	}
	ss << "\n  ],\n";

	// Serialize protected columns
	ss << "  \"protected_columns\": [";
	for (size_t i = 0; i < metadata.protected_columns.size(); i++) {
		if (i > 0)
			ss << ", ";
		ss << "\"" << metadata.protected_columns[i] << "\"";
	}
	ss << "]\n";

	ss << "}";
	return ss.str();
}

/**
 * SerializeAllToJSON: Serializes metadata for all tables to JSON format
 *
 * @return JSON string representing all table metadata
 */
string PACMetadataManager::SerializeAllToJSON() const {
	std::lock_guard<std::mutex> lock(metadata_mutex);
	std::stringstream ss;
	ss << "{\n  \"tables\": [\n";

	size_t idx = 0;
	for (const auto &entry : table_metadata) {
		if (idx > 0)
			ss << ",\n";
		ss << "    " << SerializeToJSON(entry.second);
		idx++;
	}

	ss << "\n  ]\n}";
	return ss.str();
}

/**
 * SaveToFile: Saves metadata to a JSON file
 *
 * @param filepath - Path to the file
 *
 * Throws IOException if the file can't be opened or written.
 */
void PACMetadataManager::SaveToFile(const string &filepath) {
	std::ofstream file(filepath);
	if (!file.is_open()) {
		throw IOException("Failed to open PAC metadata file for writing: " + filepath);
	}
	file << SerializeAllToJSON();
	file.close();
}

/**
 * LoadFromFile: Loads metadata from a JSON file
 *
 * @param filepath - Path to the file
 *
 * Throws IOException if the file can't be opened or read.
 */
void PACMetadataManager::LoadFromFile(const string &filepath) {
	std::ifstream file(filepath);
	if (!file.is_open()) {
		throw IOException("Failed to open PAC metadata file for reading: " + filepath);
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	file.close();

	DeserializeAllFromJSON(buffer.str());
}

/**
 * DeserializeFromJSON: Parses JSON and constructs a PACTableMetadata struct
 *
 * This function supports both old format (single local_column/referenced_column)
 * and new format (arrays local_columns/referenced_columns) for backward compatibility.
 */
PACTableMetadata PACMetadataManager::DeserializeFromJSON(const string &json) {
	// Simple JSON parsing (for production, consider using a proper JSON library)
	PACTableMetadata metadata;

	// Extract table name
	std::regex table_name_regex(R"xxx("table_name"\s*:\s*"([^"]+)")xxx");
	std::smatch match;
	if (std::regex_search(json, match, table_name_regex)) {
		metadata.table_name = match[1].str();
	}

	// Extract primary keys
	std::regex pk_regex(R"xxx("primary_keys"\s*:\s*\[(.*?)\])xxx");
	if (std::regex_search(json, match, pk_regex)) {
		string pk_list = match[1].str();
		std::regex col_regex(R"xxx("([^"]+)")xxx");
		auto begin = std::sregex_iterator(pk_list.begin(), pk_list.end(), col_regex);
		auto end = std::sregex_iterator();
		for (auto it = begin; it != end; ++it) {
			metadata.primary_key_columns.push_back((*it)[1].str());
		}
	}

	// Extract links section by finding "links" and manually parsing the array
	size_t links_pos = json.find("\"links\"");
	if (links_pos != string::npos) {
		// Find the opening bracket of the links array
		size_t array_start = json.find('[', links_pos);
		if (array_start != string::npos) {
			// Find the matching closing bracket by counting brackets
			int bracket_count = 0;
			size_t array_end = array_start;
			for (size_t i = array_start; i < json.length(); i++) {
				if (json[i] == '[') {
					bracket_count++;
				} else if (json[i] == ']') {
					bracket_count--;
					if (bracket_count == 0) {
						array_end = i;
						break;
					}
				}
			}

			// Extract the links array content (without the outer brackets)
			string links_section = json.substr(array_start + 1, array_end - array_start - 1);

			// Manually parse link objects by counting braces to handle nested arrays
			size_t pos = 0;
			while (pos < links_section.length()) {
				// Skip whitespace and commas
				while (pos < links_section.length() && (links_section[pos] == ' ' || links_section[pos] == ',' ||
				                                        links_section[pos] == '\n' || links_section[pos] == '\t')) {
					pos++;
				}

				if (pos >= links_section.length() || links_section[pos] != '{') {
					break;
				}

				// Find the matching closing brace
				int brace_count = 0;
				size_t start = pos;
				while (pos < links_section.length()) {
					if (links_section[pos] == '{') {
						brace_count++;
					} else if (links_section[pos] == '}') {
						brace_count--;
						if (brace_count == 0) {
							pos++; // Include the closing brace
							break;
						}
					}
					pos++;
				}

				// Extract the link object JSON
				string link_json = links_section.substr(start, pos - start);
				PACLink link;

				// Try new format first (local_columns/referenced_columns arrays)
				std::regex local_cols_regex(R"xxx("local_columns"\s*:\s*\[(.*?)\])xxx");
				std::regex ref_cols_regex(R"xxx("referenced_columns"\s*:\s*\[(.*?)\])xxx");
				std::regex ref_table_regex(R"xxx("referenced_table"\s*:\s*"([^"]+)")xxx");

				std::smatch link_match;
				bool is_new_format = false;

				if (std::regex_search(link_json, link_match, local_cols_regex)) {
					is_new_format = true;
					string local_cols_str = link_match[1].str();
					std::regex col_regex(R"xxx("([^"]+)")xxx");
					auto cols_begin = std::sregex_iterator(local_cols_str.begin(), local_cols_str.end(), col_regex);
					auto cols_end = std::sregex_iterator();
					for (auto col_it = cols_begin; col_it != cols_end; ++col_it) {
						link.local_columns.push_back((*col_it)[1].str());
					}
				}

				if (std::regex_search(link_json, link_match, ref_table_regex)) {
					link.referenced_table = link_match[1].str();
				}

				if (std::regex_search(link_json, link_match, ref_cols_regex)) {
					is_new_format = true;
					string ref_cols_str = link_match[1].str();
					std::regex col_regex(R"xxx("([^"]+)")xxx");
					auto cols_begin = std::sregex_iterator(ref_cols_str.begin(), refColsStr.end(), col_regex);
					auto cols_end = std::sregex_iterator();
					for (auto col_it = cols_begin; col_it != cols_end; ++col_it) {
						link.referenced_columns.push_back((*col_it)[1].str());
					}
				}

				// Fall back to old format (single local_column/referenced_column)
				if (!is_new_format) {
					std::regex local_col_regex(R"xxx("local_column"\s*:\s*"([^"]+)")xxx");
					std::regex ref_col_regex(R"xxx("referenced_column"\s*:\s*"([^"]+)")xxx");

					if (std::regex_search(link_json, link_match, local_col_regex)) {
						link.local_columns.push_back(link_match[1].str());
					}
					if (std::regex_search(link_json, link_match, ref_col_regex)) {
						link.referenced_columns.push_back(link_match[1].str());
					}
				}

				if (!link.local_columns.empty() && !link.referenced_table.empty()) {
					metadata.links.push_back(link);
				}
			}
		}
	}

	// Extract protected columns
	std::regex protected_regex(R"xxx("protected_columns"\s*:\s*\[(.*?)\])xxx");
	if (std::regex_search(json, match, protected_regex)) {
		string protected_list = match[1].str();
		std::regex col_regex(R"xxx("([^"]+)")xxx");
		auto begin = std::sregex_iterator(protected_list.begin(), protected_list.end(), col_regex);
		auto end = std::sregex_iterator();
		for (auto it = begin; it != end; ++it) {
			metadata.protected_columns.push_back((*it)[1].str());
		}
	}

	return metadata;
}

/**
 * DeserializeAllFromJSON: Deserializes metadata for all tables from JSON format
 *
 * @param json - JSON string representing all table metadata
 *
 * This function clears existing metadata and parses the JSON to restore metadata
 * for all tables. It expects the JSON to have a "tables" array containing individual
 * table objects.
 */
void PACMetadataManager::DeserializeAllFromJSON(const string &json) {
	std::lock_guard<std::mutex> lock(metadata_mutex);
	table_metadata.clear();

	// Extract all table objects using a better approach
	// Find the "tables" array
	size_t tables_start = json.find("\"tables\"");
	if (tables_start == string::npos) {
		return;
	}

	// Find the opening bracket of the tables array
	size_t array_start = json.find('[', tables_start);
	if (array_start == string::npos) {
		return;
	}

	// Manually parse table objects by counting braces
	size_t pos = array_start + 1;
	int brace_count = 0;
	size_t obj_start = string::npos;

	while (pos < json.length()) {
		char c = json[pos];

		if (c == '{') {
			if (brace_count == 0) {
				obj_start = pos;
			}
			brace_count++;
		} else if (c == '}') {
			brace_count--;
			if (brace_count == 0 && obj_start != string::npos) {
				// Extract the complete table object
				string table_json = json.substr(obj_start, pos - obj_start + 1);
				auto metadata = DeserializeFromJSON(table_json);
				if (!metadata.table_name.empty()) {
					table_metadata[metadata.table_name] = metadata;
				}
				obj_start = string::npos;
			}
		} else if (c == ']' && brace_count == 0) {
			// End of tables array
			break;
		}

		pos++;
	}
}

/**
 * Clear: Clears all metadata for tables
 *
 * This function is thread-safe and locks the metadata map during the operation.
 */
void PACMetadataManager::Clear() {
	std::lock_guard<std::mutex> lock(metadata_mutex);
	table_metadata.clear();
}

/**
 * GetMetadataFilePath: Constructs the path to the PAC metadata JSON file
 *
 * Format: <db_directory>/pac_metadata_<dbname>_<schema>.json
 *
 * For example:
 *   - Database: /data/tpch.db
 *   - Schema: main
 *   - Result: /data/pac_metadata_tpch_main.json
 *
 * Returns empty string for in-memory databases (no file saved).
 */
string PACMetadataManager::GetMetadataFilePath(ClientContext &context) {
	// Get the database path from the default catalog
	auto &db_name = DatabaseManager::GetDefaultDatabase(context);
	auto &catalog = Catalog::GetCatalog(context, db_name);
	string db_path = catalog.GetDBPath();

	// If in-memory database or empty path, return empty string (don't save to file)
	if (db_path.empty() || db_path == ":memory:") {
		return "";
	}

	// Extract schema name from the catalog search path
	string schema_name = DEFAULT_SCHEMA; // Fallback to "main"
	try {
		CatalogSearchPath search_path(context);
		auto entries = search_path.Get();
		if (!entries.empty()) {
			schema_name = entries[0].schema;
		}
	} catch (...) {
		// If we can't get search path, use default schema
		schema_name = DEFAULT_SCHEMA;
	}

	// Extract directory from database path and append metadata filename
	// Format: pac_metadata_<dbname>_<schema>.json
	string filename = "pac_metadata_" + db_name + "_" + schema_name + ".json";

	size_t last_slash = db_path.find_last_of("/\\");
	if (last_slash != string::npos) {
		return db_path.substr(0, last_slash + 1) + filename;
	}

	// No directory separator found, use current directory
	return filename;
}

// ============================================================================
// PACParserExtension Implementation
// ============================================================================

/**
 * ExtractTableName: Extracts table name from CREATE or ALTER TABLE statement
 *
 * Handles:
 *   - CREATE TABLE table_name ...
 *   - CREATE PAC TABLE table_name ...
 *   - CREATE TABLE IF NOT EXISTS table_name ...
 *   - ALTER TABLE table_name ...
 */
string PACParserExtension::ExtractTableName(const string &sql, bool is_create) {
	if (is_create) {
		// Match: CREATE [PAC] TABLE [IF NOT EXISTS] table_name
		std::regex create_regex(R"(create\s+(?:pac\s+)?table\s+(?:if\s+not\s+exists\s+)?([a-zA-Z_][a-zA-Z0-9_]*))");
		std::smatch match;
		string sql_lower = StringUtil::Lower(sql);
		if (std::regex_search(sql_lower, match, create_regex)) {
			return match[1].str();
		}
	} else {
		// Match: ALTER TABLE table_name
		std::regex alter_regex(R"(alter\s+table\s+([a-zA-Z_][a-zA-Z0-9_]*))");
		std::smatch match;
		string sql_lower = StringUtil::Lower(sql);
		if (std::regex_search(sql_lower, match, alter_regex)) {
			return match[1].str();
		}
	}
	return "";
}

/**
 * ExtractPACPrimaryKey: Parses PAC KEY clause
 *
 * Syntax: PAC KEY (col1, col2, ...)
 *
 * Supports composite keys (multiple columns).
 */
bool PACParserExtension::ExtractPACPrimaryKey(const string &clause, vector<string> &pk_columns) {
	// Match: PAC KEY (col1, col2, ...)
	std::regex pk_regex(R"(pac\s+key\s*\(\s*([^)]+)\s*\))");
	std::smatch match;
	string clause_lower = StringUtil::Lower(clause);

	if (std::regex_search(clause_lower, match, pk_regex)) {
		string cols_str = match[1].str();
		// Split by comma
		auto cols = StringUtil::Split(cols_str, ',');
		for (auto &col : cols) {
			StringUtil::Trim(col);
			if (!col.empty()) {
				pk_columns.push_back(col);
			}
		}
		return true;
	}
	return false;
}

/**
 * ExtractPACLink: Parses PAC LINK clause
 *
 * Syntax: PAC LINK (local_col1, local_col2, ...) REFERENCES table(ref_col1, ref_col2, ...)
 *
 * This defines a foreign key relationship for PAC. The number of local columns
 * must match the number of referenced columns (composite key support).
 */
bool PACParserExtension::ExtractPACLink(const string &clause, PACLink &link) {
	// Match: PAC LINK (col1, col2, ...) REFERENCES table_name(ref_col1, ref_col2, ...)
	// Support both single and composite foreign keys
	std::regex link_regex(
	    R"(pac\s+link\s*\(\s*([^)]+)\s*\)\s+references\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^)]+)\s*\))");
	std::smatch match;
	string clause_lower = StringUtil::Lower(clause);

	if (std::regex_search(clause_lower, match, link_regex)) {
		// Parse local columns (comma-separated)
		string local_cols_str = match[1].str();
		auto local_cols = StringUtil::Split(local_cols_str, ',');
		for (auto &col : local_cols) {
			StringUtil::Trim(col);
			if (!col.empty()) {
				link.local_columns.push_back(col);
			}
		}

		// Parse referenced table
		link.referenced_table = match[2].str();

		// Parse referenced columns (comma-separated)
		string ref_cols_str = match[3].str();
		auto ref_cols = StringUtil::Split(ref_cols_str, ',');
		for (auto &col : ref_cols) {
			StringUtil::Trim(col);
			if (!col.empty()) {
				link.referenced_columns.push_back(col);
			}
		}

		// Validate that number of local columns matches number of referenced columns
		if (link.local_columns.size() != link.referenced_columns.size()) {
			return false;
		}

		return !link.local_columns.empty();
	}
	return false;
}

/**
 * ExtractProtectedColumns: Parses PROTECTED clause
 *
 * Syntax: PROTECTED (col1, col2, ...)
 *
 * Protected columns are those that contain sensitive data and need PAC protection.
 */
bool PACParserExtension::ExtractProtectedColumns(const string &clause, vector<string> &protected_cols) {
	// Match: PROTECTED (col1, col2, ...)
	std::regex protected_regex(R"(protected\s*\(\s*([^)]+)\s*\))");
	std::smatch match;
	string clause_lower = StringUtil::Lower(clause);

	if (std::regex_search(clause_lower, match, protected_regex)) {
		string cols_str = match[1].str();
		auto cols = StringUtil::Split(cols_str, ',');
		for (auto &col : cols) {
			StringUtil::Trim(col);
			if (!col.empty()) {
				protected_cols.push_back(col);
			}
		}
		return true;
	}
	return false;
}

/**
 * StripPACClauses: Removes all PAC-specific clauses from SQL
 *
 * This is necessary because DuckDB's standard parser doesn't understand PAC syntax.
 * We remove PAC clauses and execute the "clean" SQL, while storing PAC metadata separately.
 *
 * Removed clauses:
 *   - PAC KEY (...)
 *   - PAC LINK (...) REFERENCES table(...)
 *   - PROTECTED (...)
 */
string PACParserExtension::StripPACClauses(const string &sql) {
	string result = sql;

	// Remove PAC KEY clauses (case-insensitive)
	result = std::regex_replace(
	    result, std::regex(R"(,?\s*[Pp][Aa][Cc]\s+[Kk][Ee][Yy]\s*\([^)]+\)\s*,?)", std::regex_constants::icase), " ");

	// Remove PAC LINK clauses (case-insensitive)
	result = std::regex_replace(
	    result,
	    std::regex(
	        R"(,?\s*[Pp][Aa][Cc]\s+[Ll][Ii][Nn][Kk]\s*\([^)]+\)\s+[Rr][Ee][Ff][Ee][Rr][Ee][Nn][Cc][Ee][Ss]\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]+\)\s*,?)",
	        std::regex_constants::icase),
	    " ");

	// Remove PROTECTED clauses (case-insensitive)
	result = std::regex_replace(
	    result,
	    std::regex(R"(,?\s*[Pp][Rr][Oo][Tt][Ee][Cc][Tt][Ee][Dd]\s*\([^)]+\)\s*,?)", std::regex_constants::icase), " ");

	// Clean up multiple spaces and trailing commas
	result = std::regex_replace(result, std::regex(R"(\s+)"), " ");
	result = std::regex_replace(result, std::regex(R"(,\s*,)"), ",");
	result = std::regex_replace(result, std::regex(R"(\(\s*,)"), "(");
	result = std::regex_replace(result, std::regex(R"(,\s*\))"), ")");

	return result;
}

/**
 * ParseCreatePACTable: Parses CREATE PAC TABLE statement
 *
 * Syntax:
 *   CREATE PAC TABLE table_name (
 *     col1 INTEGER,
 *     col2 VARCHAR,
 *     PAC KEY (col1),
 *     PAC LINK (col2) REFERENCES other_table(other_col),
 *     PROTECTED (col1, col2)
 *   );
 *
 * Returns:
 *   - stripped_sql: SQL with PAC clauses removed
 *   - metadata: Extracted PAC metadata (keys, links, protected columns)
 */
bool PACParserExtension::ParseCreatePACTable(const string &query, string &stripped_sql, PACTableMetadata &metadata) {
	string query_lower = StringUtil::Lower(query);

	// Check if this is a CREATE PAC TABLE statement
	if (query_lower.find("create pac table") == string::npos && query_lower.find("create table") == string::npos) {
		return false;
	}

	// Extract table name
	metadata.table_name = ExtractTableName(query, true);
	if (metadata.table_name.empty()) {
		return false;
	}

	// Check if any PAC clauses exist
	bool has_pac_clauses = query_lower.find("pac key") != string::npos ||
	                       query_lower.find("pac link") != string::npos ||
	                       query_lower.find("protected") != string::npos;

	if (!has_pac_clauses && query_lower.find("create pac table") == string::npos) {
		return false;
	}

	// Extract PAC clauses
	ExtractPACPrimaryKey(query, metadata.primary_key_columns);

	// Extract all PAC LINK clauses
	std::regex link_regex(R"(pac\s+link\s*\([^)]+\)\s+references\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]+\))");
	auto begin = std::sregex_iterator(query_lower.begin(), query_lower.end(), link_regex);
	auto end = std::sregex_iterator();
	for (auto it = begin; it != end; ++it) {
		PACLink link;
		if (ExtractPACLink((*it).str(), link)) {
			metadata.links.push_back(link);
		}
	}

	// Extract protected columns
	ExtractProtectedColumns(query, metadata.protected_columns);

	// Strip PAC clauses and replace "CREATE PAC TABLE" with "CREATE TABLE"
	stripped_sql = StripPACClauses(query);
	stripped_sql = std::regex_replace(stripped_sql, std::regex(R"(create\s+pac\s+table)", std::regex_constants::icase),
	                                  "CREATE TABLE");

	return true;
}

/**
 * ParseAlterTableAddPAC: Parses ALTER PAC TABLE ... ADD ... statement
 *
 * Syntax:
 *   ALTER PAC TABLE table_name ADD PAC KEY (col1);
 *   ALTER PAC TABLE table_name ADD PAC LINK (col1) REFERENCES other(col2);
 *   ALTER PAC TABLE table_name ADD PROTECTED (col1, col2);
 *
 * This operation is metadata-only (no actual DDL executed). It merges new
 * metadata with existing metadata for the table.
 *
 * Validation:
 *   - Columns must exist in the table
 *   - Protected columns can't be added twice (idempotency check)
 *   - PAC LINKs can't conflict (same local columns to different targets)
 */
bool PACParserExtension::ParseAlterTableAddPAC(const string &query, string &stripped_sql, PACTableMetadata &metadata) {
	string query_lower = StringUtil::Lower(query);

	// Check if this is an ALTER PAC TABLE statement
	// This is a PAC-specific syntax: ALTER PAC TABLE table_name ADD PAC KEY/LINK/PROTECTED
	if (query_lower.find("alter pac table") == string::npos) {
		return false;
	}

	// Extract table name - for ALTER PAC TABLE, the table name comes after "alter pac table"
	std::regex alter_pac_regex(R"(alter\s+pac\s+table\s+([a-zA-Z_][a-zA-Z0-9_]*))");
	std::smatch match;
	if (!std::regex_search(query_lower, match, alter_pac_regex)) {
		return false;
	}
	metadata.table_name = match[1].str();

	// Get existing metadata if any
	auto existing = PACMetadataManager::Get().GetTableMetadata(metadata.table_name);
	if (existing) {
		metadata = *existing;
#ifdef DEBUG
		std::cerr << "[PAC DEBUG] ParseAlterTableAddPAC: Found existing metadata for " << metadata.table_name
		          << ", links=" << existing->links.size() << ", protected=" << existing->protected_columns.size()
		          << std::endl;
#endif
	} else {
#ifdef DEBUG
		std::cerr << "[PAC DEBUG] ParseAlterTableAddPAC: No existing metadata for " << metadata.table_name << std::endl;
#endif
	}

	// Check for PAC-related keywords
	bool has_pac_key = query_lower.find("pac key") != string::npos;
	bool has_pac_link = query_lower.find("pac link") != string::npos;
	bool has_protected = query_lower.find("protected") != string::npos;

	// Extract and merge new PAC clauses
	if (has_pac_key) {
		vector<string> new_pk_cols;
		if (ExtractPACPrimaryKey(query, new_pk_cols)) {
			// Only add columns that don't already exist
			for (const auto &col : new_pk_cols) {
				if (std::find(metadata.primary_key_columns.begin(), metadata.primary_key_columns.end(), col) ==
				    metadata.primary_key_columns.end()) {
					metadata.primary_key_columns.push_back(col);
				}
			}
		}
	}

	// Extract PAC LINK clauses
	if (has_pac_link) {
		std::regex link_regex(R"(pac\s+link\s*\([^)]+\)\s+references\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]+\))",
		                      std::regex_constants::icase);
		auto begin = std::sregex_iterator(query.begin(), query.end(), link_regex);
		auto end = std::sregex_iterator();
		for (auto it = begin; it != end; ++it) {
			PACLink link;
			if (ExtractPACLink((*it).str(), link)) {
				// Check if a link already exists on these local columns (to a different table or columns)
				for (const auto &existing_link : metadata.links) {
					if (existing_link.local_columns == link.local_columns) {
						// Check if it's exactly the same link (same target table and columns)
						if (existing_link.referenced_table == link.referenced_table &&
						    existing_link.referenced_columns == link.referenced_columns) {
							// This is the exact same link - skip it (idempotent)
							continue;
						}
						// Different target - this is an error
						throw ParserException("Column(s) already have a PAC LINK defined. A column or set of columns "
						                      "can only reference one target.");
					}
				}

				// Add the link if it doesn't conflict
				bool link_exists = false;
				for (const auto &existing_link : metadata.links) {
					if (existing_link.local_columns == link.local_columns &&
					    existing_link.referenced_table == link.referenced_table &&
					    existing_link.referenced_columns == link.referenced_columns) {
						link_exists = true;
						break;
					}
				}
				if (!link_exists) {
					metadata.links.push_back(link);
				}
			}
		}
	}

	// Extract protected columns
	if (has_protected) {
		vector<string> new_protected_cols;
		if (ExtractProtectedColumns(query, new_protected_cols)) {
			// Check for duplicates in the new protected columns being added
			for (size_t i = 0; i < new_protected_cols.size(); i++) {
				for (size_t j = i + 1; j < new_protected_cols.size(); j++) {
					if (StringUtil::Lower(new_protected_cols[i]) == StringUtil::Lower(new_protected_cols[j])) {
						throw ParserException("Duplicate protected column '" + new_protected_cols[i] +
						                      "' in ALTER PAC TABLE statement");
					}
				}
			}

			// Only add columns that don't already exist
			for (const auto &col : new_protected_cols) {
				bool already_protected = false;
				for (const auto &existing_col : metadata.protected_columns) {
					if (StringUtil::Lower(existing_col) == StringUtil::Lower(col)) {
						already_protected = true;
						break;
					}
				}
				if (already_protected) {
					throw ParserException("Column '" + col + "' is already marked as protected");
				}
				metadata.protected_columns.push_back(col);
			}
		}
	}

	// For ALTER PAC TABLE, we only update metadata, no actual DDL execution needed
	stripped_sql = "";

	return true;
}

/**
 * ParseAlterTableDropPAC: Parses ALTER PAC TABLE ... DROP ... statement
 *
 * Syntax:
 *   ALTER PAC TABLE table_name DROP PAC LINK (col1);
 *   ALTER PAC TABLE table_name DROP PROTECTED (col1, col2);
 *
 * This operation is metadata-only (no actual DDL executed). It removes
 * metadata entries from the table's metadata.
 *
 * Validation:
 *   - Table must have existing metadata
 *   - Columns/links must exist in metadata (can't drop non-existent entries)
 */
bool PACParserExtension::ParseAlterTableDropPAC(const string &query, string &stripped_sql, PACTableMetadata &metadata) {
	string query_lower = StringUtil::Lower(query);

	// Check if this is an ALTER PAC TABLE ... DROP statement
	// Must check for specific DROP patterns (drop pac link, drop protected), not just "drop" anywhere
	if (query_lower.find("alter pac table") == string::npos) {
		return false;
	}

	// Check for DROP PAC LINK or DROP PROTECTED (not just "drop" which could be part of table name)
	bool has_drop_link = query_lower.find("drop pac link") != string::npos;
	bool has_drop_protected = query_lower.find("drop protected") != string::npos;

	if (!has_drop_link && !has_drop_protected) {
		return false;
	}

	// Extract table name - for ALTER PAC TABLE, the table name comes after "alter pac table"
	std::regex alter_pac_regex(R"(alter\s+pac\s+table\s+([a-zA-Z_][a-zA-Z0-9_]*))");
	std::smatch match;
	if (!std::regex_search(query_lower, match, alter_pac_regex)) {
		return false;
	}
	metadata.table_name = match[1].str();

	// Get existing metadata - it must exist for DROP operations
	auto existing = PACMetadataManager::Get().GetTableMetadata(metadata.table_name);
	if (!existing) {
		throw ParserException("Table '" + metadata.table_name + "' does not have any PAC metadata to drop");
	}
	metadata = *existing;

	// Handle DROP PAC LINK (col1, col2, ...)
	if (has_drop_link) {
		std::regex drop_link_regex(R"(drop\s+pac\s+link\s*\(\s*([^)]+)\s*\))", std::regex_constants::icase);
		if (std::regex_search(query_lower, match, drop_link_regex)) {
			string cols_str = match[1].str();
			auto drop_cols = StringUtil::Split(cols_str, ',');
			vector<string> normalized_drop_cols;
			for (auto &col : drop_cols) {
				StringUtil::Trim(col);
				if (!col.empty()) {
					normalized_drop_cols.push_back(StringUtil::Lower(col));
				}
			}

			// Find and remove the link with these local columns
			bool found = false;
			for (auto it = metadata.links.begin(); it != metadata.links.end(); ++it) {
				// Normalize local columns for comparison
				vector<string> normalized_local_cols;
				for (const auto &col : it->local_columns) {
					normalized_local_cols.push_back(StringUtil::Lower(col));
				}

				if (normalized_local_cols == normalized_drop_cols) {
					metadata.links.erase(it);
					found = true;
					break;
				}
			}

			if (!found) {
				if (normalized_drop_cols.size() == 1) {
					throw ParserException("No PAC LINK found on column '" + normalized_drop_cols[0] + "'");
				} else {
					string cols_list;
					for (size_t i = 0; i < normalized_drop_cols.size(); i++) {
						if (i > 0)
							cols_list += ", ";
						cols_list += "'" + normalized_drop_cols[i] + "'";
					}
					throw ParserException("No PAC LINK found on columns (" + cols_list + ")");
				}
			}
		}
	}

	// Handle DROP PROTECTED (col1, col2, ...)
	if (has_drop_protected) {
		std::regex drop_protected_regex(R"(drop\s+protected\s*\(\s*([^)]+)\s*\))", std::regex_constants::icase);
		if (std::regex_search(query_lower, match, drop_protected_regex)) {
			string cols_str = match[1].str();
			auto drop_cols = StringUtil::Split(cols_str, ',');
			for (auto &col : drop_cols) {
				StringUtil::Trim(col);
				if (col.empty())
					continue;

				// Find and remove the protected column
				bool found = false;
				for (auto it = metadata.protected_columns.begin(); it != metadata.protected_columns.end(); ++it) {
					if (StringUtil::Lower(*it) == StringUtil::Lower(col)) {
						metadata.protected_columns.erase(it);
						found = true;
						break;
					}
				}

				if (!found) {
					throw ParserException("Column '" + col + "' is not marked as protected");
				}
			}
		}
	}

	// For ALTER PAC TABLE DROP, we only update metadata, no actual DDL execution needed
	stripped_sql = "";

	return true;
}

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
					                    "table '" + pac_data.metadata.table_name + "'. No protected columns were added.");
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

					// Check referenced columns
					auto &ref_table = ref_table_entry->Cast<TableCatalogEntry>();
					ValidateColumnsExist(ref_table, link.referenced_columns, link.referenced_table,
					                    "referenced table '" + link.referenced_table + "'. PAC LINK was not added.");
				}
			} catch (const CatalogException &e) {
				throw;
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
