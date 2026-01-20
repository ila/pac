//
// Created by ila on 1/20/26.
//

#ifndef PAC_PARSER_HPP
#define PAC_PARSER_HPP

#include "duckdb.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/main/client_context.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>

namespace duckdb {

// Represents a PAC link (foreign key relationship without actual FK constraint)
struct PACLink {
	string local_column;
	string referenced_table;
	string referenced_column;

	PACLink() = default;
	PACLink(string local_col, string ref_table, string ref_col)
	    : local_column(std::move(local_col)), referenced_table(std::move(ref_table)),
	      referenced_column(std::move(ref_col)) {
	}
};

// PAC metadata for a single table
struct PACTableMetadata {
	string table_name;
	vector<string> primary_key_columns;
	vector<PACLink> links;
	vector<string> protected_columns;

	PACTableMetadata() = default;
	explicit PACTableMetadata(string name) : table_name(std::move(name)) {
	}
};

// Global PAC metadata manager - stores all PAC metadata in memory
class PACMetadataManager {
public:
	static PACMetadataManager &Get();

	// Add or update metadata for a table
	void AddOrUpdateTable(const string &table_name, const PACTableMetadata &metadata);

	// Get metadata for a table
	const PACTableMetadata *GetTableMetadata(const string &table_name) const;

	// Check if a table has PAC metadata
	bool HasMetadata(const string &table_name) const;

	// Save metadata to JSON file
	void SaveToFile(const string &filepath);

	// Load metadata from JSON file
	void LoadFromFile(const string &filepath);

	// Clear all metadata
	void Clear();

	// Serialize table metadata to JSON string
	static string SerializeToJSON(const PACTableMetadata &metadata);

	// Deserialize table metadata from JSON string
	static PACTableMetadata DeserializeFromJSON(const string &json);

	// Serialize all metadata to JSON
	string SerializeAllToJSON() const;

	// Deserialize all metadata from JSON
	void DeserializeAllFromJSON(const string &json);

private:
	PACMetadataManager() = default;
	unordered_map<string, PACTableMetadata> table_metadata;
	mutable std::mutex metadata_mutex;
};

// Parse data for PAC parser extension
struct PACParseData : public ParserExtensionParseData {
	string stripped_sql;
	PACTableMetadata metadata;
	bool is_pac_ddl;

	PACParseData(string sql, PACTableMetadata meta, bool is_pac)
	    : stripped_sql(std::move(sql)), metadata(std::move(meta)), is_pac_ddl(is_pac) {
	}

	unique_ptr<ParserExtensionParseData> Copy() const override {
		return make_uniq<PACParseData>(stripped_sql, metadata, is_pac_ddl);
	}

	string ToString() const override {
		return stripped_sql;
	}
};

// PAC Parser Extension - handles CREATE PAC TABLE and ALTER TABLE ... ADD PAC ...
class PACParserExtension : public ParserExtension {
public:
	PACParserExtension() {
		parse_function = PACParseFunction;
		plan_function = PACPlanFunction;
	}

	static ParserExtensionParseResult PACParseFunction(ParserExtensionInfo *info, const string &query);
	static ParserExtensionPlanResult PACPlanFunction(ParserExtensionInfo *info, ClientContext &context,
	                                                 unique_ptr<ParserExtensionParseData> parse_data);

	// Parse CREATE PAC TABLE syntax
	static bool ParseCreatePACTable(const string &query, string &stripped_sql, PACTableMetadata &metadata);

	// Parse ALTER TABLE ... ADD PAC ... syntax
	static bool ParseAlterTableAddPAC(const string &query, string &stripped_sql, PACTableMetadata &metadata);

	// Extract PAC PRIMARY KEY from CREATE statement
	static bool ExtractPACPrimaryKey(const string &clause, vector<string> &pk_columns);

	// Extract PAC LINK from statement
	static bool ExtractPACLink(const string &clause, PACLink &link);

	// Extract PROTECTED columns from statement
	static bool ExtractProtectedColumns(const string &clause, vector<string> &protected_cols);

	// Strip PAC-specific clauses from SQL
	static string StripPACClauses(const string &sql);

	// Helper to extract table name from CREATE TABLE or ALTER TABLE
	static string ExtractTableName(const string &sql, bool is_create);
};

} // namespace duckdb

#endif // PAC_PARSER_HPP
