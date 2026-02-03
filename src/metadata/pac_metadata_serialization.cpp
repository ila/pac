//
// PAC Metadata Serialization
//
// This file implements JSON serialization and deserialization for PAC metadata.
// It handles converting PACTableMetadata structures to/from JSON format and
// provides file I/O operations for persisting metadata across database sessions.
//
// Created by refactoring pac_parser.cpp on 1/22/26.
//

// IMPORTANT: <regex> must be included BEFORE duckdb headers on Windows MSVC
// because DuckDB defines its own std namespace that conflicts with <regex>
#include <regex>
#include <fstream>
#include <sstream>

#include "parser/pac_parser.hpp"
#include "metadata/pac_metadata_manager.hpp"
#include "metadata/pac_metadata_serialization.hpp"

namespace duckdb {

// ============================================================================
// JSON Serialization
// ============================================================================

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
		if (i > 0) {
			ss << ", ";
		}
		ss << "\"" << metadata.primary_key_columns[i] << "\"";
	}
	ss << "],\n";

	// Serialize links (now with support for composite keys)
	ss << "  \"links\": [\n";
	for (size_t i = 0; i < metadata.links.size(); i++) {
		if (i > 0) {
			ss << ",\n";
		}
		const auto &link = metadata.links[i];
		ss << "    {\n";

		// Serialize local_columns array
		ss << "      \"local_columns\": [";
		for (size_t j = 0; j < link.local_columns.size(); j++) {
			if (j > 0) {
				ss << ", ";
			}
			ss << "\"" << link.local_columns[j] << "\"";
		}
		ss << "],\n";

		ss << "      \"referenced_table\": \"" << link.referenced_table << "\",\n";

		// Serialize referenced_columns array
		ss << "      \"referenced_columns\": [";
		for (size_t j = 0; j < link.referenced_columns.size(); j++) {
			if (j > 0) {
				ss << ", ";
			}
			ss << "\"" << link.referenced_columns[j] << "\"";
		}
		ss << "]\n";

		ss << "    }";
	}
	ss << "\n  ],\n";

	// Serialize protected columns
	ss << "  \"protected_columns\": [";
	for (size_t i = 0; i < metadata.protected_columns.size(); i++) {
		if (i > 0) {
			ss << ", ";
		}
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
		if (idx > 0) {
			ss << ",\n";
		}
		ss << "    " << SerializeToJSON(entry.second);
		idx++;
	}

	ss << "\n  ]\n}";
	return ss.str();
}

// ============================================================================
// File I/O
// ============================================================================

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

// ============================================================================
// JSON Deserialization
// ============================================================================

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
					auto cols_begin = std::sregex_iterator(ref_cols_str.begin(), ref_cols_str.end(), col_regex);
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

} // namespace duckdb
