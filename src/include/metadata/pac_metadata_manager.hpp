//
// PAC Metadata Manager Header
//
// This file declares the PACMetadataManager class which manages PAC table metadata in memory.
// It provides thread-safe storage and retrieval of metadata for PAC-protected tables.
//
// Created by refactoring pac_parser.hpp on 1/22/26.
//

#ifndef PAC_METADATA_MANAGER_HPP
#define PAC_METADATA_MANAGER_HPP

#include "duckdb.hpp"
#include "duckdb/main/client_context.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace duckdb {

// Forward declarations
struct PACTableMetadata;

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

	// Get all table names with metadata
	vector<string> GetAllTableNames() const;

	// Remove metadata for a table (e.g., when table is dropped)
	void RemoveTable(const string &table_name);

	// Clear all metadata
	void Clear();

	// Get the metadata file path based on database path
	static string GetMetadataFilePath(ClientContext &context);

	// Serialization methods (implemented in pac_metadata_serialization.cpp)
	string SerializeToJSON(const PACTableMetadata &metadata) const;
	string SerializeAllToJSON() const;
	PACTableMetadata DeserializeFromJSON(const string &json);
	void DeserializeAllFromJSON(const string &json);

	// File I/O methods (implemented in pac_metadata_serialization.cpp)
	void SaveToFile(const string &filepath);
	void LoadFromFile(const string &filepath);

private:
	PACMetadataManager() = default;
	unordered_map<string, PACTableMetadata> table_metadata;
	mutable std::mutex metadata_mutex;
};

} // namespace duckdb

#endif // PAC_METADATA_MANAGER_HPP
