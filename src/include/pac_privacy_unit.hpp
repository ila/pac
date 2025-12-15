#pragma once

#include "duckdb.hpp"
#include <unordered_set>
#include <string>

namespace duckdb {

// Read/write pac tables file
DUCKDB_API std::unordered_set<std::string> ReadPacTablesFile(const std::string &filename);
DUCKDB_API void WritePacTablesFile(const std::string &filename, const std::unordered_set<std::string> &tables);

// Helper: check if table exists in the current catalog
DUCKDB_API bool TableExists(ClientContext &context, const std::string &table_name);

// Bind data and functions for add/remove PAC privacy unit scalar functions
struct PacPrivacyUnitBindData : public FunctionData {
	std::string table_name;
	PacPrivacyUnitBindData(std::string table_name);
	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other) const override;
};

DUCKDB_API unique_ptr<FunctionData> AddPacPrivacyUnitBind(ClientContext &context, ScalarFunction &function,
                                                          vector<unique_ptr<Expression>> &arguments);
DUCKDB_API void AddPacPrivacyUnitFun(DataChunk &args, ExpressionState &state, Vector &result);

DUCKDB_API unique_ptr<FunctionData> RemovePacPrivacyUnitBind(ClientContext &context, ScalarFunction &function,
                                                             vector<unique_ptr<Expression>> &arguments);
DUCKDB_API void RemovePacPrivacyUnitFun(DataChunk &args, ExpressionState &state, Vector &result);

} // namespace duckdb

