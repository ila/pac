#define DUCKDB_EXTENSION_MAIN

#include "pac_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

#include <fstream>
#include <unordered_set>

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "include/pac_optimizer.hpp"
#include "include/pac_privacy_unit.hpp"
#include "include/pac_aggregate.hpp"

// planner/bound expression headers needed to inspect bind-time constant arguments and bound function info
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

inline void PacScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Pac " + name.GetString() + " 🐥");
	});
}

inline void PacOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Pac " + name.GetString() + ", my linked OpenSSL version is " +
		                                   OPENSSL_VERSION_TEXT);
	});
}

// NOTE: add/remove PAC privacy unit helpers and functions moved to src/include/pac_privacy_unit.hpp and
// src/pac_privacy_unit.cpp

static void LoadInternal(ExtensionLoader &loader) {
	// Register a scalar function
	auto pac_scalar_function = ScalarFunction("pac", {LogicalType::VARCHAR}, LogicalType::VARCHAR, PacScalarFun);
	loader.RegisterFunction(pac_scalar_function);

	// Register another scalar function
	auto pac_openssl_version_scalar_function = ScalarFunction("pac_openssl_version", {LogicalType::VARCHAR},
	                                                            LogicalType::VARCHAR, PacOpenSSLVersionScalarFun);
	loader.RegisterFunction(pac_openssl_version_scalar_function);

	// Register add_pac_privacy_unit (1-arg)
	// NOTE: scalar add/remove functions removed; prefer PRAGMA add_privacy_unit / PRAGMA remove_privacy_unit

	// Register remove_pac_privacy_unit (1-arg)
	// (removed scalar variants)

	// Register pragma variants so they can be invoked as PRAGMA add_privacy_unit(...) / PRAGMA remove_privacy_unit(...)
	auto add_privacy_unit_pragma = PragmaFunction::PragmaCall("add_pac_privacy_unit", AddPrivacyUnitPragma,
	                                                           {LogicalType::VARCHAR});
	loader.RegisterFunction(add_privacy_unit_pragma);
	auto remove_privacy_unit_pragma = PragmaFunction::PragmaCall("remove_pac_privacy_unit", RemovePrivacyUnitPragma,
	                                                              {LogicalType::VARCHAR});
	loader.RegisterFunction(remove_privacy_unit_pragma);

	// Register scalar helper to delete file (tests use this cleanup helper)
	auto delete_privacy_unit_file = ScalarFunction(
		"delete_privacy_unit_file",
		{LogicalType::VARCHAR},
		LogicalType::VARCHAR,
		DeletePrivacyUnitFileFun
	);
	loader.RegisterFunction(delete_privacy_unit_file);

	auto pac_rewrite_rule = PACRewriteRule();
	auto &db = loader.GetDatabaseInstance();
	db.config.optimizer_extensions.push_back(pac_rewrite_rule);

	db.config.AddExtensionOption("pac_privacy_file", "path for privacy units", LogicalType::VARCHAR);
	// Add option to enable/disable PAC noise application (this is useful for testing, since noise affects result determinism)
	db.config.AddExtensionOption("pac_noise", "apply PAC noise", LogicalType::BOOLEAN);
	// Add option to set deterministic RNG seed for PAC functions (useful for tests)
	db.config.AddExtensionOption("pac_seed", "deterministic RNG seed for PAC functions", LogicalType::BIGINT);

	// Register pac_aggregate function(s)
	RegisterPacAggregateFunctions(loader);

}

void PacExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string PacExtension::Name() {
	return "pac";
}

std::string PacExtension::Version() const {
#ifdef EXT_VERSION_PAC
	return EXT_VERSION_PAC;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(pac, loader) {
	duckdb::LoadInternal(loader);
}
}
