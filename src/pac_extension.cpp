#define DUCKDB_EXTENSION_MAIN

#include "pac_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include <fstream>
#include <unordered_set>

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/common/types.hpp"
#include "include/pac_optimizer.hpp"
#include "include/pac_privacy_unit.hpp"
#include "include/pac_aggregate.hpp"
#include "include/pac_count.hpp"
#include "include/pac_sum_avg.hpp"
#include "include/pac_min_max.hpp"

namespace duckdb {

inline void PacScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Pac " + name.GetString() + " 🐥");
	});
}

// NOTE: add/remove PAC privacy unit helpers and functions moved to src/include/pac_privacy_unit.hpp and
// src/pac_privacy_unit.cpp

static void LoadInternal(ExtensionLoader &loader) {
	// Register a scalar function
	auto pac_scalar_function = ScalarFunction("pac", {LogicalType::VARCHAR}, LogicalType::VARCHAR, PacScalarFun);
	loader.RegisterFunction(pac_scalar_function);

	// Register add_pac_privacy_unit (1-arg)
	// NOTE: scalar add/remove functions removed; prefer PRAGMA add_privacy_unit / PRAGMA remove_privacy_unit

	// Register remove_pac_privacy_unit (1-arg)
	// (removed scalar variants)

	// Register pragma variants so they can be invoked as PRAGMA add_privacy_unit(...) / PRAGMA remove_privacy_unit(...)
	auto add_privacy_unit_pragma =
	    PragmaFunction::PragmaCall("add_pac_privacy_unit", AddPrivacyUnitPragma, {LogicalType::VARCHAR});
	loader.RegisterFunction(add_privacy_unit_pragma);
	auto remove_privacy_unit_pragma =
	    PragmaFunction::PragmaCall("remove_pac_privacy_unit", RemovePrivacyUnitPragma, {LogicalType::VARCHAR});
	loader.RegisterFunction(remove_privacy_unit_pragma);

	// Register scalar helper to delete file (tests use this cleanup helper)
	auto delete_privacy_unit_file = ScalarFunction("delete_privacy_unit_file", {LogicalType::VARCHAR},
	                                               LogicalType::VARCHAR, DeletePrivacyUnitFileFun);
	loader.RegisterFunction(delete_privacy_unit_file);

	auto pac_rewrite_rule = PACRewriteRule();
	// attach PAC-specific optimizer info so the extension can coordinate replan state
	auto pac_info = make_shared_ptr<PACOptimizerInfo>();
	pac_rewrite_rule.optimizer_info = pac_info;
	auto &db = loader.GetDatabaseInstance();
	db.config.optimizer_extensions.push_back(pac_rewrite_rule);

	db.config.AddExtensionOption("pac_privacy_file", "path for privacy units", LogicalType::VARCHAR);
	// Add option to enable/disable PAC noise application (this is useful for testing, since noise affects result
	// determinism)
	db.config.AddExtensionOption("pac_noise", "apply PAC noise", LogicalType::BOOLEAN);
	// Add option to set deterministic RNG seed for PAC functions (useful for tests)
	db.config.AddExtensionOption("pac_seed", "deterministic RNG seed for PAC functions", LogicalType::BIGINT);
	// Add option to force deterministic (architecture-agnostic) noise generation for testing (default false)
	db.config.AddExtensionOption("pac_deterministic_noise", "use architecture-agnostic noise generation for testing",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(false));
	// Add option to configure the number of samples (m) used by PAC (default 128)
	db.config.AddExtensionOption("pac_m", "number of per-sample subsets (m)", LogicalType::INTEGER);
	// Add option to toggle enforcement of per-sample array length == pac_m (default true)
	db.config.AddExtensionOption("enforce_m_values", "enforce per-sample arrays length equals pac_m",
	                             LogicalType::BOOLEAN);
	// Add option to set path where compiled PAC artifacts (CTEs) are written
	db.config.AddExtensionOption("pac_compiled_path", "path to write compiled PAC artifacts", LogicalType::VARCHAR);
	// Add option to enable/disable join elimination (stop FK chain before reaching PU)
	db.config.AddExtensionOption("pac_join_elimination", "eliminate final join to PU table", LogicalType::BOOLEAN,
	                             Value::BOOLEAN(true));
	// Add option to enable/disable conservative mode (when false, unsupported operators skip PAC compilation)
	db.config.AddExtensionOption("pac_conservative_mode",
	                             "throw errors for unsupported operators (when false, skip PAC compilation)",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(true));

	// Register pac_aggregate function(s)
	RegisterPacAggregateFunctions(loader);
	// Register pac_sum/pac_avg aggregate functions (moved to pac_sum_avg.cpp)
	RegisterPacSumFunctions(loader);
	RegisterPacAvgFunctions(loader);
	// Register pac_sum_approx/pac_avg_approx aggregate functions (approximate with 16-bit counters)
	RegisterPacSumApproxFunctions(loader);
	RegisterPacAvgApproxFunctions(loader);
	RegisterPacCountFunctions(loader);
	// Register pac_min/pac_max aggregate functions
	RegisterPacMinFunctions(loader);
	RegisterPacMaxFunctions(loader);
}

void PacExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
string PacExtension::Name() {
	return "pac";
}

string PacExtension::Version() const {
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
