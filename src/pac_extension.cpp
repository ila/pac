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
#include "include/pac_sum.hpp"
#include "include/pac_avg.hpp"
#include "include/pac_min_max.hpp"
#include "include/pac_categorical.hpp"
#include "include/pac_parser.hpp"

namespace duckdb {

// Pragma function to save PAC metadata to file
static void SavePACMetadataPragma(ClientContext &context, const FunctionParameters &parameters) {
	auto filepath = parameters.values[0].ToString();
	PACMetadataManager::Get().SaveToFile(filepath);
}

// Pragma function to load PAC metadata from file
static void LoadPACMetadataPragma(ClientContext &context, const FunctionParameters &parameters) {
	auto filepath = parameters.values[0].ToString();
	PACMetadataManager::Get().LoadFromFile(filepath);
}

// Pragma function to clear all PAC metadata (in-memory and file)
static void ClearPACMetadataPragma(ClientContext &context, const FunctionParameters &parameters) {
	// Clear in-memory metadata
	PACMetadataManager::Get().Clear();

	// Try to delete the metadata file if it exists
	try {
		string filepath = PACMetadataManager::GetMetadataFilePath(context);
		if (!filepath.empty()) {
			// Check if file exists
			std::ifstream file_check(filepath);
			if (file_check.good()) {
				file_check.close();
				// File exists, try to delete it
				if (std::remove(filepath.c_str()) != 0) {
					throw IOException("Failed to delete PAC metadata file: " + filepath);
				}
			}
			// If file doesn't exist, do nothing (no error)
		}
	} catch (const IOException &e) {
		// Re-throw IO exceptions (file deletion failures)
		throw;
	} catch (...) {
		// Ignore other exceptions (e.g., can't determine file path for in-memory DB)
		// This is fine - just clear in-memory metadata
	}
}

static void LoadInternal(ExtensionLoader &loader) {
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

	// Try to automatically load PAC metadata from database directory if it exists
	auto &db = loader.GetDatabaseInstance();
	try {
		// Clear any existing metadata first (in case extension is reloaded)
		PACMetadataManager::Get().Clear();

		// Get all attached database paths and try to load metadata from the first one's directory
		auto paths = db.GetDatabaseManager().GetAttachedDatabasePaths();
		if (!paths.empty()) {
			const string &db_path = paths[0];

			// Extract database name from path (filename without extension)
			// Or use "memory" as default for in-memory databases
			string db_name = "memory";
			size_t last_slash = db_path.find_last_of("/\\");
			if (last_slash != string::npos && last_slash + 1 < db_path.length()) {
				string filename = db_path.substr(last_slash + 1);
				size_t dot_pos = filename.find_last_of('.');
				if (dot_pos != string::npos) {
					db_name = filename.substr(0, dot_pos);
				} else {
					db_name = filename;
				}
			} else if (last_slash == string::npos && !db_path.empty()) {
				// No slash, so path is just the filename
				size_t dot_pos = db_path.find_last_of('.');
				if (dot_pos != string::npos) {
					db_name = db_path.substr(0, dot_pos);
				} else {
					db_name = db_path;
				}
			}

			// Default schema is "main"
			string schema_name = DEFAULT_SCHEMA;

			// Build metadata path with db and schema names
			string metadata_path;
			string filename = "pac_metadata_" + db_name + "_" + schema_name + ".json";

			if (last_slash != string::npos) {
				metadata_path = db_path.substr(0, last_slash + 1) + filename;
			} else {
				metadata_path = filename;
			}

#ifdef DEBUG
			std::cerr << "[PAC DEBUG] LoadInternal: Checking for metadata at: " << metadata_path << "\n";
#endif

			// Try to load the metadata file if it exists
			std::ifstream test_file(metadata_path);
			if (test_file.good()) {
				test_file.close();
				PACMetadataManager::Get().LoadFromFile(metadata_path);
#ifdef DEBUG
				std::cerr << "[PAC DEBUG] LoadInternal: Successfully loaded metadata from " << metadata_path << "\n";
				std::cerr << "[PAC DEBUG] LoadInternal: Loaded " << PACMetadataManager::Get().GetAllTableNames().size()
				          << " tables"
				          << "\n";
#endif
			} else {
#ifdef DEBUG
				std::cerr << "[PAC DEBUG] LoadInternal: Metadata file not found at " << metadata_path << "\n";
#endif
			}
		} else {
#ifdef DEBUG
			std::cerr << "[PAC DEBUG] LoadInternal: No database paths available (in-memory DB?)"
			          << "\n";
#endif
		}
	} catch (const std::exception &e) {
#ifdef DEBUG
		std::cerr << "[PAC DEBUG] LoadInternal: Failed to load metadata: " << e.what() << "\n";
#endif
	} catch (...) {
#ifdef DEBUG
		std::cerr << "[PAC DEBUG] LoadInternal: Failed to load metadata (unknown exception)"
		          << "\n";
#endif
	}

	// Register PAC optimizer rule
	auto pac_rewrite_rule = PACRewriteRule();
	// attach PAC-specific optimizer info so the extension can coordinate replan state
	auto pac_info = make_shared_ptr<PACOptimizerInfo>();
	pac_rewrite_rule.optimizer_info = pac_info;
	db.config.optimizer_extensions.push_back(pac_rewrite_rule);

	// Register PAC DROP TABLE cleanup rule (separate rule to handle DROP TABLE operations)
	auto pac_drop_table_rule = PACDropTableRule();
	db.config.optimizer_extensions.push_back(pac_drop_table_rule);

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
	// Add option to set the maximum influence (mi) parameter for PAC aggregates
	// No default value - when not set, function argument or hardcoded default (128.0) is used
	// When explicitly set by user, it overrides any function argument
	db.config.AddExtensionOption("pac_mi", "mutual information (mi) parameter for PAC aggregates", LogicalType::DOUBLE);

	// Register pac_aggregate function(s)
	RegisterPacAggregateFunctions(loader);
	// Register pac_sum/pac_avg aggregate functions (moved to pac_sum_avg.cpp)
	RegisterPacSumFunctions(loader);
	RegisterPacSumCountersFunctions(loader);
	RegisterPacAvgFunctions(loader);
	RegisterPacAvgCountersFunctions(loader);
	RegisterPacCountFunctions(loader);
	RegisterPacCountCountersFunctions(loader);
	// Register pac_min/pac_max aggregate functions
	RegisterPacMinFunctions(loader);
	RegisterPacMaxFunctions(loader);
	// Register _counters variants for categorical queries
	RegisterPacMinCountersFunctions(loader);
	RegisterPacMaxCountersFunctions(loader);

	// Register PAC categorical functions (pac_select, pac_filter, pac_mask_and, etc.)
	RegisterPacCategoricalFunctions(loader);

	// Register PAC parser extension
	db.config.parser_extensions.push_back(PACParserExtension());

	// Register PAC metadata management pragmas
	auto save_pac_metadata_pragma =
	    PragmaFunction::PragmaCall("save_pac_metadata", SavePACMetadataPragma, {LogicalType::VARCHAR});
	loader.RegisterFunction(save_pac_metadata_pragma);

	auto load_pac_metadata_pragma =
	    PragmaFunction::PragmaCall("load_pac_metadata", LoadPACMetadataPragma, {LogicalType::VARCHAR});
	loader.RegisterFunction(load_pac_metadata_pragma);

	auto clear_pac_metadata_pragma = PragmaFunction::PragmaCall("clear_pac_metadata", ClearPACMetadataPragma, {});
	loader.RegisterFunction(clear_pac_metadata_pragma);
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
