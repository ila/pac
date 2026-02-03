//
// Created by ila on 12/12/25.
//

#include "core/pac_optimizer.hpp"
#include <unordered_set>
#include <string>
#include <algorithm>

// Include public helper to access the configured PAC tables filename and read helper
#include "core/pac_privacy_unit.hpp"
#include "utils/pac_helpers.hpp"
// Include PAC bitslice compiler
#include "compiler/pac_bitslice_compiler.hpp"
// Include PAC parser for metadata management
#include "parser/pac_parser.hpp"
// Include DuckDB headers for DROP operation handling
#include "duckdb/execution/operator/schema/physical_drop.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/planner/operator/logical_simple.hpp"
// Include DuckDB optimizer headers for deferred optimizers
#include "duckdb/optimizer/column_lifetime_analyzer.hpp"
#include "duckdb/optimizer/compressed_materialization.hpp"
#include "duckdb/optimizer/optimizer.hpp"

namespace duckdb {

// ============================================================================
// RAII Guard to ensure optimizer rules are always re-enabled
// ============================================================================
// This guard ensures that COLUMN_LIFETIME and COMPRESSED_MATERIALIZATION
// are re-enabled even if an exception is thrown or an early return occurs.
class OptimizerRestoreGuard {
public:
	OptimizerRestoreGuard(ClientContext &context, PACOptimizerInfo *pac_info)
	    : context(context), pac_info(pac_info), restored(false) {
		if (pac_info) {
			std::lock_guard<std::mutex> lock(pac_info->optimizer_mutex);
			auto &config = DBConfig::GetConfig(context);

			// Restore optimizers that were disabled in pre-optimize
			if (pac_info->disabled_column_lifetime) {
				config.options.disabled_optimizers.erase(OptimizerType::COLUMN_LIFETIME);
				pac_info->disabled_column_lifetime = false;
				we_disabled_optimizers = true;
			}
			if (pac_info->disabled_compressed_materialization) {
				config.options.disabled_optimizers.erase(OptimizerType::COMPRESSED_MATERIALIZATION);
				pac_info->disabled_compressed_materialization = false;
				we_disabled_optimizers = true;
			}
			restored = true;
		}
	}

	~OptimizerRestoreGuard() {
		// Nothing to do - restoration happens in constructor
	}

	bool WeDisabledOptimizers() const {
		return we_disabled_optimizers;
	}

private:
	ClientContext &context;
	PACOptimizerInfo *pac_info;
	bool restored;
	bool we_disabled_optimizers = false;
};

// ============================================================================
// Helper function to run COLUMN_LIFETIME and COMPRESSED_MATERIALIZATION
// ============================================================================
// These optimizers are disabled in PACPreOptimizeFunction and must be run
// after PAC processing (whether PAC compilation happened or not).
static void RunDeferredOptimizers(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                  bool pac_compiled) {
	if (!plan) {
		return;
	}

#ifdef DEBUG
	if (pac_compiled) {
		Printer::Print("=== PLAN AFTER PAC COMPILATION (before deferred optimizers) ===");
		plan->Print();
	}
#endif

	// Run column lifetime analyzer
	ColumnLifetimeAnalyzer column_lifetime(input.optimizer, *plan, true);
	column_lifetime.VisitOperator(*plan);

	// Run compressed materialization (if not disabled by user)
	if (!input.optimizer.OptimizerDisabled(OptimizerType::COMPRESSED_MATERIALIZATION)) {
		statistics_map_t statistics_map;
		CompressedMaterialization compressed_materialization(input.optimizer, *plan, statistics_map);
		compressed_materialization.Compress(plan);
	}

	// Resolve operator types after running optimizers
	plan->ResolveOperatorTypes();

#ifdef DEBUG
	if (pac_compiled) {
		Printer::Print("=== FINAL PLAN (after deferred optimizers) ===");
		plan->Print();
	}
#endif
}

// ============================================================================
// PACDropTableRule - Separate optimizer rule for DROP TABLE operations
// ============================================================================

void PACDropTableRule::PACDropTableRuleFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	if (!plan || plan->type != LogicalOperatorType::LOGICAL_DROP) {
		return;
	}

	// Cast to LogicalSimple to access the parse info (DROP operations use LogicalSimple)
	auto &simple = plan->Cast<LogicalSimple>();
	if (simple.info->info_type != ParseInfoType::DROP_INFO) {
		return;
	}

	// Cast to DropInfo to access drop details
	auto &drop_info = simple.info->Cast<DropInfo>();

	// Only handle DROP TABLE operations
	if (drop_info.type != CatalogType::TABLE_ENTRY) {
		return;
	}

	string table_name = drop_info.name;

	// Check if this table has PAC metadata
	auto &metadata_mgr = PACMetadataManager::Get();
	if (!metadata_mgr.HasMetadata(table_name)) {
		// No metadata for this table, nothing to clean up
		return;
	}

#ifdef DEBUG
	std::cerr << "[PAC DEBUG] DROP TABLE detected for table with PAC metadata: " << table_name << "\n";
#endif

	// Check if any other tables have PAC LINKs pointing to this table
	auto all_tables = metadata_mgr.GetAllTableNames();
	vector<string> tables_with_links_to_dropped;

	for (const auto &other_table : all_tables) {
		if (StringUtil::Lower(other_table) == StringUtil::Lower(table_name)) {
			continue; // Skip the table being dropped
		}

		auto other_metadata = metadata_mgr.GetTableMetadata(other_table);
		if (!other_metadata) {
			continue;
		}

		// Check if this table has any links to the table being dropped
		for (const auto &link : other_metadata->links) {
			if (StringUtil::Lower(link.referenced_table) == StringUtil::Lower(table_name)) {
				tables_with_links_to_dropped.push_back(other_table);
				break;
			}
		}
	}

	// Remove links from other tables that reference the dropped table
	for (const auto &other_table : tables_with_links_to_dropped) {
		auto other_metadata = metadata_mgr.GetTableMetadata(other_table);
		if (!other_metadata) {
			continue;
		}

		// Make a copy and remove links
		PACTableMetadata updated_metadata = *other_metadata;
		updated_metadata.links.erase(std::remove_if(updated_metadata.links.begin(), updated_metadata.links.end(),
		                                            [&table_name](const PACLink &link) {
			                                            return StringUtil::Lower(link.referenced_table) ==
			                                                   StringUtil::Lower(table_name);
		                                            }),
		                             updated_metadata.links.end());

		// Update the metadata
		metadata_mgr.AddOrUpdateTable(other_table, updated_metadata);

#ifdef DEBUG
		std::cerr << "[PAC DEBUG] Removed PAC LINKs from table '" << other_table << "' that referenced dropped table '"
		          << table_name << "'"
		          << "\n";
#endif
	}

	// Remove metadata for the dropped table
	metadata_mgr.RemoveTable(table_name);

#ifdef DEBUG
	std::cerr << "[PAC DEBUG] Removed PAC metadata for dropped table: " << table_name << "\n";
#endif

	// Save updated metadata to file
	string metadata_path = PACMetadataManager::GetMetadataFilePath(input.context);
	if (!metadata_path.empty()) {
		metadata_mgr.SaveToFile(metadata_path);
#ifdef DEBUG
		std::cerr << "[PAC DEBUG] Saved updated PAC metadata after DROP TABLE"
		          << "\n";
#endif
	}
}

// ============================================================================
// PACRewriteRule - Main PAC query rewriting optimizer rule
// ============================================================================

// Pre-optimizer: disables COLUMN_LIFETIME and COMPRESSED_MATERIALIZATION
// This runs BEFORE built-in optimizers, ensuring they skip these two optimizers.
// We'll run them ourselves in the post-optimizer after PAC transformation (if needed).
void PACRewriteRule::PACPreOptimizeFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	if (!plan) {
		return;
	}

	// Get the PAC optimizer info
	PACOptimizerInfo *pac_info = nullptr;
	if (input.info) {
		pac_info = dynamic_cast<PACOptimizerInfo *>(input.info.get());
	}

	if (!pac_info) {
		return;
	}

	// Skip if a replan is already in progress (avoid infinite recursion)
	if (pac_info->replan_in_progress.load(std::memory_order_acquire)) {
		return;
	}

	// Disable COLUMN_LIFETIME and COMPRESSED_MATERIALIZATION for the built-in optimizer pass
	// We'll run them ourselves in the post-optimizer
	std::lock_guard<std::mutex> lock(pac_info->optimizer_mutex);
	auto &config = DBConfig::GetConfig(input.context);

	// Check if they're already disabled (don't double-disable)
	if (config.options.disabled_optimizers.find(OptimizerType::COLUMN_LIFETIME) ==
	    config.options.disabled_optimizers.end()) {
		config.options.disabled_optimizers.insert(OptimizerType::COLUMN_LIFETIME);
		pac_info->disabled_column_lifetime = true;
	}

	if (config.options.disabled_optimizers.find(OptimizerType::COMPRESSED_MATERIALIZATION) ==
	    config.options.disabled_optimizers.end()) {
		config.options.disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
		pac_info->disabled_compressed_materialization = true;
	}
}

void PACRewriteRule::PACRewriteRuleFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {

	// If the optimizer extension provided a PACOptimizerInfo, and a replan is already in progress,
	// skip running the PAC checks to avoid re-entrant behavior.
	PACOptimizerInfo *pac_info = nullptr;
	if (input.info) {
		pac_info = dynamic_cast<PACOptimizerInfo *>(input.info.get());
		if (pac_info && pac_info->replan_in_progress.load(std::memory_order_acquire)) {
			// A replan/compilation is in progress; bypass PACRewriteRule to avoid recursion
			return;
		}
	}

	// Use RAII guard to ensure optimizers are always re-enabled, even on exceptions or early returns
	// The guard restores the optimizers immediately in its constructor
	OptimizerRestoreGuard restore_guard(input.context, pac_info);
	bool we_disabled_optimizers = restore_guard.WeDisabledOptimizers();

	// Run the PAC compatibility checks only if the plan is a projection, order by, or aggregate (i.e., a SELECT query)
	// For EXPLAIN/EXPLAIN_ANALYZE, look at the child operator to decide whether to rewrite
	if (!plan) {
		return;
	}

	// Check if this is an EXPLAIN node - if so, we'll process its child
	LogicalOperator *check_plan = plan.get();
	if (plan->type == LogicalOperatorType::LOGICAL_EXPLAIN && !plan->children.empty()) {
		check_plan = plan->children[0].get();
	}

	if (check_plan->type != LogicalOperatorType::LOGICAL_PROJECTION &&
	    check_plan->type != LogicalOperatorType::LOGICAL_ORDER_BY &&
	    check_plan->type != LogicalOperatorType::LOGICAL_TOP_N &&
	    check_plan->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY &&
	    check_plan->type != LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
		// Not a SELECT-like query, but still need to run deferred optimizers if we disabled them
		if (we_disabled_optimizers) {
			RunDeferredOptimizers(input, plan, false);
		}
		return;
	}
	// Load configured PAC tables once
	string pac_privacy_file = GetPacPrivacyFile(input.context);

	auto pac_tables = ReadPacTablesFile(pac_privacy_file);
	// convert unordered_set returned by ReadPacTablesFile to a vector for the compatibility check
	vector<string> pac_table_list = PacTablesSetToVector(pac_tables);

	// For EXPLAIN queries, we need to operate on the child plan
	bool is_explain = (plan->type == LogicalOperatorType::LOGICAL_EXPLAIN && !plan->children.empty());
	unique_ptr<LogicalOperator> &target_plan = is_explain ? plan->children[0] : plan;

	// Delegate compatibility checks (including detecting PAC table presence and internal sample scans)
	// to PACRewriteQueryCheck. It now returns a PACCompatibilityResult with fk_paths and PKs.
	PACCompatibilityResult check = PACRewriteQueryCheck(target_plan, input.context, pac_table_list, pac_info);
	// If no FK paths were found and no configured PAC tables were scanned, nothing to do.
	// However, if the plan directly scans configured PAC tables (privacy units) we should still
	// proceed with compilation even when no FK paths (or PKs) were discovered.
	// Note: Tables with protected columns are now treated as implicit privacy units and included
	// in fk_paths automatically.
	if (check.fk_paths.empty() && check.scanned_pu_tables.empty()) {
		// No PAC tables involved, but still need to run deferred optimizers if we disabled them
		if (we_disabled_optimizers) {
			RunDeferredOptimizers(input, target_plan, false);
		}
		return;
	}

	// Determine the set of discovered privacy units (could come from fk_paths targets, scanned PAC tables,
	// or tables with protected columns which are now treated as implicit privacy units)
	vector<string> discovered_pus;
	// Consider fk_paths targets
	for (auto &kv : check.fk_paths) {
		if (!kv.second.empty()) {
			discovered_pus.push_back(kv.second.back());
		}
	}
	// Also include any configured PAC tables that were scanned directly in the plan
	for (auto &t : check.scanned_pu_tables) {
		discovered_pus.push_back(t);
	}
	// Deduplicate
	std::sort(discovered_pus.begin(), discovered_pus.end());
	discovered_pus.erase(std::unique(discovered_pus.begin(), discovered_pus.end()), discovered_pus.end());
	if (discovered_pus.empty()) {
		// Defensive: nothing discovered, but still need to run deferred optimizers
		if (we_disabled_optimizers) {
			RunDeferredOptimizers(input, target_plan, false);
		}
		return;
	}

	// compute normalized query hash once for file naming
	string normalized = NormalizeQueryForHash(input.context.GetCurrentQuery());
	string query_hash = HashStringToHex(normalized);

	vector<string> privacy_units = std::move(discovered_pus);

	// Print discovered PKs for diagnostics (if available) for each privacy unit
	for (auto &pu : privacy_units) {
		auto it = check.table_metadata.find(pu);
		if (it != check.table_metadata.end() && !it->second.pks.empty()) {
#ifdef DEBUG
			Printer::Print("Discovered primary key columns for privacy unit '" + pu + "':");
			for (const auto &col : it->second.pks) {
				Printer::Print(col);
			}
#endif
		}
	}

	// Only proceed with compilation if plan passed structural eligibility
	if (!check.eligible_for_rewrite) {
		// Not eligible for PAC compilation, but still need to run deferred optimizers
		if (we_disabled_optimizers) {
			RunDeferredOptimizers(input, target_plan, false);
		}
		return;
	}

	bool apply_noise = IsPacNoiseEnabled(input.context, true);
	bool pac_compiled = false;
	if (apply_noise) {
#ifdef DEBUG
		Printer::Print("Query requires PAC Compilation for privacy units:");
		for (auto &pu : privacy_units) {
			Printer::Print("  " + pu);
		}
#endif

		// set replan flag for duration of compilation
		ReplanGuard scoped2(pac_info);
		// Call the compiler once with all privacy units
		CompilePacBitsliceQuery(check, input, target_plan, privacy_units, normalized, query_hash);
		pac_compiled = true;
	}

	// Run the deferred optimizers (COLUMN_LIFETIME and COMPRESSED_MATERIALIZATION)
	// This must happen for ALL queries since we disabled these optimizers in pre-optimize
	if (we_disabled_optimizers) {
		RunDeferredOptimizers(input, target_plan, pac_compiled);
	}
}

} // namespace duckdb
