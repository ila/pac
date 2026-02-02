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

namespace duckdb {

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

void PACRewriteRule::PACRewriteRuleFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {

	// If the optimizer extension provided a PACOptimizerInfo, and a replan is already in progress,
	// skip running the PAC checks to avoid re-entrant behavior. We reuse the existing
	// `replan_in_progress` flag for this purpose.
	PACOptimizerInfo *pac_info = nullptr;
	if (input.info) {
		pac_info = dynamic_cast<PACOptimizerInfo *>(input.info.get());
		if (pac_info && pac_info->replan_in_progress.load(std::memory_order_acquire)) {
			// A replan/compilation is in progress; bypass PACRewriteRule to avoid recursion
			return;
		}
	}

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
		// Defensive: nothing discovered
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
		return;
	}

	bool apply_noise = IsPacNoiseEnabled(input.context, true);
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
	}
}

} // namespace duckdb
