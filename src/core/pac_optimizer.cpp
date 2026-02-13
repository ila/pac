//
// Created by ila on 12/12/25.
//

#include "core/pac_optimizer.hpp"
#include "pac_debug.hpp"
#include <string>
#include <algorithm>

// Include public helper to access the configured PAC tables filename and read helper
#include "core/pac_privacy_unit.hpp"
#include "utils/pac_helpers.hpp"
// Include PAC bitslice compiler
#include "compiler/pac_bitslice_compiler.hpp"
// Include PAC parser for metadata management
#include "parser/pac_parser.hpp"
// Include utility diff
#include "diff/pac_utility_diff.hpp"
// Include deep copy for utility diff
#include "duckdb/planner/logical_operator_deep_copy.hpp"
#include "duckdb/optimizer/optimizer.hpp"
// Include DuckDB headers for DROP operation handling
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
	if (drop_info.type != CatalogType::TABLE_ENTRY) {
		return; // Only handle DROP TABLE operations
	}
	string table_name = drop_info.name;

	// Check if this table has PAC metadata
	auto &metadata_mgr = PACMetadataManager::Get();
	if (!metadata_mgr.HasMetadata(table_name)) {
		return; // No metadata for this table, nothing to clean up
	}
#if PAC_DEBUG
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
#if PAC_DEBUG
		std::cerr << "[PAC DEBUG] Removed PAC LINKs from table '" << other_table << "' that referenced dropped table '"
		          << table_name << "'"
		          << "\n";
#endif
	}
	// Remove metadata for the dropped table
	metadata_mgr.RemoveTable(table_name);
#if PAC_DEBUG
	std::cerr << "[PAC DEBUG] Removed PAC metadata for dropped table: " << table_name << "\n";
#endif
	// Save updated metadata to file
	string metadata_path = PACMetadataManager::GetMetadataFilePath(input.context);
	if (!metadata_path.empty()) {
		metadata_mgr.SaveToFile(metadata_path);
#if PAC_DEBUG
		std::cerr << "[PAC DEBUG] Saved updated PAC metadata after DROP TABLE"
		          << "\n";
#endif
	}
}

// ============================================================================
// PACRewriteRule - Main PAC query rewriting optimizer rule
// ============================================================================

// PAC rewrites run in the pre-optimizer phase, BEFORE DuckDB's built-in optimizers.
// This way DuckDB's join ordering, filter pushdown, column lifetime, compressed
// materialization etc. all run on the PAC-transformed plan automatically.
void PACRewriteRule::PACPreOptimizeFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	if (!plan) {
		return;
	}
	// If the optimizer extension provided a PACOptimizerInfo, and a replan is already in progress,
	// skip running the PAC checks to avoid re-entrant behavior.
	PACOptimizerInfo *pac_info = nullptr;
	if (input.info) {
		pac_info = dynamic_cast<PACOptimizerInfo *>(input.info.get());
		if (pac_info && pac_info->replan_in_progress.load(std::memory_order_acquire)) {
			return;
		}
	}
	// Run the PAC compatibility checks only if the plan is a projection, order by, or aggregate (i.e., a SELECT query)
	// For EXPLAIN/EXPLAIN_ANALYZE, look at the child operator to decide whether to rewrite
	LogicalOperator *check_plan = plan.get();
	if (plan->type == LogicalOperatorType::LOGICAL_EXPLAIN && !plan->children.empty()) {
		check_plan = plan->children[0].get();
	}
	// TODO: why this particular collection of operators? Maybe check against DDL or DML
	if (check_plan->type != LogicalOperatorType::LOGICAL_PROJECTION &&
	    check_plan->type != LogicalOperatorType::LOGICAL_ORDER_BY &&
	    check_plan->type != LogicalOperatorType::LOGICAL_TOP_N &&
	    check_plan->type != LogicalOperatorType::LOGICAL_LIMIT &&
	    check_plan->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY &&
	    check_plan->type != LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
		return;
	}
	// Load configured PAC tables once
	string pac_privacy_file = GetPacPrivacyFile(input.context);
	auto pac_tables = ReadPacTablesFile(pac_privacy_file);
	vector<string> pac_table_list = PacTablesSetToVector(pac_tables);

	// For EXPLAIN queries, we need to operate on the child plan
	bool is_explain = (plan->type == LogicalOperatorType::LOGICAL_EXPLAIN && !plan->children.empty());
	unique_ptr<LogicalOperator> &target_plan = is_explain ? plan->children[0] : plan;

	// Delegate compatibility checks (including detecting PAC table presence and internal sample scans)
	PACCompatibilityResult check = PACRewriteQueryCheck(target_plan, input.context, pac_table_list, pac_info);
	if (check.fk_paths.empty() && check.scanned_pu_tables.empty()) {
		return;
	}

	// Determine the set of discovered privacy units
	vector<string> discovered_pus;
	for (auto &kv : check.fk_paths) {
		if (!kv.second.empty()) {
			discovered_pus.push_back(kv.second.back());
		}
	}
	for (auto &t : check.scanned_pu_tables) {
		discovered_pus.push_back(t);
	}
	std::sort(discovered_pus.begin(), discovered_pus.end());
	discovered_pus.erase(std::unique(discovered_pus.begin(), discovered_pus.end()), discovered_pus.end());
	if (discovered_pus.empty()) {
		return;
	}

	// compute normalized query hash once for file naming
	string normalized = NormalizeQueryForHash(input.context.GetCurrentQuery());
	string query_hash = HashStringToHex(normalized);
	vector<string> privacy_units = std::move(discovered_pus);
#if PAC_DEBUG
	for (auto &pu : privacy_units) {
		auto it = check.table_metadata.find(pu);
		if (it != check.table_metadata.end() && !it->second.pks.empty()) {
			PAC_DEBUG_PRINT("Discovered primary key columns for privacy unit '" + pu + "':");
			for (const auto &col : it->second.pks) {
				PAC_DEBUG_PRINT(col);
			}
		}
	}
#endif
	// Check if utility diff is requested — deep-copy BEFORE any compilation
	unique_ptr<LogicalOperator> ref_plan;
	idx_t num_diffcols = 0;
	string diff_output_path;
	bool do_diff = false;
	{
		Value val;
		if (input.context.TryGetCurrentSetting("pac_diffcols", val) && !val.IsNull()) {
			string val_str = val.ToString();
			auto colon = val_str.find(':');
			num_diffcols = static_cast<idx_t>(std::stoi(val_str.substr(0, colon)));
			if (colon != string::npos) {
				diff_output_path = val_str.substr(colon + 1);
			}
			do_diff = true;
			LogicalOperatorDeepCopy copier(input.optimizer.binder, nullptr);
			ref_plan = copier.DeepCopy(target_plan);
		}
	}

	if (check.eligible_for_rewrite) {
		bool apply_noise = IsPacNoiseEnabled(input.context, true);
		if (apply_noise) {
#if PAC_DEBUG
			PAC_DEBUG_PRINT("Query requires PAC Compilation for privacy units:");
			for (const auto &pu_name : privacy_units) {
				PAC_DEBUG_PRINT("  " + pu_name);
			}
#endif
			// set replan flag for duration of compilation
			ReplanGuard scoped2(pac_info);
			CompilePacBitsliceQuery(check, input, target_plan, privacy_units, normalized, query_hash);
		}
	}

	if (do_diff) {
		ApplyUtilityDiff(input, target_plan, std::move(ref_plan), num_diffcols, diff_output_path);
	}
}

} // namespace duckdb
