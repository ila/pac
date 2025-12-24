//
// Created by ila on 12/12/25.
//

#include "include/pac_optimizer.hpp"
#include <unordered_set>
#include <string>
#include <algorithm>
#include <functional>

// Include public helper to access the configured PAC tables filename and read helper
#include "include/pac_privacy_unit.hpp"
#include "include/pac_helpers.hpp"
// Include PAC compiler
#include "include/pac_compiler.hpp"
#include "include/pac_bitslice_compiler.hpp"

namespace duckdb {

// todo- optimizer rule for DROP TABLE

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

	// Run the PAC compatibility checks only if the plan is a projection or order by (i.e., a SELECT query)
	if (!plan ||
	    (plan->type != LogicalOperatorType::LOGICAL_PROJECTION && plan->type != LogicalOperatorType::LOGICAL_ORDER_BY &&
	     plan->type != LogicalOperatorType::LOGICAL_TOP_N)) {
		return;
	}
	// Load configured PAC tables once
	string pac_privacy_file = GetPacPrivacyFile(input.context);

	auto pac_tables = ReadPacTablesFile(pac_privacy_file);
	// convert unordered_set returned by ReadPacTablesFile to a vector for the compatibility check
	std::vector<std::string> pac_table_list = PacTablesSetToVector(pac_tables);

	// Delegate compatibility checks (including detecting PAC table presence and internal sample scans)
	// to PACRewriteQueryCheck. It now returns a PACCompatibilityResult with fk_paths and PKs.
	// Pass the replan_in_progress flag from the optimizer extension so the compatibility check
	// can immediately return when a replan is already in progress (avoids infinite recursion).
	bool replan_flag = false;
	if (pac_info) {
		replan_flag = pac_info->replan_in_progress.load(std::memory_order_acquire);
	}
	PACCompatibilityResult check = PACRewriteQueryCheck(*plan, input.context, pac_table_list, replan_flag);
	// If no FK paths were found and no configured PAC tables were scanned, nothing to do.
	// However, if the plan directly scans configured PAC tables (privacy units) we should still
	// proceed with compilation even when no FK paths (or PKs) were discovered.
	if (check.fk_paths.empty() && check.scanned_pac_tables.empty()) {
		return;
	}

	// Determine the set of discovered privacy units (could come from privacy_unit_pks keys or fk_paths targets)
	std::vector<std::string> discovered_pus;
	// First, privacy units for which we have PK info
	for (auto &kv : check.privacy_unit_pks) {
		discovered_pus.push_back(kv.first);
	}
	// Also consider fk_paths targets in case pk map didn't include them
	for (auto &kv : check.fk_paths) {
		if (!kv.second.empty())
			discovered_pus.push_back(kv.second.back());
	}
	// Also include any configured PAC tables that were scanned directly in the plan
	for (auto &t : check.scanned_pac_tables) {
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
	std::string normalized = NormalizeQueryForHash(input.context.GetCurrentQuery());
	std::string query_hash = HashStringToHex(normalized);
	std::string compile_method = GetPacCompileMethod(input.context, "standard");

	if (discovered_pus.size() > 1) {
		throw InvalidInputException(
		    "PAC rewrite: multiple privacy units discovered (%s); multi-privacy-unit queries are not supported");
	}

	std::vector<std::string> privacy_units = std::move(discovered_pus);

	// Print discovered PKs for diagnostics (if available) for each privacy unit
	for (auto &pu : privacy_units) {
		auto pk_it = check.privacy_unit_pks.find(pu);
		if (pk_it != check.privacy_unit_pks.end()) {
#ifdef PAC_DEBUG
			Printer::Print("Discovered primary key columns for privacy unit '" + pu + "':");
			for (const auto &col : pk_it->second) {
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
		for (auto &pu : privacy_units) {
#ifdef PAC_DEBUG
			Printer::Print("Query requires PAC Compilation for privacy unit: " + pu);
#endif

			// set replan flag for duration of compilation
			ReplanGuard scoped2(pac_info);
			compile_method = "bitslice"; // for debugging
			if (compile_method == "bitslice") {
				CompilePacBitsliceQuery(check, input, plan, pu, normalized, query_hash);
			} else {
				CompilePACQuery(input, plan, pu, normalized, query_hash);
			}
		}
	}
}

} // namespace duckdb
