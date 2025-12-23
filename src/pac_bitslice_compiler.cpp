//
// Created by ila on 12/21/25.
//

#include "include/pac_bitslice_compiler.hpp"
#include "include/pac_helpers.hpp"
#include <iostream>
#include "include/pac_compiler_helpers.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

namespace duckdb {

void CompilePacBitsliceQuery(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                             unique_ptr<LogicalOperator> &plan, const std::string &privacy_unit,
                             const std::string &query, const std::string &query_hash) {
	// Bitslice compilation is intentionally left as a stub for now.
	// Implement algorithm here when ready. For now, just emit a diagnostic.
	Printer::Print("CompilePacBitsliceQuery called for PU=" + privacy_unit + " hash=" + query_hash);

	string path = GetPacCompiledPath(input.context, ".");
	if (!path.empty() && path.back() != '/') {
		path.push_back('/');
	}
	string filename = path + privacy_unit + "_" + query_hash + "_bitslice.sql";

	// The bitslice compiler works in the following way:
	// a) the query scans PU table:
	// a.1) the PU table has 1 PK: we hash it
	// a.2) the PU table has multiple PKs: we XOR them and hash the result
	// a.3) the PU table has no PK: we hash rowid
	// b) the query does not scan PU table:
	// b.1) we follow the FK path to find the PK(s) of the PU table
	// b.2) we join the chain of tables from the scanned table to the PU table
	// b.3) we hash the PK(s) as in a)

	// Example: SELECT group_key, SUM(val) AS sum_val FROM t_single GROUP BY group_key;
	// Becomes: SELECT group_key, pac_sum(HASH(rowid), val) AS sum_val FROM t_single GROUP BY group_key;
	// todo- what is MI? what is k?

	// Case a) query scans PU table (we assume only 1 PAC table for now)
	std::vector<string> pks;
	bool use_rowid = false;
	if (!check.scanned_pac_tables.empty()) {
		if (!check.privacy_unit_pks.empty()) {
			pks = check.privacy_unit_pks.at(check.scanned_pac_tables[0]);
		} else {
			use_rowid = true;
		}
	}

	// Now we need to find the PAC scan node
	// Replan the plan without compressed materialization
	ReplanWithoutOptimizers(input.context, query, plan);
	auto pac_scan_ptr = FindPrivacyUnitGetNode(plan);
	auto &pac_get = pac_scan_ptr->get()->Cast<LogicalGet>();
	Printer::Print("ok");
}

} // namespace duckdb
