//
// Created by ila on 12/24/25.
//

#include <iostream>
#include "include/test_compiler_functions.hpp"
#include "include/test_schema_metadata.hpp"
#include "include/test_plan_traversal.hpp"
#include "include/test_pac_parser.hpp"

int main() {
	std::cerr << "===========================================\n";
	std::cerr << "Starting PAC Extension Test Suite\n";
	std::cerr << "===========================================\n";

	int total_failures = 0;

	// Test 1: Compiler Functions (ReplaceNode, column binding verification)
	std::cerr << "\n[1/4] Running compiler function tests...\n";
	int code = duckdb::RunCompilerFunctionTests();
	if (code != 0) {
		std::cerr << "❌ RunCompilerFunctionTests failed with code " << code << "\n";
		total_failures++;
	} else {
		std::cerr << "✓ Compiler function tests passed\n";
	}

	// Test 2: Schema Metadata (FindPrimaryKey, FindForeignKeys, FindForeignKeyBetween)
	std::cerr << "\n[2/4] Running schema metadata tests...\n";
	code = duckdb::RunSchemaMetadataTests();
	if (code != 0) {
		std::cerr << "❌ RunSchemaMetadataTests failed with code " << code << "\n";
		total_failures++;
	} else {
		std::cerr << "✓ Schema metadata tests passed\n";
	}

	// Test 3: Plan Traversal (FindPrivacyUnitGetNode, FindTopAggregate, etc.)
	std::cerr << "\n[3/4] Running plan traversal tests...\n";
	code = duckdb::RunPlanTraversalTests();
	if (code != 0) {
		std::cerr << "❌ RunPlanTraversalTests failed with code " << code << "\n";
		total_failures++;
	} else {
		std::cerr << "✓ Plan traversal tests passed\n";
	}

	// Test 4: PAC Parser (CREATE PAC TABLE, ALTER PAC TABLE, metadata management)
	std::cerr << "\n[4/4] Running PAC parser tests...\n";
	try {
		duckdb::TestPACParser::RunAllTests();
		std::cerr << "✓ PAC parser tests passed\n";
	} catch (const std::exception &e) {
		std::cerr << "❌ PAC parser tests failed: " << e.what() << "\n";
		total_failures++;
	}

	// Summary
	std::cerr << "\n===========================================\n";
	if (total_failures == 0) {
		std::cerr << "✅ ALL TEST SUITES PASSED!\n";
	} else {
		std::cerr << "❌ " << total_failures << " TEST SUITE(S) FAILED\n";
	}
	std::cerr << "===========================================\n";

	return total_failures;
}
