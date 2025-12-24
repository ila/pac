//
// Created by ila on 12/24/25.
//

#include <iostream>
#include "include/test_compiler_functions.hpp"
#include "include/test_privacy_columns.hpp"

int main() {
	std::cerr << "Starting unified test runner...\n";
	int code = 0;

	std::cerr << "Running compiler function tests...\n";
	code = duckdb::RunCompilerFunctionTests();
	if (code != 0) {
		std::cerr << "RunCompilerFunctionTests failed with code " << code << "\n";
		return code;
	}

	std::cerr << "Running privacy columns tests...\n";
	code = duckdb::RunPrivacyColumnsTests();
	if (code != 0) {
		std::cerr << "RunPrivacyColumnsTests failed with code " << code << "\n";
		return code;
	}

	std::cerr << "All tests passed\n";
	return 0;
}
