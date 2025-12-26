//
// Created by ila on 12/24/25.
//

#include <iostream>
#include "include/test_compiler_functions.hpp"
#include "include/test_privacy_columns.hpp"

int main() {
	std::cerr << "Starting unified test runner...\n";

	std::cerr << "Running compiler function tests...\n";
	int code = duckdb::RunCompilerFunctionTests();
	if (code != 0) {
		std::cerr << "RunCompilerFunctionTests failed with code " << code << "\n";
	}

	std::cerr << "\nRunning privacy columns tests...\n";
	code = duckdb::RunPrivacyColumnsTests();
	if (code != 0) {
		std::cerr << "RunPrivacyColumnsTests failed with code " << code << "\n";
	}

	return 0;
}
