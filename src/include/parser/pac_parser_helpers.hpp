//
// PAC Parser Helpers
//
// This file provides the implementation of PAC-specific SQL parsing helpers.
// The actual function declarations are in pac_parser.hpp as static methods
// of the PACParserExtension class.
//
// Functions implemented in pac_parser_helpers.cpp:
// - PACParserExtension::ExtractTableName
// - PACParserExtension::ExtractPACPrimaryKey
// - PACParserExtension::ExtractPACLink
// - PACParserExtension::ExtractProtectedColumns
// - PACParserExtension::StripPACClauses
// - PACParserExtension::ParseCreatePACTable
// - PACParserExtension::ParseAlterTableAddPAC
// - PACParserExtension::ParseAlterTableDropPAC
//
// Created by refactoring pac_parser.cpp on 1/22/26.
//

#ifndef PAC_PARSER_HELPERS_HPP
#define PAC_PARSER_HELPERS_HPP

#include "pac_parser.hpp"

namespace duckdb {

// All parser helper functions are declared as static methods of PACParserExtension
// in pac_parser.hpp. This header exists for organizational purposes and to allow
// pac_parser_helpers.cpp to be compiled as a separate translation unit.

} // namespace duckdb

#endif // PAC_PARSER_HELPERS_HPP
