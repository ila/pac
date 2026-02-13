//
// PAC Metadata Serialization
//
// This file provides the implementation of JSON serialization and deserialization
// for PAC metadata. The actual function declarations are in pac_metadata_manager.hpp
// as methods of the PACMetadataManager class.
//
// Functions implemented in pac_metadata_serialization.cpp:
// - PACMetadataManager::SerializeToJSON
// - PACMetadataManager::SerializeAllToJSON
// - PACMetadataManager::DeserializeFromJSON
// - PACMetadataManager::DeserializeAllFromJSON
// - PACMetadataManager::SaveToFile
// - PACMetadataManager::LoadFromFile
//
// Created by refactoring pac_parser.cpp on 1/22/26.
//

#ifndef PAC_METADATA_SERIALIZATION_HPP
#define PAC_METADATA_SERIALIZATION_HPP

#include "pac_metadata_manager.hpp"

namespace duckdb {

// All serialization functions are declared as methods of PACMetadataManager
// in pac_metadata_manager.hpp. This header exists for organizational purposes
// and to allow pac_metadata_serialization.cpp to be compiled as a separate
// translation unit.

} // namespace duckdb

#endif // PAC_METADATA_SERIALIZATION_HPP
