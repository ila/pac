#pragma once

#include "duckdb.hpp"

namespace duckdb {

class PacExtension : public Extension {
public:
	void Load(ExtensionLoader &db) override;
	string Name() override;
	string Version() const override;
};

} // namespace duckdb
