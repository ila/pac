#include "utils/pac_helpers.hpp"

#include <sstream>
#include <functional>
#include <cctype>
#include <duckdb/optimizer/column_binding_replacer.hpp>
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_set_operation.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"
#include "duckdb/planner/operator/logical_column_data_get.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_cte.hpp"
#include "duckdb/planner/operator/logical_recursive_cte.hpp"
#include "duckdb/planner/operator/logical_expression_get.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <queue>

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "duckdb/parser/constraints/foreign_key_constraint.hpp"
#include "core/pac_optimizer.hpp"
#include "parser/pac_parser.hpp"

using idx_set = std::unordered_set<idx_t>;

namespace duckdb {

string Sanitize(const string &in) {
	string out;
	for (char c : in) {
		out.push_back(std::isalnum(static_cast<unsigned char>(c)) || c == '_' ? c : '_');
	}
	return out;
}

string NormalizeQueryForHash(const string &query) {
	string s = query;
	std::replace(s.begin(), s.end(), '\n', ' ');
	std::replace(s.begin(), s.end(), '\r', ' ');
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
	string out;
	out.reserve(s.size());
	bool in_space = false;
	for (char c : s) {
		if (std::isspace(static_cast<unsigned char>(c))) {
			if (!in_space) {
				out.push_back(' ');
				in_space = true;
			}
		} else {
			out.push_back(c);
			in_space = false;
		}
	}
	// trim
	if (!out.empty() && out.front() == ' ') {
		out.erase(out.begin());
	}
	if (!out.empty() && out.back() == ' ') {
		out.pop_back();
	}
	return out;
}

string HashStringToHex(const string &input) {
	size_t h = std::hash<string> {}(input);
	std::ostringstream ss;
	ss << std::hex << h;
	return ss.str();
}

idx_t GetNextTableIndex(unique_ptr<LogicalOperator> &plan) {
	idx_t max_index = DConstants::INVALID_INDEX;
	vector<unique_ptr<LogicalOperator> *> stack;
	stack.push_back(&plan);
	while (!stack.empty()) {
		auto cur_ptr = stack.back();
		stack.pop_back();
		auto &cur = *cur_ptr;
		if (!cur) {
			continue;
		}
		auto tbls = cur->GetTableIndex();
		for (auto t : tbls) {
			if (t != DConstants::INVALID_INDEX && (max_index == DConstants::INVALID_INDEX || t > max_index)) {
				max_index = t;
			}
		}
		for (auto &c : cur->children) {
			stack.push_back(&c);
		}
	}
	return (max_index == DConstants::INVALID_INDEX) ? 0 : (max_index + 1);
}

static void CollectTableIndicesRecursive(LogicalOperator *node, idx_set &out) {
	if (!node) {
		return;
	}
	auto tbls = node->GetTableIndex();
	for (auto t : tbls) {
		if (t != DConstants::INVALID_INDEX) {
			out.insert(t);
		}
	}
	for (auto &c : node->children) {
		CollectTableIndicesRecursive(c.get(), out);
	}
}

static void CollectTableIndicesExcluding(LogicalOperator *node, LogicalOperator *skip, idx_set &out) {
	if (!node) {
		return;
	}
	if (node == skip) {
		return; // skip this subtree entirely
	}
	auto tbls = node->GetTableIndex();
	for (auto t : tbls) {
		if (t != DConstants::INVALID_INDEX) {
			out.insert(t);
		}
	}
	for (auto &c : node->children) {
		CollectTableIndicesExcluding(c.get(), skip, out);
	}
}

static void ApplyIndexMapToSubtree(LogicalOperator *node, const std::unordered_map<idx_t, idx_t> &map) {
	if (!node) {
		return;
	}
	// Update operator-specific index fields where applicable
	if (auto get_ptr = dynamic_cast<LogicalGet *>(node)) {
		if (map.find(get_ptr->table_index) != map.end()) {
			get_ptr->table_index = map.at(get_ptr->table_index);
		}
	} else if (auto proj_ptr = dynamic_cast<LogicalProjection *>(node)) {
		if (map.find(proj_ptr->table_index) != map.end()) {
			proj_ptr->table_index = map.at(proj_ptr->table_index);
		}
	} else if (auto setop_ptr = dynamic_cast<LogicalSetOperation *>(node)) {
		if (map.find(setop_ptr->table_index) != map.end()) {
			setop_ptr->table_index = map.at(setop_ptr->table_index);
		}
	} else if (auto insert_ptr = dynamic_cast<LogicalInsert *>(node)) {
		if (map.find(insert_ptr->table_index) != map.end()) {
			insert_ptr->table_index = map.at(insert_ptr->table_index);
		}
	} else if (auto dummy_ptr = dynamic_cast<LogicalDummyScan *>(node)) {
		if (map.find(dummy_ptr->table_index) != map.end()) {
			dummy_ptr->table_index = map.at(dummy_ptr->table_index);
		}
	} else if (auto coldata_ptr = dynamic_cast<LogicalColumnDataGet *>(node)) {
		if (map.find(coldata_ptr->table_index) != map.end()) {
			coldata_ptr->table_index = map.at(coldata_ptr->table_index);
		}
	} else if (auto upd_ptr = dynamic_cast<LogicalUpdate *>(node)) {
		if (map.find(upd_ptr->table_index) != map.end()) {
			upd_ptr->table_index = map.at(upd_ptr->table_index);
		}
	} else if (auto del_ptr = dynamic_cast<LogicalDelete *>(node)) {
		if (map.find(del_ptr->table_index) != map.end()) {
			del_ptr->table_index = map.at(del_ptr->table_index);
		}
	} else if (auto cte_ptr = dynamic_cast<LogicalCTE *>(node)) {
		if (map.find(cte_ptr->table_index) != map.end()) {
			cte_ptr->table_index = map.at(cte_ptr->table_index);
		}
	} else if (auto rcte_ptr = dynamic_cast<LogicalRecursiveCTE *>(node)) {
		if (map.find(rcte_ptr->table_index) != map.end()) {
			rcte_ptr->table_index = map.at(rcte_ptr->table_index);
		}
	} else if (auto eg_ptr = dynamic_cast<LogicalExpressionGet *>(node)) {
		if (map.find(eg_ptr->table_index) != map.end()) {
			eg_ptr->table_index = map.at(eg_ptr->table_index);
		}
	}

	// Special handling for LogicalAggregate: multiple indices
	if (auto agg_ptr = dynamic_cast<LogicalAggregate *>(node)) {
		if (map.find(agg_ptr->aggregate_index) != map.end()) {
			agg_ptr->aggregate_index = map.at(agg_ptr->aggregate_index);
		}
		if (map.find(agg_ptr->group_index) != map.end()) {
			agg_ptr->group_index = map.at(agg_ptr->group_index);
		}
		if (agg_ptr->groupings_index != DConstants::INVALID_INDEX && map.find(agg_ptr->groupings_index) != map.end()) {
			agg_ptr->groupings_index = map.at(agg_ptr->groupings_index);
		}
	}

	// Recurse
	for (auto &c : node->children) {
		ApplyIndexMapToSubtree(c.get(), map);
	}
}

void ReplaceNode(unique_ptr<LogicalOperator> &root, unique_ptr<LogicalOperator> &old_node,
                 unique_ptr<LogicalOperator> &new_node) {
	// Validate inputs
	if (!old_node) {
		throw InternalException("ReplaceNode: old_node must not be null");
	}
	if (!new_node) {
		throw InternalException("ReplaceNode: new_node must not be null");
	}

	// Keep pointer to the old subtree (before destruction)
	LogicalOperator *old_subtree_ptr = old_node.get();
	if (!old_subtree_ptr) {
		throw InternalException("ReplaceNode: referenced old subtree is null");
	}

	// Collect top-level bindings from the old subtree (these are the bindings advertised by the old child)
	vector<ColumnBinding> old_top_bindings = old_subtree_ptr->GetColumnBindings();

	// Collect all table indices present in the old subtree so we can exclude them when computing external indices
	idx_set old_subtree_indices;
	CollectTableIndicesRecursive(old_subtree_ptr, old_subtree_indices);

	// Compute external indices (everything in the plan except the old subtree)
	idx_set external_indices;
	CollectTableIndicesExcluding(root.get(), old_subtree_ptr, external_indices);

	// Replace the slot with the new node (this destroys the old subtree)
	old_node = std::move(new_node);
	LogicalOperator *subtree_root = old_node.get();
	if (!subtree_root) {
		throw InternalException("ReplaceNode: inserted subtree is null after move");
	}

	// Collect table indices present in the newly inserted subtree
	idx_set new_subtree_indices;
	CollectTableIndicesRecursive(subtree_root, new_subtree_indices);

	// Build index remapping for any new-subtree indices that collide with external indices
	std::unordered_map<idx_t, idx_t> index_map;
	if (!new_subtree_indices.empty()) {
		idx_t next_idx = GetNextTableIndex(root);
		for (auto idx : new_subtree_indices) {
			if (idx == DConstants::INVALID_INDEX) {
				continue;
			}
			if (external_indices.find(idx) != external_indices.end()) {
				// find a fresh index not in external_indices and not in new_subtree_indices
				while (external_indices.find(next_idx) != external_indices.end() ||
				       new_subtree_indices.find(next_idx) != new_subtree_indices.end()) {
					++next_idx;
				}
				index_map[idx] = next_idx;
				external_indices.insert(next_idx);
				++next_idx;
			}
		}
	}

	// Apply index remapping to the inserted subtree if necessary
	if (!index_map.empty()) {
		ApplyIndexMapToSubtree(subtree_root, index_map);
	}

	// After remap, get the new top-level bindings
	vector<ColumnBinding> new_top_bindings = subtree_root->GetColumnBindings();

	// Build positional replacement map from old_top_bindings -> new_top_bindings
	ColumnBindingReplacer replacer;
	const idx_t n = (std::min)(old_top_bindings.size(), new_top_bindings.size());
	replacer.replacement_bindings.reserve(n);
	for (idx_t i = 0; i < n; ++i) {
		if (old_top_bindings[i] != new_top_bindings[i]) {
			replacer.replacement_bindings.emplace_back(old_top_bindings[i], new_top_bindings[i]);
		}
	}

	if (!replacer.replacement_bindings.empty()) {
		replacer.stop_operator = subtree_root;
		replacer.VisitOperator(*root);
	}
}

// Find the primary key column name for a given table. Searches the client's catalog search path
// for the table and returns the first column name of the primary key constraint (if any).
// Also checks PAC KEY metadata if no database constraint is found.
// Returns empty vector when no primary key exists.
vector<string> FindPrimaryKey(ClientContext &context, const string &table_name) {
	Connection con(*context.db);
	Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));

	// Helper that checks a vector of column names exist and are numeric; returns the vector if valid,
	// otherwise returns empty vector.
	auto check_and_return_numeric_columns = [&](TableCatalogEntry &table_entry,
	                                            const vector<string> &cols) -> vector<string> {
		vector<string> out;
		out.reserve(cols.size());
		for (auto &col_name : cols) {
			// TableCatalogEntry::GetColumnIndex expects a non-const string reference
			string tmp = col_name;
			// If column not found, treat as no PK
			try {
				auto col_idx = table_entry.GetColumnIndex(tmp);
				auto &col = table_entry.GetColumn(col_idx);
				if (!col.Type().IsNumeric()) {
					// found a non-numeric PK column — treat as no primary key
					return {};
				}
				out.push_back(col.GetName());
			} catch (...) {
				return {};
			}
		}
		return out;
	};

	// If schema-qualified name is provided (schema.table), prefer that exact lookup
	auto dot_pos = table_name.find('.');
	if (dot_pos != string::npos) {
		string schema = table_name.substr(0, dot_pos);
		string tbl = table_name.substr(dot_pos + 1);
		auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema, tbl, OnEntryNotFound::RETURN_NULL);
		if (!entry) {
			return {};
		}
		auto &table_entry = entry->Cast<TableCatalogEntry>();
		auto pk = table_entry.GetPrimaryKey();
		if (pk && pk->type == ConstraintType::UNIQUE) {
			auto &unique = pk->Cast<UniqueConstraint>();
			// If explicit column names present, validate all are numeric
			if (!unique.GetColumnNames().empty()) {
				auto cols = unique.GetColumnNames();
				auto validated = check_and_return_numeric_columns(table_entry, cols);
				if (!validated.empty()) {
					return validated;
				}
			}
			// Otherwise fall back to index-based single-column PK
			if (unique.HasIndex()) {
				auto idx = unique.GetIndex();
				auto &col = table_entry.GetColumn(idx);
				if (col.Type().IsNumeric()) {
					return {col.GetName()};
				}
			}
		}
		// No database PK constraint found - check PAC KEY metadata
		auto &metadata_mgr = PACMetadataManager::Get();
		auto *pac_meta = metadata_mgr.GetTableMetadata(tbl);
		if (pac_meta && !pac_meta->primary_key_columns.empty()) {
			auto validated = check_and_return_numeric_columns(table_entry, pac_meta->primary_key_columns);
			if (!validated.empty()) {
				return validated;
			}
		}
		return {};
	}

	// Non-qualified name: walk the search path
	CatalogSearchPath path(context);
	for (auto &entry_path : path.Get()) {
		auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, entry_path.schema, table_name,
		                              OnEntryNotFound::RETURN_NULL);
		if (!entry) {
			continue;
		}
		auto &table_entry = entry->Cast<TableCatalogEntry>();
		auto pk = table_entry.GetPrimaryKey();
		if (pk && pk->type == ConstraintType::UNIQUE) {
			auto &unique = pk->Cast<UniqueConstraint>();
			if (!unique.GetColumnNames().empty()) {
				auto cols = unique.GetColumnNames();
				auto validated = check_and_return_numeric_columns(table_entry, cols);
				if (!validated.empty()) {
					return validated;
				}
				// otherwise continue searching other schemas
			}
			if (unique.HasIndex()) {
				auto idx = unique.GetIndex();
				auto &col = table_entry.GetColumn(idx);
				if (col.Type().IsNumeric()) {
					return {col.GetName()};
				}
			}
		}
		// No database PK constraint found - check PAC KEY metadata
		auto &metadata_mgr = PACMetadataManager::Get();
		auto *pac_meta = metadata_mgr.GetTableMetadata(table_name);
		if (pac_meta && !pac_meta->primary_key_columns.empty()) {
			auto validated = check_and_return_numeric_columns(table_entry, pac_meta->primary_key_columns);
			if (!validated.empty()) {
				return validated;
			}
		}
	}

	return {};
}

// Find foreign key constraints declared on the given table. Mirrors FindPrimaryKey's lookup logic
// and returns a vector of (referenced_table_name, fk_column_names) pairs for every FOREIGN KEY
// constraint defined on the table (i.e., where this table is the foreign-key side).
// Also includes PAC LINK relationships defined in PAC metadata.
vector<std::pair<string, vector<string>>> FindForeignKeys(ClientContext &context, const string &table_name) {
	Connection con(*context.db);
	Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
	vector<std::pair<string, vector<string>>> result;

	auto process_entry = [&](CatalogEntry *entry_ptr) {
		if (!entry_ptr) {
			return;
		}
		auto &table_entry = entry_ptr->Cast<TableCatalogEntry>();
		auto &constraints = table_entry.GetConstraints();
		for (auto &constraint : constraints) {
			if (!constraint) {
				continue;
			}
			if (constraint->type != ConstraintType::FOREIGN_KEY) {
				continue;
			}
			auto &fk = constraint->Cast<ForeignKeyConstraint>();
			// We only care about constraints where this table is the foreign-key table (append constraint)
			if (!fk.info.IsAppendConstraint()) {
				continue;
			}
			// Build referenced table name: DO NOT return schema-qualified name here; only return the table name.
			// The schema-qualified variant adds complexity and is left for future work.
			string ref_table = fk.info.table;
			// fk.fk_columns contains the column names on THIS table that reference the other
			result.emplace_back(ref_table, fk.fk_columns);
		}
	};

	// Extract unqualified table name for PAC metadata lookup
	string unqualified_table_name = table_name;
	auto dot_pos = table_name.find('.');
	if (dot_pos != string::npos) {
		unqualified_table_name = table_name.substr(dot_pos + 1);
	}

	// If schema-qualified name is provided (schema.table), prefer that exact lookup
	if (dot_pos != string::npos) {
		string schema = table_name.substr(0, dot_pos);
		string tbl = table_name.substr(dot_pos + 1);
		auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema, tbl, OnEntryNotFound::RETURN_NULL);
		if (entry) {
			process_entry(entry.get());
		}
		// If not found in catalog, continue to check PAC metadata below
	} else {
		// Non-qualified name: walk the search path
		CatalogSearchPath path(context);
		for (auto &entry_path : path.Get()) {
			auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, entry_path.schema, table_name,
			                              OnEntryNotFound::RETURN_NULL);
			if (!entry) {
				continue;
			}
			process_entry(entry.get());
		}
	}

	// Also check for PAC links in the metadata manager
	// PAC links supplement FK constraints, so we add them to the result
	auto &metadata_mgr = PACMetadataManager::Get();
	auto *pac_metadata = metadata_mgr.GetTableMetadata(unqualified_table_name);

	if (pac_metadata) {
		// Add each PAC link as a foreign key relationship
		for (auto &link : pac_metadata->links) {
			// PAC links now support composite keys with local_columns and referenced_columns arrays
			result.emplace_back(link.referenced_table, link.local_columns);
		}
	}

	return result;
}

// Find foreign-key path(s) from any of `table_names` to any of `privacy_units`.
// This function resolves table names using the client's catalog search path, then performs a
// BFS over outgoing FK edges (A -> referenced_table) to find the shortest path from each start
// table to any privacy unit. Returns a map from the original start table string to the path
// (vector of table names from start to privacy unit inclusive). NOTE: table names are returned
// unqualified (no schema prefix) to keep logic simple.
std::unordered_map<string, vector<string>>
FindForeignKeyBetween(ClientContext &context, const vector<string> &privacy_units, const vector<string> &table_names) {
	Connection con(*context.db);
	Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));

	auto ResolveQualified = [&](const string &tbl_name) -> string {
		// If schema-qualified name is provided (schema.table), prefer that exact lookup but return
		// only the unqualified table name (tbl).
		auto dot_pos = tbl_name.find('.');
		if (dot_pos != string::npos) {
			string schema = tbl_name.substr(0, dot_pos);
			string tbl = tbl_name.substr(dot_pos + 1);
			auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema, tbl, OnEntryNotFound::RETURN_NULL);
			if (entry) {
				return StringUtil::Lower(tbl); // return unqualified table name, normalized to lowercase
			}
			return StringUtil::Lower(tbl_name); // fallback to original, normalized
		}
		// Non-qualified: walk the search path; if found return unqualified table name
		CatalogSearchPath path(context);
		for (auto &entry_path : path.Get()) {
			auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, entry_path.schema, tbl_name,
			                              OnEntryNotFound::RETURN_NULL);
			if (entry) {
				return StringUtil::Lower(tbl_name); // already unqualified, normalized to lowercase
			}
		}
		return StringUtil::Lower(tbl_name); // fallback, normalized to lowercase
	};

	// canonicalize privacy units (use unqualified names, normalized to lowercase)
	std::unordered_set<string> privacy_set;
	for (auto &pu : privacy_units) {
		privacy_set.insert(ResolveQualified(pu));
	}

	std::unordered_map<string, vector<string>> result;

	for (auto &start : table_names) {
		string start_name = ResolveQualified(start);
		// BFS queue of unqualified table names (all lowercase)
		std::queue<string> q;
		std::unordered_set<string> visited;
		std::unordered_map<string, string> parent;

		visited.insert(start_name);
		q.push(start_name);

		bool found = false;
		string found_target;

		while (!q.empty() && !found) {
			string cur = q.front();
			q.pop();
			// Find outgoing FK edges from cur
			auto fks = FindForeignKeys(context, cur);
			for (auto &p : fks) {
				string neighbor = p.first; // referenced table name (unqualified now)
				string neighbor_name = StringUtil::Lower(ResolveQualified(neighbor)); // normalize to lowercase
				if (visited.find(neighbor_name) != visited.end()) {
					continue;
				}
				visited.insert(neighbor_name);
				parent[neighbor_name] = cur;
				if (privacy_set.find(neighbor_name) != privacy_set.end()) {
					found = true;
					found_target = neighbor_name;
					break;
				}
				q.push(neighbor_name);
			}
		}

		if (found) {
			// reconstruct path from start_name to found_target
			vector<string> path;
			string cur = found_target;
			while (true) {
				path.push_back(cur);
				if (cur == start_name) {
					break;
				}
				auto it = parent.find(cur);
				if (it == parent.end()) {
					break; // safety
				}
				cur = it->second;
			}
			reverse(path.begin(), path.end());
			result[start] = std::move(path);
		}
	}

	return result;
}

// ReplanGuard implementation
ReplanGuard::ReplanGuard(PACOptimizerInfo *info) : info(info), prev(false) {
	if (info) {
		prev = info->replan_in_progress.load(std::memory_order_acquire);
		info->replan_in_progress.store(true, std::memory_order_release);
	}
}

ReplanGuard::~ReplanGuard() {
	if (info) {
		info->replan_in_progress.store(prev, std::memory_order_release);
	}
}

// Configuration helpers
string GetPacPrivacyFile(ClientContext &context, const string &default_filename) {
	Value v;
	context.TryGetCurrentSetting("pac_privacy_file", v);
	if (!v.IsNull()) {
		return v.ToString();
	}
	return default_filename;
}

string GetPacCompiledPath(ClientContext &context, const string &default_path) {
	Value v;
	context.TryGetCurrentSetting("pac_compiled_path", v);
	string path = v.IsNull() ? default_path : v.ToString();
	if (!path.empty() && path.back() != '/') {
		path.push_back('/');
	}
	return path;
}

int64_t GetPacM(ClientContext &context, int64_t default_m) {
	Value v;
	context.TryGetCurrentSetting("pac_m", v);
	if (!v.IsNull()) {
		try {
			int64_t m = v.GetValue<int64_t>();
			return std::max<int64_t>(1, m);
		} catch (...) {
			// fallback to default if conversion fails
			return default_m;
		}
	}
	return default_m;
}

bool IsPacNoiseEnabled(ClientContext &context, bool default_value) {
	Value v;
	context.TryGetCurrentSetting("pac_noise", v);
	if (v.IsNull()) {
		return default_value;
	}
	try {
		return v.GetValue<bool>();
	} catch (...) {
		return default_value;
	}
}

vector<string> PacTablesSetToVector(const std::unordered_set<string> &set) {
	vector<string> out;
	out.reserve(set.size());
	for (auto &s : set) {
		out.push_back(s);
	}
	std::sort(out.begin(), out.end());
	return out;
}

// Add implementation for GetPacCompileMethod
string GetPacCompileMethod(ClientContext &context, const string &default_method) {
	Value v;
	context.TryGetCurrentSetting("pac_compile_method", v);
	if (v.IsNull()) {
		return default_method;
	}
	try {
		string s = v.ToString();
		std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
		if (s == "standard" || s == "bitslice") {
			return s;
		}
		// fallback
		return default_method;
	} catch (...) {
		return default_method;
	}
}

// Helper to safely retrieve boolean settings with defaults
bool GetBooleanSetting(ClientContext &context, const string &setting_name, bool default_value) {
	Value val;
	if (context.TryGetCurrentSetting(setting_name, val) && !val.IsNull()) {
		return val.GetValue<bool>();
	}
	return default_value;
}

// Helper to trace a binding back through the plan to find which LogicalGet it originates from
static LogicalGet *TraceBindingToSource(LogicalOperator &plan, const ColumnBinding &binding) {
	// Build a map from table_index to the operator that produces it
	std::unordered_map<idx_t, LogicalOperator *> table_index_to_op;

	std::function<void(LogicalOperator &)> collect_ops = [&](LogicalOperator &op) {
		auto bindings = op.GetColumnBindings();
		if (!bindings.empty()) {
			// Map this operator's table_index to the operator itself
			idx_t op_table_index = bindings[0].table_index;
			table_index_to_op[op_table_index] = &op;
		}
		for (auto &child : op.children) {
			if (child) {
				collect_ops(*child);
			}
		}
	};
	collect_ops(plan);

	// Start from the binding's table_index and trace backwards
	idx_t current_table_index = binding.table_index;
	idx_t current_column_index = binding.column_index;

	// Limit iterations to prevent infinite loops
	int max_iterations = 100;
	int iterations = 0;

	while (iterations++ < max_iterations) {
		auto it = table_index_to_op.find(current_table_index);
		if (it == table_index_to_op.end()) {
			return nullptr;
		}

		auto *op = it->second;

		// If we found a LogicalGet, we've reached the source
		if (op->type == LogicalOperatorType::LOGICAL_GET) {
			auto &get = op->Cast<LogicalGet>();
			return &get;
		}

		// For PROJECTION nodes, we need to look at the expression for this column
		if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = op->Cast<LogicalProjection>();

			// Check if this column_index is within the projection's expressions
			if (current_column_index >= proj.expressions.size()) {
				return nullptr;
			}

			// Get the expression for this column
			auto &expr = proj.expressions[current_column_index];

			// If it's a BoundColumnRefExpression, follow it
			if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &col_ref = expr->Cast<BoundColumnRefExpression>();
				current_table_index = col_ref.binding.table_index;
				current_column_index = col_ref.binding.column_index;
				continue;
			}

			// For other expression types, we can't trace further
			return nullptr;
		}

		// For AGGREGATE nodes, check if the column comes from groups
		if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			auto &agg = op->Cast<LogicalAggregate>();

			// Groups come first in the output column bindings
			if (current_column_index < agg.groups.size()) {
				// This is a group column, trace through the group expression
				auto &group_expr = agg.groups[current_column_index];

				if (group_expr->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = group_expr->Cast<BoundColumnRefExpression>();
					current_table_index = col_ref.binding.table_index;
					current_column_index = col_ref.binding.column_index;
					continue;
				}
				return nullptr;
			}
			// Aggregate expressions - we can't trace further
			return nullptr;
		}

		// For joins, we need to figure out which child the column comes from
		if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN || op->type == LogicalOperatorType::LOGICAL_JOIN ||
		    op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN || op->type == LogicalOperatorType::LOGICAL_ANY_JOIN ||
		    op->type == LogicalOperatorType::LOGICAL_ASOF_JOIN) {

			if (op->children.empty()) {
				return nullptr;
			}

			// Get left and right child bindings
			auto left_count = op->children[0] ? op->children[0]->GetColumnBindings().size() : 0;
			// Note: right_count could be used for validation but isn't needed for this logic

			// Check if column comes from left child
			if (current_column_index < left_count && op->children[0]) {
				auto left_bindings = op->children[0]->GetColumnBindings();
				if (current_column_index < left_bindings.size()) {
					current_table_index = left_bindings[current_column_index].table_index;
					current_column_index = left_bindings[current_column_index].column_index;
					continue;
				}
			}

			// Check if column comes from right child
			if (op->children.size() > 1 && op->children[1]) {
				idx_t right_col_idx = current_column_index - left_count;
				auto right_bindings = op->children[1]->GetColumnBindings();
				if (right_col_idx < right_bindings.size()) {
					current_table_index = right_bindings[right_col_idx].table_index;
					current_column_index = right_bindings[right_col_idx].column_index;
					continue;
				}
			}

			return nullptr;
		}

		// For other operators (filters, etc.), trace through the child
		if (!op->children.empty() && op->children[0]) {
			auto child_bindings = op->children[0]->GetColumnBindings();
			if (current_column_index < child_bindings.size()) {
				current_table_index = child_bindings[current_column_index].table_index;
				current_column_index = child_bindings[current_column_index].column_index;
			} else {
				return nullptr;
			}
		} else {
			return nullptr;
		}
	}

	return nullptr;
}

// Helper to collect all equi-join key equivalences from the plan.
// Returns pairs of column bindings that are equal due to equi-join conditions.
static void CollectJoinKeyEquivalences(LogicalOperator &op,
                                       vector<std::pair<ColumnBinding, ColumnBinding>> &equivalences) {
	// Check if this is a comparison join with equi-join conditions
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &join = op.Cast<LogicalComparisonJoin>();

		for (auto &cond : join.conditions) {
			// Only consider equality conditions
			if (cond.comparison != ExpressionType::COMPARE_EQUAL) {
				continue;
			}

			// Check if both sides are column references
			if (cond.left && cond.right && cond.left->type == ExpressionType::BOUND_COLUMN_REF &&
			    cond.right->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &left_ref = cond.left->Cast<BoundColumnRefExpression>();
				auto &right_ref = cond.right->Cast<BoundColumnRefExpression>();
				equivalences.emplace_back(left_ref.binding, right_ref.binding);
			}
		}
	}

	// Recursively process children
	for (auto &child : op.children) {
		if (child) {
			CollectJoinKeyEquivalences(*child, equivalences);
		}
	}
}

// Helper to find all bindings equivalent to the given binding via join key equivalences
static void FindEquivalentBindings(const ColumnBinding &binding,
                                   const vector<std::pair<ColumnBinding, ColumnBinding>> &equivalences,
                                   std::unordered_set<uint64_t> &visited, vector<ColumnBinding> &result) {
	// Create a unique key for the binding
	uint64_t key = (static_cast<uint64_t>(binding.table_index) << 32) | binding.column_index;
	if (visited.find(key) != visited.end()) {
		return;
	}
	visited.insert(key);
	result.push_back(binding);

	// Find all equivalences involving this binding
	for (auto &eq : equivalences) {
		if (eq.first.table_index == binding.table_index && eq.first.column_index == binding.column_index) {
			FindEquivalentBindings(eq.second, equivalences, visited, result);
		}
		if (eq.second.table_index == binding.table_index && eq.second.column_index == binding.column_index) {
			FindEquivalentBindings(eq.first, equivalences, visited, result);
		}
	}
}

bool ColumnBelongsToTable(LogicalOperator &plan, const string &table_name, const ColumnBinding &binding) {
	// First, collect all join key equivalences from the plan
	vector<std::pair<ColumnBinding, ColumnBinding>> equivalences;
	CollectJoinKeyEquivalences(plan, equivalences);

	// Find all bindings equivalent to the given binding (including itself)
	std::unordered_set<uint64_t> visited;
	vector<ColumnBinding> equivalent_bindings;
	FindEquivalentBindings(binding, equivalences, visited, equivalent_bindings);

	// Check if any equivalent binding traces back to the target table
	for (auto &eq_binding : equivalent_bindings) {
		auto *source_get = TraceBindingToSource(plan, eq_binding);
		if (!source_get) {
			continue;
		}

		auto table_entry = source_get->GetTable();
		if (!table_entry) {
			continue;
		}

		if (table_entry->name == table_name) {
			return true;
		}
	}

	return false;
}

} // namespace duckdb
