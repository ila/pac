#include "include/pac_helpers.hpp"

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
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <queue>

// Add include for TableCatalogEntry to access GetPrimaryKey and column APIs
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
// Add include for ForeignKeyConstraint
#include "duckdb/parser/constraints/foreign_key_constraint.hpp"

using idx_set = std::unordered_set<idx_t>;

namespace duckdb {

string Sanitize(const string &in) {
	string out;
	for (char c : in) {
		out.push_back(std::isalnum((unsigned char)c) || c == '_' ? c : '_');
	}
	return out;
}

std::string NormalizeQueryForHash(const std::string &query) {
    std::string s = query;
    std::replace(s.begin(), s.end(), '\n', ' ');
    std::replace(s.begin(), s.end(), '\r', ' ');
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    std::string out;
    out.reserve(s.size());
    bool in_space = false;
    for (char c : s) {
        if (std::isspace((unsigned char)c)) {
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
    if (!out.empty() && out.front() == ' ') out.erase(out.begin());
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::string HashStringToHex(const std::string &input) {
    size_t h = std::hash<std::string>{}(input);
    std::stringstream ss;
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
        if (!cur) continue;
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

// Forward declarations of helper functions defined later in this file
static void CollectTableIndicesRecursive(LogicalOperator *node, idx_set &out);
static void CollectTableIndicesExcluding(LogicalOperator *node, LogicalOperator *skip, idx_set &out);
static void ApplyIndexMapToSubtree(LogicalOperator *node, const std::unordered_map<idx_t, idx_t> &map);
static void CollectColumnBindingsRecursive(LogicalOperator *node, vector<ColumnBinding> &out);

void ReplaceNode(unique_ptr<LogicalOperator> &root,
                  unique_ptr<LogicalOperator> &old_node,
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
            if (idx == DConstants::INVALID_INDEX) continue;
            if (external_indices.find(idx) != external_indices.end()) {
                // find a fresh index not in external_indices and not in new_subtree_indices
                while (external_indices.find(next_idx) != external_indices.end() || new_subtree_indices.find(next_idx) != new_subtree_indices.end()) {
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
        if (!(old_top_bindings[i] == new_top_bindings[i])) {
            replacer.replacement_bindings.emplace_back(old_top_bindings[i], new_top_bindings[i]);
        }
    }

    if (!replacer.replacement_bindings.empty()) {
        replacer.stop_operator = subtree_root;
        replacer.VisitOperator(*root);
    }
}

static void CollectColumnBindingsRecursive(LogicalOperator *node, vector<ColumnBinding> &out) {
    if (!node) return;
    auto binds = node->GetColumnBindings();
    out.insert(out.end(), binds.begin(), binds.end());
    for (auto &c : node->children) CollectColumnBindingsRecursive(c.get(), out);
}

static void CollectTableIndicesRecursive(LogicalOperator *node, idx_set &out) {
    if (!node) return;
    auto tbls = node->GetTableIndex();
    for (auto t : tbls) {
        if (t != DConstants::INVALID_INDEX) out.insert(t);
    }
    for (auto &c : node->children) {
        CollectTableIndicesRecursive(c.get(), out);
    }
}

static void CollectTableIndicesExcluding(LogicalOperator *node, LogicalOperator *skip, idx_set &out) {
    if (!node) return;
    if (node == skip) return; // skip this subtree entirely
    auto tbls = node->GetTableIndex();
    for (auto t : tbls) {
        if (t != DConstants::INVALID_INDEX) out.insert(t);
    }
    for (auto &c : node->children) {
        CollectTableIndicesExcluding(c.get(), skip, out);
    }
}

static void ApplyIndexMapToSubtree(LogicalOperator *node, const std::unordered_map<idx_t, idx_t> &map) {
    if (!node) return;
    // Update operator-specific index fields where applicable
    if (auto get_ptr = dynamic_cast<LogicalGet *>(node)) {
        if (map.find(get_ptr->table_index) != map.end()) get_ptr->table_index = map.at(get_ptr->table_index);
    } else if (auto proj_ptr = dynamic_cast<LogicalProjection *>(node)) {
        if (map.find(proj_ptr->table_index) != map.end()) proj_ptr->table_index = map.at(proj_ptr->table_index);
    } else if (auto setop_ptr = dynamic_cast<LogicalSetOperation *>(node)) {
        if (map.find(setop_ptr->table_index) != map.end()) setop_ptr->table_index = map.at(setop_ptr->table_index);
    } else if (auto insert_ptr = dynamic_cast<LogicalInsert *>(node)) {
        if (map.find(insert_ptr->table_index) != map.end()) insert_ptr->table_index = map.at(insert_ptr->table_index);
    } else if (auto dummy_ptr = dynamic_cast<LogicalDummyScan *>(node)) {
        if (map.find(dummy_ptr->table_index) != map.end()) dummy_ptr->table_index = map.at(dummy_ptr->table_index);
    } else if (auto coldata_ptr = dynamic_cast<LogicalColumnDataGet *>(node)) {
        if (map.find(coldata_ptr->table_index) != map.end()) coldata_ptr->table_index = map.at(coldata_ptr->table_index);
    } else if (auto upd_ptr = dynamic_cast<LogicalUpdate *>(node)) {
        if (map.find(upd_ptr->table_index) != map.end()) upd_ptr->table_index = map.at(upd_ptr->table_index);
    } else if (auto del_ptr = dynamic_cast<LogicalDelete *>(node)) {
        if (map.find(del_ptr->table_index) != map.end()) del_ptr->table_index = map.at(del_ptr->table_index);
    } else if (auto cte_ptr = dynamic_cast<LogicalCTE*>(node)) {
        if (map.find(cte_ptr->table_index) != map.end()) cte_ptr->table_index = map.at(cte_ptr->table_index);
    } else if (auto rcte_ptr = dynamic_cast<LogicalRecursiveCTE*>(node)) {
        if (map.find(rcte_ptr->table_index) != map.end()) rcte_ptr->table_index = map.at(rcte_ptr->table_index);
    } else if (auto eg_ptr = dynamic_cast<LogicalExpressionGet*>(node)) {
        if (map.find(eg_ptr->table_index) != map.end()) eg_ptr->table_index = map.at(eg_ptr->table_index);
    }

    // Special handling for LogicalAggregate: multiple indices
    if (auto agg_ptr = dynamic_cast<LogicalAggregate *>(node)) {
        if (map.find(agg_ptr->aggregate_index) != map.end()) agg_ptr->aggregate_index = map.at(agg_ptr->aggregate_index);
        if (map.find(agg_ptr->group_index) != map.end()) agg_ptr->group_index = map.at(agg_ptr->group_index);
        if (agg_ptr->groupings_index != DConstants::INVALID_INDEX && map.find(agg_ptr->groupings_index) != map.end())
            agg_ptr->groupings_index = map.at(agg_ptr->groupings_index);
    }

    // Recurse
    for (auto &c : node->children) ApplyIndexMapToSubtree(c.get(), map);
}

// Find the primary key column name for a given table. Searches the client's catalog search path
// for the table and returns the first column name of the primary key constraint (if any).
// Returns empty string when no primary key exists.
vector<std::string> FindPrimaryKey(ClientContext &context, const std::string &table_name) {
	Connection con(*context.db);
    Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));

    // If schema-qualified name is provided (schema.table), prefer that exact lookup
    auto dot_pos = table_name.find('.');
    if (dot_pos != std::string::npos) {
        std::string schema = table_name.substr(0, dot_pos);
        std::string tbl = table_name.substr(dot_pos + 1);
        auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema, tbl, OnEntryNotFound::RETURN_NULL);
        if (!entry) return {};
        auto &table_entry = entry->Cast<TableCatalogEntry>();
        auto pk = table_entry.GetPrimaryKey();
        if (!pk) return {};
        if (pk->type == ConstraintType::UNIQUE) {
            auto &unique = pk->Cast<UniqueConstraint>();
            // Prefer explicit column names if present
            if (!unique.GetColumnNames().empty()) {
                return unique.GetColumnNames();
            }
            // Otherwise fall back to index-based single-column PK
            if (unique.HasIndex()) {
                auto idx = unique.GetIndex();
                auto &col = table_entry.GetColumn(idx);
                return {col.GetName()};
            }
        }
        return {};
    }

    // Non-qualified name: walk the search path
    CatalogSearchPath path(context);
    for (auto &entry_path : path.Get()) {
        auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, entry_path.schema, table_name,
                                      OnEntryNotFound::RETURN_NULL);
        if (!entry) continue;
        auto &table_entry = entry->Cast<TableCatalogEntry>();
        auto pk = table_entry.GetPrimaryKey();
        if (!pk) continue;
        if (pk->type == ConstraintType::UNIQUE) {
            auto &unique = pk->Cast<UniqueConstraint>();
            if (!unique.GetColumnNames().empty()) {
                return unique.GetColumnNames();
            }
            if (unique.HasIndex()) {
                auto idx = unique.GetIndex();
                auto &col = table_entry.GetColumn(idx);
                return {col.GetName()};
            }
        }
    }

    return {};
}

// Find foreign key constraints declared on the given table. Mirrors FindPrimaryKey's lookup logic
// and returns a vector of (referenced_table_name, fk_column_names) pairs for every FOREIGN KEY
// constraint defined on the table (i.e., where this table is the foreign-key side).
vector<std::pair<std::string, vector<std::string>>> FindForeignKeys(ClientContext &context, const std::string &table_name) {
    Connection con(*context.db);
    Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
    vector<std::pair<std::string, vector<std::string>>> result;

    auto process_entry = [&](CatalogEntry *entry_ptr) {
        if (!entry_ptr) return;
        auto &table_entry = entry_ptr->Cast<TableCatalogEntry>();
        auto &constraints = table_entry.GetConstraints();
        for (auto &constraint : constraints) {
            if (!constraint) continue;
            if (constraint->type != ConstraintType::FOREIGN_KEY) continue;
            auto &fk = constraint->Cast<ForeignKeyConstraint>();
            // We only care about constraints where this table is the foreign-key table (append constraint)
            if (!fk.info.IsAppendConstraint()) continue;
            // Build referenced table name (schema.table) if schema present
            std::string ref_table;
            if (!fk.info.schema.empty()) ref_table = fk.info.schema + "." + fk.info.table;
            else ref_table = fk.info.table;
            // fk.fk_columns contains the column names on THIS table that reference the other
            result.emplace_back(ref_table, fk.fk_columns);
        }
    };

    // If schema-qualified name is provided (schema.table), prefer that exact lookup
    auto dot_pos = table_name.find('.');
    if (dot_pos != std::string::npos) {
        std::string schema = table_name.substr(0, dot_pos);
        std::string tbl = table_name.substr(dot_pos + 1);
        auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema, tbl, OnEntryNotFound::RETURN_NULL);
        if (!entry) return {};
        process_entry(entry.get());
        return result;
    }

    // Non-qualified name: walk the search path
    CatalogSearchPath path(context);
    for (auto &entry_path : path.Get()) {
        auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, entry_path.schema, table_name,
                                      OnEntryNotFound::RETURN_NULL);
        if (!entry) continue;
        process_entry(entry.get());
    }

    return result;
}

// Find foreign-key path(s) from any of `table_names` to any of `privacy_units`.
// This function resolves table names using the client's catalog search path, then performs a
// BFS over outgoing FK edges (A -> referenced_table) to find the shortest path from each start
// table to any privacy unit. Returns a map from the original start table string to the path
// (vector of qualified table names from start to privacy unit inclusive).
std::unordered_map<std::string, std::vector<std::string>> FindForeignKeyBetween(
    ClientContext &context, const std::vector<std::string> &privacy_units, const std::vector<std::string> &table_names) {
    Connection con(*context.db);
    Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));

    auto ResolveQualified = [&](const std::string &tbl_name) -> std::string {
        auto dot_pos = tbl_name.find('.');
        if (dot_pos != std::string::npos) {
            std::string schema = tbl_name.substr(0, dot_pos);
            std::string tbl = tbl_name.substr(dot_pos + 1);
            auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema, tbl, OnEntryNotFound::RETURN_NULL);
            if (entry) return schema + "." + tbl;
            return tbl_name;
        }
        CatalogSearchPath path(context);
        for (auto &entry_path : path.Get()) {
            auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, entry_path.schema, tbl_name,
                                          OnEntryNotFound::RETURN_NULL);
            if (entry) return entry_path.schema + "." + tbl_name;
        }
        return tbl_name; // fallback: return as-is if not found
    };

    // canonicalize privacy units
    std::unordered_set<std::string> privacy_set;
    for (auto &pu : privacy_units) {
        privacy_set.insert(ResolveQualified(pu));
    }

    std::unordered_map<std::string, std::vector<std::string>> result;

    for (auto &start : table_names) {
        std::string start_qual = ResolveQualified(start);
        // BFS queue of qualified table names
        std::queue<std::string> q;
        std::unordered_set<std::string> visited;
        std::unordered_map<std::string, std::string> parent;

        visited.insert(start_qual);
        q.push(start_qual);

        bool found = false;
        std::string found_target;

        while (!q.empty() && !found) {
            std::string cur = q.front();
            q.pop();
            // Find outgoing FK edges from cur
            auto fks = FindForeignKeys(context, cur);
            for (auto &p : fks) {
                std::string neighbor = p.first; // referenced table name (may be qualified or not)
                std::string neighbor_qual = ResolveQualified(neighbor);
                if (visited.find(neighbor_qual) != visited.end()) continue;
                visited.insert(neighbor_qual);
                parent[neighbor_qual] = cur;
                if (privacy_set.find(neighbor_qual) != privacy_set.end()) {
                    found = true;
                    found_target = neighbor_qual;
                    break;
                }
                q.push(neighbor_qual);
            }
        }

        if (found) {
            // reconstruct path from start_qual to found_target
            std::vector<std::string> path;
            std::string cur = found_target;
            while (true) {
                path.push_back(cur);
                if (cur == start_qual) break;
                auto it = parent.find(cur);
                if (it == parent.end()) break; // safety
                cur = it->second;
            }
            std::reverse(path.begin(), path.end());
            result[start] = std::move(path);
        }
    }

    return result;
}

} // namespace duckdb
