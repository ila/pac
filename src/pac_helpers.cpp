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

void UpdateParent(unique_ptr<LogicalOperator> &root,
                  unique_ptr<LogicalOperator> &child_ref,
                  unique_ptr<LogicalOperator> new_parent) {
    // Basic preconditions
    if (!child_ref) {
        // Nothing to do
        return;
    }
    if (!new_parent) {
        throw InternalException("UpdateParent: new_parent is null");
    }
    if (!new_parent->children.empty()) {
        throw InvalidInputException("UpdateParent: new_parent must not have children");
    }

    // Keep pointer to original child (will be moved)
    LogicalOperator *original_child = child_ref.get();

    // Collect table indices used outside the subtree (we will avoid colliding with these)
    idx_set external_indices;
    CollectTableIndicesExcluding(root.get(), original_child, external_indices);

    // Collect all column bindings and table indices from the subtree to insert
    vector<ColumnBinding> subtree_old_bindings;
    CollectColumnBindingsRecursive(original_child, subtree_old_bindings);
    idx_set subtree_indices;
    CollectTableIndicesRecursive(original_child, subtree_indices);

    // Insert: make the original child a child of new_parent, and place new_parent where child_ref was.
    new_parent->children.emplace_back(std::move(child_ref));
    // child_ref now becomes the new_parent
    child_ref = std::move(new_parent);
    LogicalOperator *subtree_root = child_ref.get();

    // Build mapping for any table index in subtree that collides with external indices
    std::unordered_map<idx_t, idx_t> index_map;
    if (!subtree_indices.empty()) {
        // Start allocating new indices from the next available table index in the plan
        idx_t next_idx = GetNextTableIndex(root);
        for (auto old_idx : subtree_indices) {
            if (old_idx == DConstants::INVALID_INDEX) continue;
            // Only remap when there's a collision with an external index
            if (external_indices.find(old_idx) != external_indices.end()) {
                // assign next unique index
                while (external_indices.find(next_idx) != external_indices.end() || subtree_indices.find(next_idx) != subtree_indices.end()) {
                    ++next_idx;
                }
                index_map[old_idx] = next_idx;
                // Mark next_idx as used to avoid duplicates
                external_indices.insert(next_idx);
                ++next_idx;
            }
        }
    }

    // If we need to remap indices, apply to subtree operator fields and create column binding replacements
    if (!index_map.empty()) {
        // Apply mapping to operator-local index fields within the subtree
        ApplyIndexMapToSubtree(subtree_root, index_map);
        // Build column-binding replacements from the collected subtree_old_bindings
        ColumnBindingReplacer replacer;
        replacer.replacement_bindings.reserve(subtree_old_bindings.size());
        for (const auto &old_cb : subtree_old_bindings) {
            auto it = index_map.find(old_cb.table_index);
            if (it != index_map.end()) {
                ColumnBinding new_cb = ColumnBinding(it->second, old_cb.column_index);
                replacer.replacement_bindings.emplace_back(old_cb, new_cb);
            }
        }
        // Apply replacer to the tree, stopping at the newly inserted subtree
        replacer.stop_operator = subtree_root;
        replacer.VisitOperator(*root);
    } else {
        // No index remapping required. To preserve old behaviour we still perform binding positional replacement
        // for the top node: map old top-level bindings where positions changed
        vector<ColumnBinding> old_top = original_child->GetColumnBindings();
        vector<ColumnBinding> new_top = subtree_root->GetColumnBindings();
        ColumnBindingReplacer replacer;
        const idx_t n = (std::min)(old_top.size(), new_top.size());
        for (idx_t i = 0; i < n; ++i) {
            if (!(old_top[i] == new_top[i])) {
                replacer.replacement_bindings.emplace_back(old_top[i], new_top[i]);
            }
        }
        if (!replacer.replacement_bindings.empty()) {
            replacer.stop_operator = subtree_root;
            replacer.VisitOperator(*root);
        }
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

} // namespace duckdb
