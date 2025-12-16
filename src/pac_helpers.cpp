#include "pac_helpers.hpp"

#include <sstream>
#include <functional>
#include <cctype>

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

} // namespace duckdb
