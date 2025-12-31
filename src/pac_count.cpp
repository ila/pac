#include "include/pac_count.hpp"

namespace duckdb {

static unique_ptr<FunctionData> // Bind function for pac_count with optional mi parameter (must be constant)
PacCountBind(ClientContext &ctx, AggregateFunction &func, vector<unique_ptr<Expression>> &args) {
	double mi = 128.0; // default

	// Handle mi parameter - check each argument position for a foldable numeric constant
	for (idx_t i = 1; i < args.size(); i++) {
		auto &arg = args[i];
		if (arg->IsFoldable() && (arg->return_type.IsNumeric() || arg->return_type.id() == LogicalTypeId::UNKNOWN)) {
			auto val = ExpressionExecutor::EvaluateScalar(ctx, *arg);
			if (val.type().IsNumeric()) {
				mi = val.GetValue<double>();
				if (mi < 0.0) {
					throw InvalidInputException("pac_count: mi must be >= 0");
				}
				break;
			}
		}
	}

	uint64_t seed = std::random_device {}();
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = uint64_t(pac_seed_val.GetValue<int64_t>());
	}
	return make_uniq<PacBindData>(mi, seed);
}

static idx_t PacCountStateSize(const AggregateFunction &) {
	return sizeof(PacCountState);
}

static void PacCountInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	memset(state_ptr, 0, sizeof(PacCountState));
}

#ifdef PAC_COUNT_NONBANKED
AUTOVECTORIZE static inline void PacCountUpdateHash(uint64_t key_hash, PacCountState &state) {
	for (int j = 0; j < 64; j++) {
		state.probabilistic_total[j] += (key_hash >> j) & 1ULL;
	}
}
#else
AUTOVECTORIZE static inline void PacCountUpdateHash(uint64_t key_hash, PacCountState &state,
                                                    ArenaAllocator &allocator) {
	// SWAR update to subtotal8
	for (int j = 0; j < 8; j++) {
		state.probabilistic_subtotal8[j] += (key_hash >> j) & PAC_COUNT_MASK;
	}
	// Flush if subtotal8 would overflow
	if (++state.subtotal8_count == 255) {
		state.Flush(allocator);
	}
}
#endif

#ifdef PAC_COUNT_NONBANKED
#define UPDATE_HASH(hash, state, allocator) PacCountUpdateHash(hash, state)
#else
#define UPDATE_HASH(hash, state, allocator) PacCountUpdateHash(hash, state, allocator)
#endif

// Row-based update functions
void PacCountUpdate(Vector inputs[], AggregateInputData &aggr, idx_t input_total, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_total == 1 || input_total == 2);
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);
	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);
		if (idata.validity.RowIsValid(idx)) {
			UPDATE_HASH(input_data[idx], state, aggr.allocator);
		}
	}
}

void PacCountScatterUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	UnifiedVectorFormat idata, sdata;
	inputs[0].ToUnifiedFormat(count, idata);
	states.ToUnifiedFormat(count, sdata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	auto state_p = UnifiedVectorFormat::GetData<PacCountState *>(sdata);
	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);
		if (idata.validity.RowIsValid(idx)) {
			UPDATE_HASH(input_data[idx], *state_p[sdata.sel->get_index(i)], aggr.allocator);
		}
	}
}

// Column-based update functions (count non-null values)
void PacCountColumnUpdate(Vector inputs[], AggregateInputData &aggr, idx_t input_total, data_ptr_t state_ptr,
                          idx_t count) {
	D_ASSERT(input_total >= 2);
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);
	UnifiedVectorFormat hash_data, col_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, col_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto c_idx = col_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(h_idx) && col_data.validity.RowIsValid(c_idx)) {
			UPDATE_HASH(hashes[h_idx], state, aggr.allocator);
		}
	}
}

void PacCountColumnScatterUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	UnifiedVectorFormat hash_data, col_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, col_data);
	states.ToUnifiedFormat(count, sdata);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto state_p = UnifiedVectorFormat::GetData<PacCountState *>(sdata);
	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto c_idx = col_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(h_idx) && col_data.validity.RowIsValid(c_idx)) {
			UPDATE_HASH(hashes[h_idx], *state_p[sdata.sel->get_index(i)], aggr.allocator);
		}
	}
}

#ifdef PAC_COUNT_NONBANKED
AUTOVECTORIZE void PacCountCombine(Vector &src, Vector &dst, AggregateInputData &, idx_t count) {
	auto src_state = FlatVector::GetData<PacCountState *>(src);
	auto dst_state = FlatVector::GetData<PacCountState *>(dst);
	for (idx_t i = 0; i < count; i++) {
		for (int j = 0; j < 64; j++) {
			dst_state[i]->probabilistic_total[j] += src_state[i]->probabilistic_total[j];
		}
	}
}
#else
static inline int RequiredLevel(uint64_t count) {
	return (count <= UINT16_MAX) ? 16 : (count <= UINT32_MAX) ? 32 : 64;
}

// Combine src banks into dst banks (same type, direct add)
template <typename T>
AUTOVECTORIZE static inline void CombineBanks(PacCountState *d, const PacCountState *s) {
	constexpr int ELEMS_PER_BANK = 64 / sizeof(T);
	int num_banks = PacCountState::BanksForLevel(sizeof(T) * 8);
	for (int b = 0; b < num_banks; b++) {
		auto *dp = reinterpret_cast<T *>(d->GetBank(b));
		auto *sp = reinterpret_cast<const T *>(s->GetBank(b));
		for (int j = 0; j < ELEMS_PER_BANK; j++)
			dp[j] += sp[j];
	}
}

// Combine src (narrower) into dst (wider) using element access
template <typename DST_T, typename SRC_T>
AUTOVECTORIZE static inline void CombineBanksWiden(PacCountState *d, const PacCountState *s) {
	constexpr int ELEMS_PER_BANK = 64 / sizeof(DST_T);
	int num_banks = PacCountState::BanksForLevel(sizeof(DST_T) * 8);
	for (int b = 0; b < num_banks; b++) {
		auto *dp = reinterpret_cast<DST_T *>(d->GetBank(b));
		for (int j = 0; j < ELEMS_PER_BANK; j++) {
			dp[j] += s->GetTotalElement<SRC_T>(b * ELEMS_PER_BANK + j);
		}
	}
}

AUTOVECTORIZE void PacCountCombine(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	auto src_state = FlatVector::GetData<PacCountState *>(src);
	auto dst_state = FlatVector::GetData<PacCountState *>(dst);
	for (idx_t i = 0; i < count; i++) {
		auto *s = src_state[i];
		auto *d = dst_state[i];

		s->Flush(aggr.allocator);
		d->Flush(aggr.allocator);
		if (s->total_level == 0)
			continue;

		// Determine required level (max of src, dst, and level needed for sum)
		uint64_t combined_max = s->GetMaxPossibleCount() + d->GetMaxPossibleCount();
		int required = RequiredLevel(combined_max);
		if (s->total_level > required)
			required = s->total_level;
		if (d->total_level > required)
			required = d->total_level;

		// Initialize or upgrade dst to required level
		if (d->total_level == 0) {
			d->AllocateBanks(aggr.allocator, required);
			d->total_level = required;
		}
		if (d->total_level == 16 && required > 16)
			d->UpgradeTotal<uint16_t, uint32_t>(aggr.allocator, 32);
		if (d->total_level == 32 && required > 32)
			d->UpgradeTotal<uint32_t, uint64_t>(aggr.allocator, 64);

		// Combine based on dst level and src level
		if (d->total_level == 16) {
			CombineBanks<uint16_t>(d, s);
		} else if (d->total_level == 32) {
			if (s->total_level == 32)
				CombineBanks<uint32_t>(d, s);
			else
				CombineBanksWiden<uint32_t, uint16_t>(d, s);
		} else {
			if (s->total_level == 64)
				CombineBanks<uint64_t>(d, s);
			else if (s->total_level == 32)
				CombineBanksWiden<uint64_t, uint32_t>(d, s);
			else
				CombineBanksWiden<uint64_t, uint16_t>(d, s);
		}
	}
}
#endif

void PacCountFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state = FlatVector::GetData<PacCountState *>(states);
	auto data = FlatVector::GetData<int64_t>(result);
	uint64_t seed = input.bind_data ? input.bind_data->Cast<PacBindData>().seed : std::random_device {}();
	std::mt19937_64 gen(seed);
	double mi = input.bind_data->Cast<PacBindData>().mi;

	for (idx_t i = 0; i < count; i++) {
		state[i]->Flush(input.allocator);
		double buf[64];
		state[i]->GetTotalsAsDouble(buf);
		data[offset + i] =
		    static_cast<int64_t>(PacNoisySampleFrom64Counters(buf, mi, gen)) + static_cast<int64_t>(buf[41]);
	}
}

void RegisterPacCountFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_count");

	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT}, LogicalType::BIGINT, PacCountStateSize,
	                                      PacCountInitialize, PacCountScatterUpdate, PacCountCombine, PacCountFinalize,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate, PacCountBind));

	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT, LogicalType::DOUBLE}, LogicalType::BIGINT,
	                                      PacCountStateSize, PacCountInitialize, PacCountScatterUpdate, PacCountCombine,
	                                      PacCountFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate,
	                                      PacCountBind));

	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT, LogicalType::ANY}, LogicalType::BIGINT,
	                                      PacCountStateSize, PacCountInitialize, PacCountColumnScatterUpdate,
	                                      PacCountCombine, PacCountFinalize, FunctionNullHandling::SPECIAL_HANDLING,
	                                      PacCountColumnUpdate, PacCountBind));

	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT, LogicalType::ANY, LogicalType::DOUBLE},
	                                      LogicalType::BIGINT, PacCountStateSize, PacCountInitialize,
	                                      PacCountColumnScatterUpdate, PacCountCombine, PacCountFinalize,
	                                      FunctionNullHandling::SPECIAL_HANDLING, PacCountColumnUpdate, PacCountBind));

	loader.RegisterFunction(fcn_set);
}

} // namespace duckdb
