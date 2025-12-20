#include "include/pac_aggregate.hpp"
#include "include/pac_sum.hpp"

namespace duckdb {

// ============================================================================
// pac_sum aggregate function
// ============================================================================
// State: 64 sums, one for each bit position
// Update: for each (key_hash, value), add value to sums[i] if bit i of key_hash is set
// Finalize: compute the PAC-noised sum from the 64 counters
//
// For integers: uses two-level accumulation with small_totals (input type) and totals (large type)
// For floats: uses direct accumulation in the same type


// =========================
// Float pac_sum (cascaded float32/float64 accumulation)
// =========================



// Internal update for float cascade
static inline void PacSumFloatCascadeUpdateInternal(PacSumFloatCascadeState &state, uint64_t key_hash, double value) {
#ifdef PAC_SUM_NONCASCADING
	AddToTotals64Float(state.totals64, value, key_hash);
#else
	if (!FloatNeedsDouble(value)) {
		AddToTotals32Float(state.totals32, static_cast<float>(value), key_hash);
		state.count32++;
		state.Flush32(false);
	} else {
		AddToTotals64Float(state.totals64, value, key_hash);
	}
#endif
}

// Float cascade update
template <class FLOAT_TYPE>
static void PacSumFloatCascadeUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr,
                                     idx_t count) {
	D_ASSERT(input_count == 2);
	auto &state = *reinterpret_cast<PacSumFloatCascadeState *>(state_ptr);
	if (state.seen_null) {
		return;
	}

	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<FLOAT_TYPE>(value_data);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state.seen_null = true;
			return;
		}
		PacSumFloatCascadeUpdateInternal(state, hashes[hash_idx], static_cast<double>(values[value_idx]));
	}
}

// Float cascade scatter update
template <class FLOAT_TYPE>
static void PacSumFloatCascadeScatterUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states,
                                            idx_t count) {
	D_ASSERT(input_count == 2);

	UnifiedVectorFormat hash_data, value_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<FLOAT_TYPE>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacSumFloatCascadeState *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		auto state_idx = sdata.sel->get_index(i);
		auto state = state_ptrs[state_idx];

		if (state->seen_null) {
			continue;
		}
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state->seen_null = true;
			continue;
		}
		PacSumFloatCascadeUpdateInternal(*state, hashes[hash_idx], static_cast<double>(values[value_idx]));
	}
}

static void PacSumFloatCascadeCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto sdata = FlatVector::GetData<PacSumFloatCascadeState *>(source);
	auto tdata = FlatVector::GetData<PacSumFloatCascadeState *>(target);
	for (idx_t i = 0; i < count; i++) {
		if (sdata[i]->seen_null) {
			tdata[i]->seen_null = true;
		}
		if (tdata[i]->seen_null) {
			continue;
		}
#ifndef PAC_SUM_NONCASCADING
		// Force flush both to totals64
		sdata[i]->Flush32(true);
		tdata[i]->Flush32(true);
#endif
		for (int j = 0; j < 64; j++) {
			tdata[i]->totals64[j] += sdata[i]->totals64[j];
		}
	}
}

static void PacSumFloatCascadeFinalize(Vector &states, AggregateInputData &aggr_input, Vector &result, idx_t count,
                                       idx_t offset) {
	auto sdata = FlatVector::GetData<PacSumFloatCascadeState *>(states);
	auto rdata = FlatVector::GetData<double>(result);
	auto &result_mask = FlatVector::Validity(result);

	double mi = 128.0;
	if (aggr_input.bind_data) {
		mi = aggr_input.bind_data->Cast<PacBindData>().mi;
	}
	thread_local std::mt19937_64 gen(std::random_device{}());

	for (idx_t i = 0; i < count; i++) {
		auto state = sdata[i];

		if (state->seen_null) {
			result_mask.SetInvalid(offset + i);
			continue;
		}

#ifndef PAC_SUM_NONCASCADING
		// Force flush to totals64
		state->Flush32(true);
#endif

		double noise = PacNoisySampleFrom64Counters(state->totals64, mi, gen);
		rdata[offset + i] = state->totals64[0] + noise;
	}
}

// Wrapper functions for float and double input types
static void PacSumFloatCascadeUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t input_count,
                                          data_ptr_t state_ptr, idx_t count) {
	PacSumFloatCascadeUpdate<float>(inputs, aggr, input_count, state_ptr, count);
}

static void PacSumFloatCascadeUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t input_count,
                                           data_ptr_t state_ptr, idx_t count) {
	PacSumFloatCascadeUpdate<double>(inputs, aggr, input_count, state_ptr, count);
}

static void PacSumFloatCascadeScatterFloat(Vector inputs[], AggregateInputData &aggr, idx_t input_count,
                                           Vector &states, idx_t count) {
	PacSumFloatCascadeScatterUpdate<float>(inputs, aggr, input_count, states, count);
}

static void PacSumFloatCascadeScatterDouble(Vector inputs[], AggregateInputData &aggr, idx_t input_count,
                                            Vector &states, idx_t count) {
	PacSumFloatCascadeScatterUpdate<double>(inputs, aggr, input_count, states, count);
}

// =========================
// Legacy non-cascaded float state (kept for compatibility)
// =========================

template <class FLOAT_TYPE>
struct PacSumFloatState {
	// Sums first for 64-byte alignment (SIMD-friendly)
	alignas(64) FLOAT_TYPE sums[64];
	bool seen_null;
};

template <class FLOAT_TYPE>
static idx_t PacSumFloatStateSize(const AggregateFunction &) {
	return sizeof(PacSumFloatState<FLOAT_TYPE>);
}

template <class FLOAT_TYPE>
static void PacSumFloatInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &state = *reinterpret_cast<PacSumFloatState<FLOAT_TYPE> *>(state_ptr);
	state.seen_null = false;
	for (int i = 0; i < 64; i++) {
		state.sums[i] = FLOAT_TYPE(0);
	}
}

template <class FLOAT_TYPE>
AUTOVECTORIZE
static void PacSumFloatUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr,
                              idx_t count) {
	D_ASSERT(input_count == 2);
	auto &state = *reinterpret_cast<PacSumFloatState<FLOAT_TYPE> *>(state_ptr);

	if (state.seen_null) {
		return; // Already NULL, skip processing
	}

	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<FLOAT_TYPE>(value_data);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state.seen_null = true;
			return; // Stop processing after seeing NULL
		}
		uint64_t key_hash = hashes[hash_idx];
		FLOAT_TYPE value = values[value_idx];

		for (int j = 0; j < 64; j++) {
#ifdef FILTER_WITH_MULT
			state.sums[j] += value * static_cast<FLOAT_TYPE>((key_hash >> j) & 1ULL);
#else
			uint64_t mask = BitToMask((key_hash >> j) & 1ULL);
			state.sums[j] += MaskValue(value, mask);
#endif
		}
	}
}

template <class FLOAT_TYPE>
AUTOVECTORIZE
static void PacSumFloatScatterUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states,
                                     idx_t count) {
	D_ASSERT(input_count == 2);

	UnifiedVectorFormat hash_data, value_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<FLOAT_TYPE>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacSumFloatState<FLOAT_TYPE> *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		auto state_idx = sdata.sel->get_index(i);
		auto state = state_ptrs[state_idx];

		if (state->seen_null) {
			continue; // Already NULL for this group
		}

		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state->seen_null = true;
			continue;
		}

		uint64_t key_hash = hashes[hash_idx];
		FLOAT_TYPE value = values[value_idx];

		for (int j = 0; j < 64; j++) {
#ifdef FILTER_WITH_MULT
			state->sums[j] += value * static_cast<FLOAT_TYPE>((key_hash >> j) & 1ULL);
#else
			uint64_t mask = BitToMask((key_hash >> j) & 1ULL);
			state->sums[j] += MaskValue(value, mask);
#endif
		}
	}
}

template <class FLOAT_TYPE>
static void PacSumFloatCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto sdata = FlatVector::GetData<PacSumFloatState<FLOAT_TYPE> *>(source);
	auto tdata = FlatVector::GetData<PacSumFloatState<FLOAT_TYPE> *>(target);
	for (idx_t i = 0; i < count; i++) {
		if (sdata[i]->seen_null) {
			tdata[i]->seen_null = true;
		}
		if (tdata[i]->seen_null) {
			continue;
		}
		for (int j = 0; j < 64; j++) {
			tdata[i]->sums[j] += sdata[i]->sums[j];
		}
	}
}

template <class FLOAT_TYPE>
static void PacSumFloatFinalize(Vector &states, AggregateInputData &aggr_input, Vector &result, idx_t count, idx_t offset) {
	auto sdata = FlatVector::GetData<PacSumFloatState<FLOAT_TYPE> *>(states);
	auto rdata = FlatVector::GetData<double>(result);
	auto &result_mask = FlatVector::Validity(result);

	// Get mi from bind data, default to 128.0
	double mi = 128.0;
	if (aggr_input.bind_data) {
		mi = aggr_input.bind_data->Cast<PacBindData>().mi;
	}
	thread_local std::mt19937_64 gen(std::random_device{}());

	for (idx_t i = 0; i < count; i++) {
		auto state = sdata[i];

		if (state->seen_null) {
			result_mask.SetInvalid(offset + i);
			continue;
		}

		// Convert sums to double array
		double sums_d[64];
		ToDoubleArray(state->sums, sums_d);
		// Compute noisy sampled result: sums[0] + noise
		double noise = PacNoisySampleFrom64Counters(sums_d, mi, gen);
		rdata[offset + i] = static_cast<double>(state->sums[0]) + noise;
	}
}

// =========================
// Integer pac_sum (cascaded multi-level accumulation for SIMD efficiency)
// =========================

#ifndef PAC_SUM_NONCASCADING
// Number of top bits reserved for overflow headroom at each level
// Flush threshold = 2^TopBits (we can safely sum that many values without overflow)
static constexpr int kTopBits8 = 3;   // threshold 8
static constexpr int kTopBits16 = 4;  // threshold 16
static constexpr int kTopBits32 = 5;  // threshold 32
static constexpr int kTopBits64 = 8;  // threshold 256

static constexpr uint32_t kFlushThreshold8 = 1 << kTopBits8;
static constexpr uint32_t kFlushThreshold16 = 1 << kTopBits16;
static constexpr uint32_t kFlushThreshold32 = 1 << kTopBits32;
static constexpr uint32_t kFlushThreshold64 = 1 << kTopBits64;

// Signed versions - check magnitude
static inline bool HasTopBitsSet8(int64_t value) {
	uint64_t abs_v = value >= 0 ? static_cast<uint64_t>(value) : static_cast<uint64_t>(-value);
	return (abs_v >> (8 - kTopBits8)) != 0;
}
static inline bool HasTopBitsSet16(int64_t value) {
	uint64_t abs_v = value >= 0 ? static_cast<uint64_t>(value) : static_cast<uint64_t>(-value);
	return (abs_v >> (16 - kTopBits16)) != 0;
}
static inline bool HasTopBitsSet32(int64_t value) {
	uint64_t abs_v = value >= 0 ? static_cast<uint64_t>(value) : static_cast<uint64_t>(-value);
	return (abs_v >> (32 - kTopBits32)) != 0;
}
static inline bool HasTopBitsSet64(int64_t value) {
	uint64_t abs_v = value >= 0 ? static_cast<uint64_t>(value) : static_cast<uint64_t>(-value);
	return (abs_v >> (64 - kTopBits64)) != 0;
}

// Unsigned versions - check value directly
static inline bool HasTopBitsSet8(uint64_t value) {
	return (value >> (8 - kTopBits8)) != 0;
}
static inline bool HasTopBitsSet16(uint64_t value) {
	return (value >> (16 - kTopBits16)) != 0;
}
static inline bool HasTopBitsSet32(uint64_t value) {
	return (value >> (32 - kTopBits32)) != 0;
}
static inline bool HasTopBitsSet64(uint64_t value) {
	return (value >> (64 - kTopBits64)) != 0;
}
#endif

// Signed integer state with cascaded levels
// Buffer: 1024 (totals128) + 512 (totals64) + 256 (totals32) + 128 (totals16) + 64 (totals8) + 64 (alignment) = 2048 bytes
struct PacSumSignedState {
	char totals_buf[2048];
	hugeint_t *totals128;
#ifndef PAC_SUM_NONCASCADING
	int64_t *totals64;
	int32_t *totals32;
	int16_t *totals16;
	int8_t *totals8;
	uint32_t count8;
	uint32_t count16;
	uint32_t count32;
	uint32_t count64;
#endif
	bool seen_null;

#ifndef PAC_SUM_NONCASCADING
	// Flush totals64 -> totals128
	inline void Flush64(bool force) {
		if (force || count64 >= kFlushThreshold64) {
			for (int i = 0; i < 64; i++) {
				totals128[i] += hugeint_t(totals64[i]);
				totals64[i] = 0;
			}
			count64 = 0;
		}
	}

	// Flush totals32 -> totals64, then cascade
	inline void Flush32(bool force) {
		if (force || count32 >= kFlushThreshold32) {
			for (int i = 0; i < 64; i++) {
				totals64[i] += totals32[i];
				totals32[i] = 0;
			}
			count64 += count32;
			count32 = 0;
			Flush64(force);
		}
	}

	// Flush totals16 -> totals32, then cascade
	inline void Flush16(bool force) {
		if (force || count16 >= kFlushThreshold16) {
			for (int i = 0; i < 64; i++) {
				totals32[i] += totals16[i];
				totals16[i] = 0;
			}
			count32 += count16;
			count16 = 0;
			Flush32(force);
		}
	}

	// Flush totals8 -> totals16, then cascade
	inline void Flush8(bool force) {
		if (force || count8 >= kFlushThreshold8) {
			for (int i = 0; i < 64; i++) {
				totals16[i] += totals8[i];
				totals8[i] = 0;
			}
			count16 += count8;
			count8 = 0;
			Flush16(force);
		}
	}
#endif
};

// Unsigned integer state with cascaded levels
// Buffer: 1024 (totals128) + 512 (totals64) + 256 (totals32) + 128 (totals16) + 64 (totals8) + 64 (alignment) = 2048 bytes
struct PacSumUnsignedState {
	char totals_buf[2048];
	hugeint_t *totals128;
#ifndef PAC_SUM_NONCASCADING
	uint64_t *totals64;
	uint32_t *totals32;
	uint16_t *totals16;
	uint8_t *totals8;
	uint32_t count8;
	uint32_t count16;
	uint32_t count32;
	uint32_t count64;
#endif
	bool seen_null;

#ifndef PAC_SUM_NONCASCADING
	// Flush totals64 -> totals128
	inline void Flush64(bool force) {
		if (force || count64 >= kFlushThreshold64) {
			for (int i = 0; i < 64; i++) {
				totals128[i] += hugeint_t(totals64[i]);
				totals64[i] = 0;
			}
			count64 = 0;
		}
	}

	// Flush totals32 -> totals64, then cascade
	inline void Flush32(bool force) {
		if (force || count32 >= kFlushThreshold32) {
			for (int i = 0; i < 64; i++) {
				totals64[i] += totals32[i];
				totals32[i] = 0;
			}
			count64 += count32;
			count32 = 0;
			Flush64(force);
		}
	}

	// Flush totals16 -> totals32, then cascade
	inline void Flush16(bool force) {
		if (force || count16 >= kFlushThreshold16) {
			for (int i = 0; i < 64; i++) {
				totals32[i] += totals16[i];
				totals16[i] = 0;
			}
			count32 += count16;
			count16 = 0;
			Flush32(force);
		}
	}

	// Flush totals8 -> totals16, then cascade
	inline void Flush8(bool force) {
		if (force || count8 >= kFlushThreshold8) {
			for (int i = 0; i < 64; i++) {
				totals16[i] += totals8[i];
				totals8[i] = 0;
			}
			count16 += count8;
			count8 = 0;
			Flush16(force);
		}
	}
#endif
};

// Type alias for backward compatibility with template code
template <class INPUT_TYPE, class ACC_TYPE>
using PacSumIntegerState = typename std::conditional<std::is_signed<INPUT_TYPE>::value, PacSumSignedState, PacSumUnsignedState>::type;

template <class INPUT_TYPE, class ACC_TYPE>
static idx_t PacSumIntegerStateSize(const AggregateFunction &) {
	return sizeof(PacSumIntegerState<INPUT_TYPE, ACC_TYPE>);
}

static void PacSumSignedInitialize(PacSumSignedState &state) {
	memset(state.totals_buf, 0, sizeof(state.totals_buf));

	// Compute aligned pointers into totals_buf
	// Layout: totals128 (1024) | totals64 (512) | totals32 (256) | totals16 (128) | totals8 (64)
	uintptr_t base = AlignTo64(reinterpret_cast<uintptr_t>(state.totals_buf));
	state.totals128 = reinterpret_cast<hugeint_t *>(base);
#ifndef PAC_SUM_NONCASCADING
	state.totals64 = reinterpret_cast<int64_t *>(base + 1024);
	state.totals32 = reinterpret_cast<int32_t *>(base + 1024 + 512);
	state.totals16 = reinterpret_cast<int16_t *>(base + 1024 + 512 + 256);
	state.totals8 = reinterpret_cast<int8_t *>(base + 1024 + 512 + 256 + 128);
	state.count8 = 0;
	state.count16 = 0;
	state.count32 = 0;
	state.count64 = 0;
#endif
	state.seen_null = false;
}

static void PacSumUnsignedInitialize(PacSumUnsignedState &state) {
	memset(state.totals_buf, 0, sizeof(state.totals_buf));

	// Compute aligned pointers into totals_buf
	// Layout: totals128 (1024) | totals64 (512) | totals32 (256) | totals16 (128) | totals8 (64)
	uintptr_t base = AlignTo64(reinterpret_cast<uintptr_t>(state.totals_buf));
	state.totals128 = reinterpret_cast<hugeint_t *>(base);
#ifndef PAC_SUM_NONCASCADING
	state.totals64 = reinterpret_cast<uint64_t *>(base + 1024);
	state.totals32 = reinterpret_cast<uint32_t *>(base + 1024 + 512);
	state.totals16 = reinterpret_cast<uint16_t *>(base + 1024 + 512 + 256);
	state.totals8 = reinterpret_cast<uint8_t *>(base + 1024 + 512 + 256 + 128);
	state.count8 = 0;
	state.count16 = 0;
	state.count32 = 0;
	state.count64 = 0;
#endif
	state.seen_null = false;
}

template <class INPUT_TYPE, class ACC_TYPE>
static void PacSumIntegerInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	if (std::is_signed<INPUT_TYPE>::value) {
		PacSumSignedInitialize(*reinterpret_cast<PacSumSignedState *>(state_ptr));
	} else {
		PacSumUnsignedInitialize(*reinterpret_cast<PacSumUnsignedState *>(state_ptr));
	}
}

// Internal update for a single value - signed
static inline void PacSumSignedUpdateInternal(PacSumSignedState &state, uint64_t key_hash, int64_t value) {
#ifdef PAC_SUM_NONCASCADING
	AddToTotals128Signed(state.totals128, value, key_hash);
#else
	if (!HasTopBitsSet8(value)) {
		AddToTotals8Signed(state.totals8, value, key_hash);
		state.count8++;
		state.Flush8(false);
	} else if (!HasTopBitsSet16(value)) {
		AddToTotals16Signed(state.totals16, value, key_hash);
		state.count16++;
		state.Flush16(false);
	} else if (!HasTopBitsSet32(value)) {
		AddToTotals32Signed(state.totals32, value, key_hash);
		state.count32++;
		state.Flush32(false);
	} else {
		AddToTotals64Signed(state.totals64, value, key_hash);
		state.count64++;
		state.Flush64(false);
	}
#endif
}

// Internal update for a single value - unsigned
static inline void PacSumUnsignedUpdateInternal(PacSumUnsignedState &state, uint64_t key_hash, uint64_t value) {
#ifdef PAC_SUM_NONCASCADING
	AddToTotals128Unsigned(state.totals128, value, key_hash);
#else
	if (!HasTopBitsSet8(value)) {
		AddToTotals8Unsigned(state.totals8, value, key_hash);
		state.count8++;
		state.Flush8(false);
	} else if (!HasTopBitsSet16(value)) {
		AddToTotals16Unsigned(state.totals16, value, key_hash);
		state.count16++;
		state.Flush16(false);
	} else if (!HasTopBitsSet32(value)) {
		AddToTotals32Unsigned(state.totals32, value, key_hash);
		state.count32++;
		state.Flush32(false);
	} else {
		AddToTotals64Unsigned(state.totals64, value, key_hash);
		state.count64++;
		state.Flush64(false);
	}
#endif
}

// Signed integer update
template <class INPUT_TYPE>
static void PacSumSignedUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr,
                               idx_t count) {
	D_ASSERT(input_count == 2);
	auto &state = *reinterpret_cast<PacSumSignedState *>(state_ptr);
	if (state.seen_null) {
		return;
	}

	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state.seen_null = true;
			return;
		}
		PacSumSignedUpdateInternal(state, hashes[hash_idx], static_cast<int64_t>(values[value_idx]));
	}
}

// Unsigned integer update
template <class INPUT_TYPE>
static void PacSumUnsignedUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr,
                                 idx_t count) {
	D_ASSERT(input_count == 2);
	auto &state = *reinterpret_cast<PacSumUnsignedState *>(state_ptr);
	if (state.seen_null) {
		return;
	}

	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state.seen_null = true;
			return;
		}
		PacSumUnsignedUpdateInternal(state, hashes[hash_idx], static_cast<uint64_t>(values[value_idx]));
	}
}

// Wrapper template that dispatches to signed or unsigned based on INPUT_TYPE
template <class INPUT_TYPE, class ACC_TYPE>
static void PacSumIntegerUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                data_ptr_t state_ptr, idx_t count) {
	if (std::is_signed<INPUT_TYPE>::value) {
		PacSumSignedUpdate<INPUT_TYPE>(inputs, aggr_input_data, input_count, state_ptr, count);
	} else {
		PacSumUnsignedUpdate<INPUT_TYPE>(inputs, aggr_input_data, input_count, state_ptr, count);
	}
}

// Signed integer scatter update
template <class INPUT_TYPE>
static void PacSumSignedScatterUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states,
                                      idx_t count) {
	D_ASSERT(input_count == 2);

	UnifiedVectorFormat hash_data, value_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacSumSignedState *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		auto state_idx = sdata.sel->get_index(i);
		auto state = state_ptrs[state_idx];

		if (state->seen_null) {
			continue;
		}
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state->seen_null = true;
			continue;
		}
		PacSumSignedUpdateInternal(*state, hashes[hash_idx], static_cast<int64_t>(values[value_idx]));
	}
}

// Unsigned integer scatter update
template <class INPUT_TYPE>
static void PacSumUnsignedScatterUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states,
                                        idx_t count) {
	D_ASSERT(input_count == 2);

	UnifiedVectorFormat hash_data, value_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacSumUnsignedState *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		auto state_idx = sdata.sel->get_index(i);
		auto state = state_ptrs[state_idx];

		if (state->seen_null) {
			continue;
		}
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state->seen_null = true;
			continue;
		}
		PacSumUnsignedUpdateInternal(*state, hashes[hash_idx], static_cast<uint64_t>(values[value_idx]));
	}
}

// Wrapper template that dispatches to signed or unsigned based on INPUT_TYPE
template <class INPUT_TYPE, class ACC_TYPE>
static void PacSumIntegerScatterUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                       Vector &states, idx_t count) {
	if (std::is_signed<INPUT_TYPE>::value) {
		PacSumSignedScatterUpdate<INPUT_TYPE>(inputs, aggr_input_data, input_count, states, count);
	} else {
		PacSumUnsignedScatterUpdate<INPUT_TYPE>(inputs, aggr_input_data, input_count, states, count);
	}
}

template <class INPUT_TYPE, class ACC_TYPE>
static void PacSumIntegerCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto sdata = FlatVector::GetData<PacSumIntegerState<INPUT_TYPE, ACC_TYPE> *>(source);
	auto tdata = FlatVector::GetData<PacSumIntegerState<INPUT_TYPE, ACC_TYPE> *>(target);
	for (idx_t i = 0; i < count; i++) {
		if (sdata[i]->seen_null) {
			tdata[i]->seen_null = true;
		}
		if (tdata[i]->seen_null) {
			continue;
		}
#ifndef PAC_SUM_NONCASCADING
		// Force flush both states to totals128
		sdata[i]->Flush8(true);
		tdata[i]->Flush8(true);
#endif
		// Combine totals128
		for (int j = 0; j < 64; j++) {
			tdata[i]->totals128[j] += sdata[i]->totals128[j];
		}
	}
}

template <class INPUT_TYPE, class ACC_TYPE>
static void PacSumIntegerFinalize(Vector &states, AggregateInputData &aggr_input, Vector &result, idx_t count, idx_t offset) {
	auto sdata = FlatVector::GetData<PacSumIntegerState<INPUT_TYPE, ACC_TYPE> *>(states);
	auto rdata = FlatVector::GetData<ACC_TYPE>(result);
	auto &result_mask = FlatVector::Validity(result);

	// Get mi from bind data, default to 128.0
	double mi = 128.0;
	if (aggr_input.bind_data) {
		mi = aggr_input.bind_data->Cast<PacBindData>().mi;
	}
	thread_local std::mt19937_64 gen(std::random_device{}());

	for (idx_t i = 0; i < count; i++) {
		auto state = sdata[i];

		if (state->seen_null) {
			result_mask.SetInvalid(offset + i);
			continue;
		}

#ifndef PAC_SUM_NONCASCADING
		// Force flush all levels to totals128
		state->Flush8(true);
#endif

		// Convert totals128 to double array
		double totals_d[64];
		ToDoubleArray(state->totals128, totals_d);
		// Compute noisy sampled result: totals128[0] + noise
		double noise = PacNoisySampleFrom64Counters(totals_d, mi, gen);
		double result_d = ToDouble(state->totals128[0]) + noise;
		// Cast back to accumulator type
		rdata[offset + i] = FromDouble<ACC_TYPE>(result_d);
	}
}

// =========================
// Explicit instantiations for float types
// =========================
static void PacSumUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                              idx_t count) {
	PacSumFloatUpdate<float>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                               idx_t count) {
	PacSumFloatUpdate<double>(inputs, aggr, input_count, state, count);
}
static void PacSumScatterFloat(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                               idx_t count) {
	PacSumFloatScatterUpdate<float>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterDouble(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                idx_t count) {
	PacSumFloatScatterUpdate<double>(inputs, aggr, input_count, states, count);
}

// =========================
// Explicit instantiations for signed integer types (accumulate to hugeint_t)
// =========================
static void PacSumUpdateTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                idx_t count) {
	PacSumIntegerUpdate<int8_t, hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumIntegerUpdate<int16_t, hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateInteger(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                idx_t count) {
	PacSumIntegerUpdate<int32_t, hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateBigInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                               idx_t count) {
	PacSumIntegerUpdate<int64_t, hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                idx_t count) {
	PacSumIntegerUpdate<hugeint_t, hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumScatterTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                 idx_t count) {
	PacSumIntegerScatterUpdate<int8_t, hugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumIntegerScatterUpdate<int16_t, hugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterInteger(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                 idx_t count) {
	PacSumIntegerScatterUpdate<int32_t, hugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterBigInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                idx_t count) {
	PacSumIntegerScatterUpdate<int64_t, hugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                 idx_t count) {
	PacSumIntegerScatterUpdate<hugeint_t, hugeint_t>(inputs, aggr, input_count, states, count);
}

// =========================
// Explicit instantiations for unsigned integer types (accumulate to hugeint_t like DuckDB's SUM)
// =========================
static void PacSumUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumIntegerUpdate<uint8_t, hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                  idx_t count) {
	PacSumIntegerUpdate<uint16_t, hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumIntegerUpdate<uint32_t, hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                idx_t count) {
	PacSumIntegerUpdate<uint64_t, hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumIntegerUpdate<uhugeint_t, hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumScatterUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumIntegerScatterUpdate<uint8_t, hugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                   idx_t count) {
	PacSumIntegerScatterUpdate<uint16_t, hugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUInteger(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumIntegerScatterUpdate<uint32_t, hugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                 idx_t count) {
	PacSumIntegerScatterUpdate<uint64_t, hugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumIntegerScatterUpdate<uhugeint_t, hugeint_t>(inputs, aggr, input_count, states, count);
}

// UHUGEINT -> DOUBLE (like DuckDB's SUM): reads uhugeint_t, accumulates as double
static void PacSumUpdateUHugeIntToDouble(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr,
                                         idx_t count) {
	D_ASSERT(input_count == 2);
	auto &state = *reinterpret_cast<PacSumFloatState<double> *>(state_ptr);

	if (state.seen_null) {
		return;
	}

	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<uhugeint_t>(value_data);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state.seen_null = true;
			return;
		}
		uint64_t key_hash = hashes[hash_idx];
		double value = Uhugeint::Cast<double>(values[value_idx]);
		for (idx_t j = 0; j < 64; j++) {
			double mask = -static_cast<double>((key_hash >> j) & 1);
			state.sums[j] += (value * mask) - (value * (mask + 1.0));
		}
	}
}

static void PacSumScatterUHugeIntToDouble(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states,
                                          idx_t count) {
	D_ASSERT(input_count == 2);

	UnifiedVectorFormat hash_data, value_data, state_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	states.ToUnifiedFormat(count, state_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<uhugeint_t>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacSumFloatState<double> *>(state_data);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		auto state_idx = state_data.sel->get_index(i);
		auto &state = *state_ptrs[state_idx];

		if (state.seen_null) {
			continue;
		}
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state.seen_null = true;
			continue;
		}
		uint64_t key_hash = hashes[hash_idx];
		double value = Uhugeint::Cast<double>(values[value_idx]);
		for (idx_t j = 0; j < 64; j++) {
			double mask = -static_cast<double>((key_hash >> j) & 1);
			state.sums[j] += (value * mask) - (value * (mask + 1.0));
		}
	}
}

static void PacSumFloatCascadeInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &state = *reinterpret_cast<PacSumFloatCascadeState *>(state_ptr);
	memset(state.totals_buf, 0, sizeof(state.totals_buf));

	// Compute aligned pointers into totals_buf
	uintptr_t base = AlignTo64(reinterpret_cast<uintptr_t>(state.totals_buf));
	state.totals64 = reinterpret_cast<double *>(base);  // 64 * 8 = 512 bytes
#ifndef PAC_SUM_NONCASCADING
	state.totals32 = reinterpret_cast<float *>(base + 512);  // 64 * 4 = 256 bytes
	state.count32 = 0;
#endif
	state.seen_null = false;
}

void RegisterPacSumFunctions(ExtensionLoader &loader) {
    // Register pac_sum aggregate function set
	// Input: (UBIGINT key_hash, <numeric> value, optional DOUBLE mi)
	// Output: HUGEINT for all integers (like DuckDB's SUM), DOUBLE for floats
	// Supports all numeric types that SUM supports
	AggregateFunctionSet pac_sum_set("pac_sum");

	// Helper macro to add both 2-arg and 3-arg versions
#define ADD_PAC_SUM_INT(INPUT_TYPE, ACC_TYPE, RESULT_TYPE, ScatterFn, CombineFn, FinalizeFn, UpdateFn) \
	pac_sum_set.AddFunction(AggregateFunction( \
	    "pac_sum", {LogicalType::UBIGINT, INPUT_TYPE}, RESULT_TYPE, \
	    PacSumIntegerStateSize<ACC_TYPE, ACC_TYPE>, PacSumIntegerInitialize<ACC_TYPE, ACC_TYPE>, ScatterFn, \
	    CombineFn, FinalizeFn, FunctionNullHandling::DEFAULT_NULL_HANDLING, UpdateFn, PacSumBind)); \
	pac_sum_set.AddFunction(AggregateFunction( \
	    "pac_sum", {LogicalType::UBIGINT, INPUT_TYPE, LogicalType::DOUBLE}, RESULT_TYPE, \
	    PacSumIntegerStateSize<ACC_TYPE, ACC_TYPE>, PacSumIntegerInitialize<ACC_TYPE, ACC_TYPE>, ScatterFn, \
	    CombineFn, FinalizeFn, FunctionNullHandling::DEFAULT_NULL_HANDLING, UpdateFn, PacSumBind))

	// Signed integers (accumulate to hugeint_t, return HUGEINT)
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::TINYINT}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<int8_t, hugeint_t>, PacSumIntegerInitialize<int8_t, hugeint_t>, PacSumScatterTinyInt,
	    PacSumCombineTinyInt, PacSumFinalizeTinyInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateTinyInt, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::TINYINT, LogicalType::DOUBLE}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<int8_t, hugeint_t>, PacSumIntegerInitialize<int8_t, hugeint_t>, PacSumScatterTinyInt,
	    PacSumCombineTinyInt, PacSumFinalizeTinyInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateTinyInt, PacSumBind));

	// BOOLEAN treated as TINYINT (like DuckDB's SUM)
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::BOOLEAN}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<int8_t, hugeint_t>, PacSumIntegerInitialize<int8_t, hugeint_t>, PacSumScatterTinyInt,
	    PacSumCombineTinyInt, PacSumFinalizeTinyInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateTinyInt, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::BOOLEAN, LogicalType::DOUBLE}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<int8_t, hugeint_t>, PacSumIntegerInitialize<int8_t, hugeint_t>, PacSumScatterTinyInt,
	    PacSumCombineTinyInt, PacSumFinalizeTinyInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateTinyInt, PacSumBind));

	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::SMALLINT}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<int16_t, hugeint_t>, PacSumIntegerInitialize<int16_t, hugeint_t>, PacSumScatterSmallInt,
	    PacSumCombineSmallInt, PacSumFinalizeSmallInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateSmallInt, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::SMALLINT, LogicalType::DOUBLE}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<int16_t, hugeint_t>, PacSumIntegerInitialize<int16_t, hugeint_t>, PacSumScatterSmallInt,
	    PacSumCombineSmallInt, PacSumFinalizeSmallInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateSmallInt, PacSumBind));

	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::INTEGER}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<int32_t, hugeint_t>, PacSumIntegerInitialize<int32_t, hugeint_t>, PacSumScatterInteger,
	    PacSumCombineInteger, PacSumFinalizeInteger, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateInteger, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::INTEGER, LogicalType::DOUBLE}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<int32_t, hugeint_t>, PacSumIntegerInitialize<int32_t, hugeint_t>, PacSumScatterInteger,
	    PacSumCombineInteger, PacSumFinalizeInteger, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateInteger, PacSumBind));

	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::BIGINT}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<int64_t, hugeint_t>, PacSumIntegerInitialize<int64_t, hugeint_t>, PacSumScatterBigInt,
	    PacSumCombineBigInt, PacSumFinalizeBigInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateBigInt, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::BIGINT, LogicalType::DOUBLE}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<int64_t, hugeint_t>, PacSumIntegerInitialize<int64_t, hugeint_t>, PacSumScatterBigInt,
	    PacSumCombineBigInt, PacSumFinalizeBigInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateBigInt, PacSumBind));

	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::HUGEINT}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<hugeint_t, hugeint_t>, PacSumIntegerInitialize<hugeint_t, hugeint_t>,
	    PacSumScatterHugeInt, PacSumCombineHugeInt, PacSumFinalizeHugeInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateHugeInt, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::HUGEINT, LogicalType::DOUBLE}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<hugeint_t, hugeint_t>, PacSumIntegerInitialize<hugeint_t, hugeint_t>,
	    PacSumScatterHugeInt, PacSumCombineHugeInt, PacSumFinalizeHugeInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateHugeInt, PacSumBind));

	// Unsigned integers (accumulate to hugeint_t like DuckDB's SUM, return HUGEINT)
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UTINYINT}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<uint8_t, hugeint_t>, PacSumIntegerInitialize<uint8_t, hugeint_t>,
	    PacSumScatterUTinyInt, PacSumCombineUTinyInt, PacSumFinalizeUTinyInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUTinyInt, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UTINYINT, LogicalType::DOUBLE}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<uint8_t, hugeint_t>, PacSumIntegerInitialize<uint8_t, hugeint_t>,
	    PacSumScatterUTinyInt, PacSumCombineUTinyInt, PacSumFinalizeUTinyInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUTinyInt, PacSumBind));

	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::USMALLINT}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<uint16_t, hugeint_t>, PacSumIntegerInitialize<uint16_t, hugeint_t>,
	    PacSumScatterUSmallInt, PacSumCombineUSmallInt, PacSumFinalizeUSmallInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUSmallInt, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::USMALLINT, LogicalType::DOUBLE}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<uint16_t, hugeint_t>, PacSumIntegerInitialize<uint16_t, hugeint_t>,
	    PacSumScatterUSmallInt, PacSumCombineUSmallInt, PacSumFinalizeUSmallInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUSmallInt, PacSumBind));

	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UINTEGER}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<uint32_t, hugeint_t>, PacSumIntegerInitialize<uint32_t, hugeint_t>,
	    PacSumScatterUInteger, PacSumCombineUInteger, PacSumFinalizeUInteger,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUInteger, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UINTEGER, LogicalType::DOUBLE}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<uint32_t, hugeint_t>, PacSumIntegerInitialize<uint32_t, hugeint_t>,
	    PacSumScatterUInteger, PacSumCombineUInteger, PacSumFinalizeUInteger,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUInteger, PacSumBind));

	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UBIGINT}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<uint64_t, hugeint_t>, PacSumIntegerInitialize<uint64_t, hugeint_t>,
	    PacSumScatterUBigInt, PacSumCombineUBigInt, PacSumFinalizeUBigInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUBigInt, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::DOUBLE}, LogicalType::HUGEINT,
	    PacSumIntegerStateSize<uint64_t, hugeint_t>, PacSumIntegerInitialize<uint64_t, hugeint_t>,
	    PacSumScatterUBigInt, PacSumCombineUBigInt, PacSumFinalizeUBigInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUBigInt, PacSumBind));

	// UHUGEINT uses float state and returns DOUBLE (like DuckDB's SUM)
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UHUGEINT}, LogicalType::DOUBLE,
	    PacSumFloatStateSize<double>, PacSumFloatInitialize<double>,
	    PacSumScatterUHugeIntToDouble, PacSumCombineDouble, PacSumFinalizeDouble,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUHugeIntToDouble, PacSumBind));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UHUGEINT, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	    PacSumFloatStateSize<double>, PacSumFloatInitialize<double>,
	    PacSumScatterUHugeIntToDouble, PacSumCombineDouble, PacSumFinalizeDouble,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUHugeIntToDouble, PacSumBind));

	// Floating point (use cascaded float32/float64 state, return DOUBLE)
	pac_sum_set.AddFunction(
	    AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::FLOAT}, LogicalType::DOUBLE,
	                      PacSumFloatCascadeStateSize, PacSumFloatCascadeInitialize, PacSumFloatCascadeScatterFloat,
	                      PacSumFloatCascadeCombine, PacSumFloatCascadeFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                      PacSumFloatCascadeUpdateFloat, PacSumBind));
	pac_sum_set.AddFunction(
	    AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::FLOAT, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      PacSumFloatCascadeStateSize, PacSumFloatCascadeInitialize, PacSumFloatCascadeScatterFloat,
	                      PacSumFloatCascadeCombine, PacSumFloatCascadeFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                      PacSumFloatCascadeUpdateFloat, PacSumBind));

	pac_sum_set.AddFunction(
	    AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      PacSumFloatCascadeStateSize, PacSumFloatCascadeInitialize, PacSumFloatCascadeScatterDouble,
	                      PacSumFloatCascadeCombine, PacSumFloatCascadeFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                      PacSumFloatCascadeUpdateDouble, PacSumBind));
	pac_sum_set.AddFunction(
	    AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      PacSumFloatCascadeStateSize, PacSumFloatCascadeInitialize, PacSumFloatCascadeScatterDouble,
	                      PacSumFloatCascadeCombine, PacSumFloatCascadeFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                      PacSumFloatCascadeUpdateDouble, PacSumBind));

#undef ADD_PAC_SUM_INT

	loader.RegisterFunction(pac_sum_set);
}

// Forwarding wrappers (non-static) for combine and finalize functions declared in header
void PacSumCombineFloat(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumFloatCombine<float>(source, target, aggr, count);
}

void PacSumCombineDouble(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumFloatCombine<double>(source, target, aggr, count);
}

void PacSumCombineTinyInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumIntegerCombine<int8_t, hugeint_t>(source, target, aggr, count);
}
void PacSumCombineSmallInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumIntegerCombine<int16_t, hugeint_t>(source, target, aggr, count);
}
void PacSumCombineInteger(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumIntegerCombine<int32_t, hugeint_t>(source, target, aggr, count);
}
void PacSumCombineBigInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumIntegerCombine<int64_t, hugeint_t>(source, target, aggr, count);
}
void PacSumCombineHugeInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumIntegerCombine<hugeint_t, hugeint_t>(source, target, aggr, count);
}
void PacSumCombineUTinyInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumIntegerCombine<uint8_t, hugeint_t>(source, target, aggr, count);
}
void PacSumCombineUSmallInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumIntegerCombine<uint16_t, hugeint_t>(source, target, aggr, count);
}
void PacSumCombineUInteger(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumIntegerCombine<uint32_t, hugeint_t>(source, target, aggr, count);
}
void PacSumCombineUBigInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumIntegerCombine<uint64_t, hugeint_t>(source, target, aggr, count);
}
void PacSumCombineUHugeInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
    PacSumIntegerCombine<uhugeint_t, hugeint_t>(source, target, aggr, count);
}

// Finalize wrappers
void PacSumFinalizeFloat(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumFloatFinalize<float>(states, aggr, result, count, offset);
}
void PacSumFinalizeDouble(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumFloatFinalize<double>(states, aggr, result, count, offset);
}
void PacSumFinalizeTinyInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumIntegerFinalize<int8_t, hugeint_t>(states, aggr, result, count, offset);
}
void PacSumFinalizeSmallInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumIntegerFinalize<int16_t, hugeint_t>(states, aggr, result, count, offset);
}
void PacSumFinalizeInteger(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumIntegerFinalize<int32_t, hugeint_t>(states, aggr, result, count, offset);
}
void PacSumFinalizeBigInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumIntegerFinalize<int64_t, hugeint_t>(states, aggr, result, count, offset);
}
void PacSumFinalizeHugeInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumIntegerFinalize<hugeint_t, hugeint_t>(states, aggr, result, count, offset);
}
void PacSumFinalizeUTinyInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumIntegerFinalize<uint8_t, hugeint_t>(states, aggr, result, count, offset);
}
void PacSumFinalizeUSmallInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumIntegerFinalize<uint16_t, hugeint_t>(states, aggr, result, count, offset);
}
void PacSumFinalizeUInteger(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumIntegerFinalize<uint32_t, hugeint_t>(states, aggr, result, count, offset);
}
void PacSumFinalizeUBigInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumIntegerFinalize<uint64_t, hugeint_t>(states, aggr, result, count, offset);
}
void PacSumFinalizeUHugeInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
    PacSumIntegerFinalize<uhugeint_t, hugeint_t>(states, aggr, result, count, offset);
}

} // namespace duckdb

