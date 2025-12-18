#include "include/pac_aggregate.hpp"

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/aggregate_function.hpp"

#include <random>
#include <cmath>
#include <limits>
#include <cstring>

// Enable AVX2 vectorization for update functions
#define AUTOVECTORIZE __attribute__((target("avx2")))

// Every argument to pac_aggregate is the output of a query evaluated on a random subsample of the privacy unit

namespace duckdb {

// ============================================================================
// pac_count aggregate function
// ============================================================================
// State: 64 counters, one for each bit position
// Update: for each input key_hash, increment count[i] if bit i is set
// Finalize: compute the variance of the 64 counters
//
// Uses SIMD-friendly intermediate accumulators (small_totals) that get
// flushed to the main totals every 256 updates.

struct PacCountState {
	uint8_t update_count;     // Counts updates, flushes when wraps to 0
	uint64_t small_totals[8]; // SIMD-friendly intermediate accumulators
	uint64_t totals[64];      // Final counters

	void Flush() {
		const uint8_t *small = reinterpret_cast<const uint8_t *>(small_totals);
		for (int i = 0; i < 64; i++) {
			totals[i] += small[i];
		}
		memset(small_totals, 0, sizeof(small_totals));
	}
};

// SIMD-friendly mask: extracts bits at positions 0, 8, 16, 24, 32, 40, 48, 56
#define PAC_COUNT_MASK                                                                                                 \
	((1ULL << 0) | (1ULL << 8) | (1ULL << 16) | (1ULL << 24) | (1ULL << 32) | (1ULL << 40) | (1ULL << 48) |            \
	 (1ULL << 56))

// Explicit function definitions for pac_count (used by manual AggregateFunction registration)
static idx_t PacCountStateSize(const AggregateFunction &) {
	return sizeof(PacCountState);
}

static void PacCountInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);
	state.update_count = 0;
	memset(state.small_totals, 0, sizeof(state.small_totals));
	memset(state.totals, 0, sizeof(state.totals));
}

// Simple update for pac_count (ungrouped aggregation)
static void PacCountUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_count == 1);
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);
	auto &input = inputs[0];

	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);

	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);
		if (!idata.validity.RowIsValid(idx)) {
			continue;
		}
		uint64_t key_hash = input_data[idx];
		for (int j = 0; j < 8; j++) {
			state.small_totals[j] += (key_hash >> j) & PAC_COUNT_MASK;
		}
		if (++state.update_count == 0) {
			// update_count wrapped (256 updates), flush to totals
			state.Flush();
		}
	}
}

// Scatter update for pac_count (grouped aggregation)
AUTOVECTORIZE
static void PacCountScatterUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states,
                                  idx_t count) {
	D_ASSERT(input_count == 1);

	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);

	UnifiedVectorFormat sdata;
	states.ToUnifiedFormat(count, sdata);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacCountState *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto input_idx = idata.sel->get_index(i);
		if (!idata.validity.RowIsValid(input_idx)) {
			continue;
		}

		auto state_idx = sdata.sel->get_index(i);
		auto state = state_ptrs[state_idx];

		uint64_t key_hash = input_data[input_idx];
		for (int j = 0; j < 8; j++) {
			state->small_totals[j] += (key_hash >> j) & PAC_COUNT_MASK;
		}
		if (++state->update_count == 0) {
			state->Flush();
		}
	}
}

static void PacCountCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto sdata = FlatVector::GetData<PacCountState *>(source);
	auto tdata = FlatVector::GetData<PacCountState *>(target);
	for (idx_t i = 0; i < count; i++) {
		auto src = sdata[i];
		auto tgt = tdata[i];
		// Flush any unflushed small_totals
		if (src->update_count != 0) {
			src->Flush();
		}
		if (tgt->update_count != 0) {
			tgt->Flush();
		}
		for (int j = 0; j < 64; j++) {
			tgt->totals[j] += src->totals[j];
		}
	}
}

// Finalize: compute noisy sample from the 64 counters
static double PacNoisySampleFrom64Counters(const uint64_t counters[64], double mi, std::mt19937_64 &gen) {
    constexpr int N = 64;
    // Compute sum and sum-of-squares in one pass (auto-vectorizable?)
    double S = 0.0;
    double Q = 0.0;
    for (int i = 0; i < N; ++i) {
        double v = (double)counters[i];
        S += v;
        Q += v * v;
    }

    // Pick random index J in [0, N-1]
    std::uniform_int_distribution<int> uid(0, N - 1);
    int J = uid(gen);
    double yJ = (double)counters[J];

    // Compute leave-one-out sums
    const double n1 = double(N - 1);           // 63.0
    double sum_minus = S - yJ;                 // sum excluding yJ
    double sumsq_minus = Q - yJ * yJ;          // sumsq excluding yJ

    // Leave-one-out population variance (divide by n1)
    double sigma2 = 0.0;
    if (n1 > 1.0) {
        // variance = (sumsq_minus - sum_minus^2 / n1) / n1
        double tmp = sumsq_minus - (sum_minus * sum_minus) / n1;
        // Numeric safety: clamp small negative rounding error to zero
        if (tmp <= 0.0) {
            sigma2 = 0.0;
        } else {
            sigma2 = tmp / n1;
        }
    }

    double delta = sigma2 / (2.0 * mi);
    if (delta <= 0.0 || !std::isfinite(delta)) {
        return yJ;
    }

    // Sample normal(0, sqrt(delta)).
    // this can probably be optimized further
    std::normal_distribution<double> gauss(0.0, std::sqrt(delta));
    return yJ + gauss(gen);
}


static void PacCountFinalize(Vector &states, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto sdata = FlatVector::GetData<PacCountState *>(states);
    auto rdata = FlatVector::GetData<double>(result);

    // Default mi (fixme)
    const double mi_default = 128.0;
    thread_local std::mt19937_64 gen(std::random_device{}());

    for (idx_t i = 0; i < count; i++) {
        auto state = sdata[i];
        if (state->update_count != 0) {
            state->Flush();
        }
        // Compute noisy sampled result from the 64 counters
        double res = PacNoisySampleFrom64Counters(state->totals, mi_default, gen);
        rdata[offset + i] = res;
    }
}

// ============================================================================
// pac_sum aggregate function
// ============================================================================
// State: 64 sums, one for each bit position
// Update: for each (key_hash, value), add value to sums[i] if bit i of key_hash is set
// Finalize: compute the variance of the 64 sums
//
// For integers: uses two-level accumulation with small_totals (input type) and totals (large type)
// For floats: uses direct accumulation in the same type

// Helper to convert any numeric type to double for variance calculation
template <class T>
static inline double ToDouble(const T &val) {
	return static_cast<double>(val);
}

template <>
inline double ToDouble<hugeint_t>(const hugeint_t &val) {
	return Hugeint::Cast<double>(val);
}

template <>
inline double ToDouble<uhugeint_t>(const uhugeint_t &val) {
	return Uhugeint::Cast<double>(val);
}

// =========================
// Branchless mask helpers
// =========================

// Convert a bit (0 or 1) to a mask (0x0 or 0xFF...F)
static inline uint64_t BitToMask(uint64_t bit) {
	return ~(bit - 1ULL);
}

// Apply mask to value - generic template for integers
template <class T>
static inline T MaskValue(T value, uint64_t mask) {
	return static_cast<T>(value & static_cast<T>(mask));
}

// Specialization for float
template <>
inline float MaskValue<float>(float value, uint64_t mask) {
	union { float f; uint32_t u; } v;
	v.f = value;
	v.u &= static_cast<uint32_t>(mask);
	return v.f;
}

// Specialization for double
template <>
inline double MaskValue<double>(double value, uint64_t mask) {
	union { double d; uint64_t u; } v;
	v.d = value;
	v.u &= mask;
	return v.d;
}

// Specialization for hugeint_t
template <>
inline hugeint_t MaskValue<hugeint_t>(hugeint_t value, uint64_t mask) {
	return hugeint_t(value.lower & mask, value.upper & static_cast<int64_t>(mask));
}

// Specialization for uhugeint_t
template <>
inline uhugeint_t MaskValue<uhugeint_t>(uhugeint_t value, uint64_t mask) {
	return uhugeint_t(value.lower & mask, value.upper & mask);
}

// =========================
// Floating point pac_sum (direct accumulation)
// =========================

template <class FLOAT_TYPE>
struct PacSumFloatState {
	bool seen_null;
	FLOAT_TYPE sums[64];
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
			uint64_t mask = BitToMask((key_hash >> j) & 1ULL);
			state.sums[j] += MaskValue(value, mask);
		}
	}
}

template <class FLOAT_TYPE>
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
			uint64_t mask = BitToMask((key_hash >> j) & 1ULL);
			state->sums[j] += MaskValue(value, mask);
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
static void PacSumFloatFinalize(Vector &states, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	auto sdata = FlatVector::GetData<PacSumFloatState<FLOAT_TYPE> *>(states);
	auto rdata = FlatVector::GetData<double>(result);
	auto &result_mask = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto state = sdata[i];

		if (state->seen_null) {
			result_mask.SetInvalid(offset + i);
			continue;
		}

		double total = 0.0;
		for (int j = 0; j < 64; j++) {
			total += static_cast<double>(state->sums[j]);
		}
		double mean = total / 64.0;

		double variance = 0.0;
		for (int j = 0; j < 64; j++) {
			double diff = static_cast<double>(state->sums[j]) - mean;
			variance += diff * diff;
		}
		variance /= 64.0;

		rdata[offset + i] = variance;
	}
}

// =========================
// Integer pac_sum (two-level accumulation)
// =========================

template <class INPUT_TYPE, class ACC_TYPE>
struct PacSumIntegerState {
	bool seen_null;
	uint32_t update_count;
	INPUT_TYPE small_totals[64];
	ACC_TYPE totals[64];

	void Flush() {
		for (int i = 0; i < 64; i++) {
			totals[i] += static_cast<ACC_TYPE>(small_totals[i]);
			small_totals[i] = INPUT_TYPE(0);
		}
		update_count = 0;
	}
};

// Check if value has top B bits set (meaning it's large and could cause overflow)
// B values: uint8=3, uint16=5, uint32=8, uint64=16, hugeint=16
// Flush threshold is (1 << B): 8, 32, 256, 65536, 65536

// Flush thresholds for each type
static inline constexpr uint32_t FlushThreshold(uint8_t) { return 1 << 3; }   // 8
static inline constexpr uint32_t FlushThreshold(int8_t) { return 1 << 3; }
static inline constexpr uint32_t FlushThreshold(uint16_t) { return 1 << 5; }  // 32
static inline constexpr uint32_t FlushThreshold(int16_t) { return 1 << 5; }
static inline constexpr uint32_t FlushThreshold(uint32_t) { return 1 << 8; }  // 256
static inline constexpr uint32_t FlushThreshold(int32_t) { return 1 << 8; }
static inline constexpr uint32_t FlushThreshold(uint64_t) { return 1 << 16; } // 65536
static inline constexpr uint32_t FlushThreshold(int64_t) { return 1 << 16; }
static inline constexpr uint32_t FlushThreshold(hugeint_t) { return 1 << 16; }
static inline constexpr uint32_t FlushThreshold(uhugeint_t) { return 1 << 16; }

// Unsigned version - check top B bits
static inline bool HasTopBitsSet(uint8_t value) {
	return (value >> (8 - 3)) != 0;   // top 3 bits
}
static inline bool HasTopBitsSet(uint16_t value) {
	return (value >> (16 - 5)) != 0;  // top 5 bits
}
static inline bool HasTopBitsSet(uint32_t value) {
	return (value >> (32 - 8)) != 0;  // top 8 bits
}
static inline bool HasTopBitsSet(uint64_t value) {
	return (value >> (64 - 16)) != 0; // top 16 bits
}

// Signed version - check magnitude
static inline bool HasTopBitsSet(int8_t value) {
	uint8_t abs_val = value < 0 ? static_cast<uint8_t>(-value) : static_cast<uint8_t>(value);
	return (abs_val >> (8 - 3)) != 0;
}
static inline bool HasTopBitsSet(int16_t value) {
	uint16_t abs_val = value < 0 ? static_cast<uint16_t>(-value) : static_cast<uint16_t>(value);
	return (abs_val >> (16 - 5)) != 0;
}
static inline bool HasTopBitsSet(int32_t value) {
	uint32_t abs_val = value < 0 ? static_cast<uint32_t>(-value) : static_cast<uint32_t>(value);
	return (abs_val >> (32 - 8)) != 0;
}
static inline bool HasTopBitsSet(int64_t value) {
	uint64_t abs_val = value < 0 ? static_cast<uint64_t>(-value) : static_cast<uint64_t>(value);
	return (abs_val >> (64 - 16)) != 0;
}

// hugeint_t (signed 128-bit) - check top 16 bits
static inline bool HasTopBitsSet(hugeint_t value) {
	// Top 16 bits of 128-bit means upper must have magnitude >= 2^48
	constexpr uint64_t threshold = uint64_t(1) << 48;
	if (value.upper >= 0) {
		return static_cast<uint64_t>(value.upper) >= threshold;
	} else {
		return static_cast<uint64_t>(-(value.upper + 1)) >= threshold;
	}
}

// uhugeint_t (unsigned 128-bit) - check top 16 bits
static inline bool HasTopBitsSet(uhugeint_t value) {
	constexpr uint64_t threshold = uint64_t(1) << 48;
	return value.upper >= threshold;
}

template <class INPUT_TYPE, class ACC_TYPE>
static idx_t PacSumIntegerStateSize(const AggregateFunction &) {
	return sizeof(PacSumIntegerState<INPUT_TYPE, ACC_TYPE>);
}

template <class INPUT_TYPE, class ACC_TYPE>
static void PacSumIntegerInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &state = *reinterpret_cast<PacSumIntegerState<INPUT_TYPE, ACC_TYPE> *>(state_ptr);
	state.seen_null = false;
	state.update_count = 0;
	memset(state.small_totals, 0, sizeof(state.small_totals));
	memset(state.totals, 0, sizeof(state.totals));
}

template <class INPUT_TYPE, class ACC_TYPE>
static void PacSumIntegerUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr,
                                idx_t count) {
	D_ASSERT(input_count == 2);
	auto &state = *reinterpret_cast<PacSumIntegerState<INPUT_TYPE, ACC_TYPE> *>(state_ptr);

	if (state.seen_null) {
		return; // Already NULL, skip processing
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
			return; // Stop processing after seeing NULL
		}

		uint64_t key_hash = hashes[hash_idx];
		INPUT_TYPE value = values[value_idx];

		// Check if value is large (top B bits set) - flush first to avoid overflow
		bool is_large = HasTopBitsSet(value);
		if (is_large && state.update_count > 0) {
			state.Flush();
		}

		// Add to small_totals (branchless)
		for (int j = 0; j < 64; j++) {
			uint64_t mask = BitToMask((key_hash >> j) & 1ULL);
			state.small_totals[j] += MaskValue(value, mask);
		}

		// Flush if we've accumulated (1 << B) values or if this was a large value
		if (++state.update_count >= FlushThreshold(INPUT_TYPE()) || is_large) {
			state.Flush();
		}
	}
}

template <class INPUT_TYPE, class ACC_TYPE>
static void PacSumIntegerScatterUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states,
                                       idx_t count) {
	D_ASSERT(input_count == 2);

	UnifiedVectorFormat hash_data, value_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacSumIntegerState<INPUT_TYPE, ACC_TYPE> *>(sdata);

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
		INPUT_TYPE value = values[value_idx];

		bool is_large = HasTopBitsSet(value);
		if (is_large && state->update_count > 0) {
			state->Flush();
		}

		for (int j = 0; j < 64; j++) {
			uint64_t mask = BitToMask((key_hash >> j) & 1ULL);
			state->small_totals[j] += MaskValue(value, mask);
		}

		if (++state->update_count >= FlushThreshold(INPUT_TYPE()) || is_large) {
			state->Flush();
		}
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
		// Flush both states first
		if (sdata[i]->update_count > 0) {
			sdata[i]->Flush();
		}
		if (tdata[i]->update_count > 0) {
			tdata[i]->Flush();
		}
		// Combine totals
		for (int j = 0; j < 64; j++) {
			tdata[i]->totals[j] += sdata[i]->totals[j];
		}
	}
}

template <class INPUT_TYPE, class ACC_TYPE>
static void PacSumIntegerFinalize(Vector &states, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	auto sdata = FlatVector::GetData<PacSumIntegerState<INPUT_TYPE, ACC_TYPE> *>(states);
	auto rdata = FlatVector::GetData<double>(result);
	auto &result_mask = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto state = sdata[i];

		if (state->seen_null) {
			result_mask.SetInvalid(offset + i);
			continue;
		}

		// Flush any remaining small_totals
		if (state->update_count > 0) {
			state->Flush();
		}

		double total = 0.0;
		for (int j = 0; j < 64; j++) {
			total += ToDouble(state->totals[j]);
		}
		double mean = total / 64.0;

		double variance = 0.0;
		for (int j = 0; j < 64; j++) {
			double diff = ToDouble(state->totals[j]) - mean;
			variance += diff * diff;
		}
		variance /= 64.0;

		rdata[offset + i] = variance;
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
// Explicit instantiations for unsigned integer types (accumulate to uhugeint_t)
// =========================
static void PacSumUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumIntegerUpdate<uint8_t, uhugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                  idx_t count) {
	PacSumIntegerUpdate<uint16_t, uhugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumIntegerUpdate<uint32_t, uhugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                idx_t count) {
	PacSumIntegerUpdate<uint64_t, uhugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumIntegerUpdate<uhugeint_t, uhugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumScatterUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumIntegerScatterUpdate<uint8_t, uhugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                   idx_t count) {
	PacSumIntegerScatterUpdate<uint16_t, uhugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUInteger(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumIntegerScatterUpdate<uint32_t, uhugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                 idx_t count) {
	PacSumIntegerScatterUpdate<uint64_t, uhugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumIntegerScatterUpdate<uhugeint_t, uhugeint_t>(inputs, aggr, input_count, states, count);
}

// =========================
// Combine and Finalize wrappers
// =========================
static void PacSumCombineFloat(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumFloatCombine<float>(source, target, aggr, count);
}
static void PacSumCombineDouble(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumFloatCombine<double>(source, target, aggr, count);
}
// Type-specific combine functions (must match state type exactly because Flush touches small_totals)
static void PacSumCombineTinyInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumIntegerCombine<int8_t, hugeint_t>(source, target, aggr, count);
}
static void PacSumCombineSmallInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumIntegerCombine<int16_t, hugeint_t>(source, target, aggr, count);
}
static void PacSumCombineInteger(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumIntegerCombine<int32_t, hugeint_t>(source, target, aggr, count);
}
static void PacSumCombineBigInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumIntegerCombine<int64_t, hugeint_t>(source, target, aggr, count);
}
static void PacSumCombineHugeInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumIntegerCombine<hugeint_t, hugeint_t>(source, target, aggr, count);
}
static void PacSumCombineUTinyInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumIntegerCombine<uint8_t, uhugeint_t>(source, target, aggr, count);
}
static void PacSumCombineUSmallInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumIntegerCombine<uint16_t, uhugeint_t>(source, target, aggr, count);
}
static void PacSumCombineUInteger(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumIntegerCombine<uint32_t, uhugeint_t>(source, target, aggr, count);
}
static void PacSumCombineUBigInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumIntegerCombine<uint64_t, uhugeint_t>(source, target, aggr, count);
}
static void PacSumCombineUHugeInt(Vector &source, Vector &target, AggregateInputData &aggr, idx_t count) {
	PacSumIntegerCombine<uhugeint_t, uhugeint_t>(source, target, aggr, count);
}

// Type-specific finalize functions
static void PacSumFinalizeFloat(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumFloatFinalize<float>(states, aggr, result, count, offset);
}
static void PacSumFinalizeDouble(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumFloatFinalize<double>(states, aggr, result, count, offset);
}
static void PacSumFinalizeTinyInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumIntegerFinalize<int8_t, hugeint_t>(states, aggr, result, count, offset);
}
static void PacSumFinalizeSmallInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumIntegerFinalize<int16_t, hugeint_t>(states, aggr, result, count, offset);
}
static void PacSumFinalizeInteger(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumIntegerFinalize<int32_t, hugeint_t>(states, aggr, result, count, offset);
}
static void PacSumFinalizeBigInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumIntegerFinalize<int64_t, hugeint_t>(states, aggr, result, count, offset);
}
static void PacSumFinalizeHugeInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumIntegerFinalize<hugeint_t, hugeint_t>(states, aggr, result, count, offset);
}
static void PacSumFinalizeUTinyInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumIntegerFinalize<uint8_t, uhugeint_t>(states, aggr, result, count, offset);
}
static void PacSumFinalizeUSmallInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count,
                                    idx_t offset) {
	PacSumIntegerFinalize<uint16_t, uhugeint_t>(states, aggr, result, count, offset);
}
static void PacSumFinalizeUInteger(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumIntegerFinalize<uint32_t, uhugeint_t>(states, aggr, result, count, offset);
}
static void PacSumFinalizeUBigInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumIntegerFinalize<uint64_t, uhugeint_t>(states, aggr, result, count, offset);
}
static void PacSumFinalizeUHugeInt(Vector &states, AggregateInputData &aggr, Vector &result, idx_t count, idx_t offset) {
	PacSumIntegerFinalize<uhugeint_t, uhugeint_t>(states, aggr, result, count, offset);
}


struct PacAggregateLocalState : public FunctionLocalState {
	explicit PacAggregateLocalState(uint64_t seed) : gen(seed) {}
	std::mt19937_64 gen;
};

// Compute second-moment variance (not unbiased estimator)
static double ComputeSecondMomentVariance(const std::vector<double> &values) {
	idx_t n = values.size();
	if (n <= 1) {
		return 0.0;
	}

	double mean = 0.0;
	for (auto v : values) mean += v;
	mean /= double(n);

	double var = 0.0;
	for (auto v : values) {
		double d = v - mean;
		var += d * d;
	}
	return var / double(n);
}

DUCKDB_API unique_ptr<FunctionLocalState>
PacAggregateInit(ExpressionState &state, const BoundFunctionExpression &, FunctionData *) {
	uint64_t seed = std::random_device{}();
	Value pac_seed;
	if (state.GetContext().TryGetCurrentSetting("pac_seed", pac_seed) && !pac_seed.IsNull()) {
		seed = uint64_t(pac_seed.GetValue<int64_t>());
	}
	return make_uniq<PacAggregateLocalState>(seed);
}

DUCKDB_API void PacAggregateScalar(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &vals = args.data[0];
	auto &cnts = args.data[1];
	auto &mi_vec = args.data[2];
	auto &k_vec  = args.data[3];

	idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto res = FlatVector::GetData<double>(result);
	FlatVector::Validity(result).SetAllValid(count);

	auto &local = ExecuteFunctionState::GetFunctionState(state)->Cast<PacAggregateLocalState>();
	auto &gen = local.gen;

	for (idx_t row = 0; row < count; row++) {
		bool refuse = false;

		// --- read mi, k ---
		double mi = mi_vec.GetValue(row).GetValue<double>();
		if (mi <= 0.0) {
			throw InvalidInputException("pac_aggregate: mi must be > 0");
		}
		int k = FlatVector::GetData<int32_t>(k_vec)[row];

		// --- extract lists ---
		UnifiedVectorFormat vvals, vcnts;
		vals.ToUnifiedFormat(count, vvals);
		cnts.ToUnifiedFormat(count, vcnts);

		idx_t r = vvals.sel ? vvals.sel->get_index(row) : row;
		if (!vvals.validity.RowIsValid(r) || !vcnts.validity.RowIsValid(r)) {
			result.SetValue(row, Value());
			continue;
		}

		auto *vals_entries = UnifiedVectorFormat::GetData<list_entry_t>(vvals);
		auto *cnts_entries = UnifiedVectorFormat::GetData<list_entry_t>(vcnts);

		auto ve = vals_entries[r];
		auto ce = cnts_entries[r];

		// Values and counts arrays must have the same length (one count per sample position).
		if (ve.length != ce.length) {
			throw InvalidInputException("pac_aggregate: values and counts length mismatch");
		}
		idx_t vals_len = ve.length;
		idx_t cnts_len = ce.length;

		// Read configured m from session settings (default 128)
		int64_t m_cfg = 128;
		Value m_val;
		if (state.GetContext().TryGetCurrentSetting("pac_m", m_val) && !m_val.IsNull()) {
			m_cfg = m_val.GetValue<int64_t>();
			if (m_cfg <= 0) {
				m_cfg = 128;
			}
		}

		// Read enforce_m_values flag (default true)
		bool enforce_m_values = true;
		Value enforce_val;
		if (state.GetContext().TryGetCurrentSetting("enforce_m_values", enforce_val) && !enforce_val.IsNull()) {
			enforce_m_values = enforce_val.GetValue<bool>();
		}

		// Enforce per-sample array length equals configured m (only if enabled)
		if (enforce_m_values && (int64_t)vals_len != m_cfg) {
			throw InvalidInputException(StringUtil::Format("pac_aggregate: expected per-sample array length %lld but got %llu", (long long)m_cfg, (unsigned long long)vals_len));
		}

		auto &vals_child = ListVector::GetEntry(vals);
		auto &cnts_child = ListVector::GetEntry(cnts);
		vals_child.Flatten(ve.offset + ve.length);
		cnts_child.Flatten(ce.offset + ce.length);

		auto *vdata = FlatVector::GetData<double>(vals_child);
		auto *cdata = FlatVector::GetData<int32_t>(cnts_child);

		std::vector<double> values;
		values.reserve(vals_len);

		int32_t max_count = 0;
		for (idx_t i = 0; i < vals_len; i++) {
			if (!FlatVector::Validity(vals_child).RowIsValid(ve.offset + i)) {
				refuse = true;
				break;
			}
			values.push_back(vdata[ve.offset + i]);
			int32_t cnt_val = 0;
			if (i < cnts_len) {
				cnt_val = cdata[ce.offset + i];
			}
			max_count = std::max(max_count, cnt_val);
		}

		if (refuse || values.empty() || max_count < k) {
			result.SetValue(row, Value());
			continue;
		}

		// ---------------- PAC core ----------------

		// 1. pick J
		std::uniform_int_distribution<idx_t> unif(0, values.size() - 1);
		idx_t J = unif(gen);
		double yJ = values[J];

		// 2. leave-one-out variance
		std::vector<double> loo;
		loo.reserve(values.size() - 1);
		for (idx_t i = 0; i < values.size(); i++) {
			if (i != J) {
				loo.push_back(values[i]);
			}
		}

		double sigma2 = ComputeSecondMomentVariance(loo);
		double delta  = sigma2 / (2.0 * mi);

		if (delta <= 0.0 || !std::isfinite(delta)) {
			res[row] = yJ;
			continue;
		}

		// 3. noise
		std::normal_distribution<double> gauss(0.0, std::sqrt(delta));
		res[row] = yJ + gauss(gen);
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void RegisterPacAggregateFunctions(ExtensionLoader &loader) {
	auto fun = ScalarFunction(
		"pac_aggregate",
		{LogicalType::LIST(LogicalType::DOUBLE),
		 LogicalType::LIST(LogicalType::INTEGER),
		 LogicalType::DOUBLE,
		 LogicalType::INTEGER},
		LogicalType::DOUBLE,
		PacAggregateScalar,
		nullptr, nullptr, nullptr,
		PacAggregateInit);

	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	loader.RegisterFunction(fun);

	// Register pac_count aggregate function
	// Input: UBIGINT key_hash
	// Output: DOUBLE (variance of the 64 bit counters)
	// Uses SIMD-friendly update with intermediate accumulators
	AggregateFunction pac_count("pac_count", {LogicalType::UBIGINT}, LogicalType::DOUBLE, PacCountStateSize,
	                            PacCountInitialize, PacCountScatterUpdate, PacCountCombine, PacCountFinalize,
	                            FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate);
	loader.RegisterFunction(pac_count);

	// Register pac_sum aggregate function set
	// Input: (UBIGINT key_hash, <numeric> value)
	// Output: DOUBLE (variance of the 64 sums)
	// Supports all numeric types that SUM supports
	AggregateFunctionSet pac_sum_set("pac_sum");

	// Signed integers (use PacSumIntegerState<INPUT_TYPE, hugeint_t>)
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::TINYINT}, LogicalType::DOUBLE,
	    PacSumIntegerStateSize<int8_t, hugeint_t>, PacSumIntegerInitialize<int8_t, hugeint_t>, PacSumScatterTinyInt,
	    PacSumCombineTinyInt, PacSumFinalizeTinyInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateTinyInt));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::SMALLINT}, LogicalType::DOUBLE,
	    PacSumIntegerStateSize<int16_t, hugeint_t>, PacSumIntegerInitialize<int16_t, hugeint_t>, PacSumScatterSmallInt,
	    PacSumCombineSmallInt, PacSumFinalizeSmallInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateSmallInt));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::INTEGER}, LogicalType::DOUBLE,
	    PacSumIntegerStateSize<int32_t, hugeint_t>, PacSumIntegerInitialize<int32_t, hugeint_t>, PacSumScatterInteger,
	    PacSumCombineInteger, PacSumFinalizeInteger, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateInteger));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::BIGINT}, LogicalType::DOUBLE,
	    PacSumIntegerStateSize<int64_t, hugeint_t>, PacSumIntegerInitialize<int64_t, hugeint_t>, PacSumScatterBigInt,
	    PacSumCombineBigInt, PacSumFinalizeBigInt, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacSumUpdateBigInt));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::HUGEINT}, LogicalType::DOUBLE,
	    PacSumIntegerStateSize<hugeint_t, hugeint_t>, PacSumIntegerInitialize<hugeint_t, hugeint_t>,
	    PacSumScatterHugeInt, PacSumCombineHugeInt, PacSumFinalizeHugeInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateHugeInt));

	// Unsigned integers (use PacSumIntegerState<INPUT_TYPE, uhugeint_t>)
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UTINYINT}, LogicalType::DOUBLE,
	    PacSumIntegerStateSize<uint8_t, uhugeint_t>, PacSumIntegerInitialize<uint8_t, uhugeint_t>,
	    PacSumScatterUTinyInt, PacSumCombineUTinyInt, PacSumFinalizeUTinyInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUTinyInt));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::USMALLINT}, LogicalType::DOUBLE,
	    PacSumIntegerStateSize<uint16_t, uhugeint_t>, PacSumIntegerInitialize<uint16_t, uhugeint_t>,
	    PacSumScatterUSmallInt, PacSumCombineUSmallInt, PacSumFinalizeUSmallInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUSmallInt));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UINTEGER}, LogicalType::DOUBLE,
	    PacSumIntegerStateSize<uint32_t, uhugeint_t>, PacSumIntegerInitialize<uint32_t, uhugeint_t>,
	    PacSumScatterUInteger, PacSumCombineUInteger, PacSumFinalizeUInteger,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUInteger));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UBIGINT}, LogicalType::DOUBLE,
	    PacSumIntegerStateSize<uint64_t, uhugeint_t>, PacSumIntegerInitialize<uint64_t, uhugeint_t>,
	    PacSumScatterUBigInt, PacSumCombineUBigInt, PacSumFinalizeUBigInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUBigInt));
	pac_sum_set.AddFunction(AggregateFunction(
	    "pac_sum", {LogicalType::UBIGINT, LogicalType::UHUGEINT}, LogicalType::DOUBLE,
	    PacSumIntegerStateSize<uhugeint_t, uhugeint_t>, PacSumIntegerInitialize<uhugeint_t, uhugeint_t>,
	    PacSumScatterUHugeInt, PacSumCombineUHugeInt, PacSumFinalizeUHugeInt,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateUHugeInt));

	// Floating point (use PacSumFloatState<FLOAT_TYPE>)
	pac_sum_set.AddFunction(
	    AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::FLOAT}, LogicalType::DOUBLE,
	                      PacSumFloatStateSize<float>, PacSumFloatInitialize<float>, PacSumScatterFloat,
	                      PacSumCombineFloat, PacSumFinalizeFloat, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                      PacSumUpdateFloat));
	pac_sum_set.AddFunction(
	    AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      PacSumFloatStateSize<double>, PacSumFloatInitialize<double>, PacSumScatterDouble,
	                      PacSumCombineDouble, PacSumFinalizeDouble, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                      PacSumUpdateDouble));

	loader.RegisterFunction(pac_sum_set);
}

} // namespace duckdb
