//
// Created by ila on 12/19/25.
//

#ifndef PAC_COUNT_HPP
#define PAC_COUNT_HPP

#include "duckdb.hpp"
#include "pac_aggregate.hpp"

namespace duckdb {

void RegisterPacCountFunctions(ExtensionLoader &);

// PAC_COUNT(key_hash) implements a COUNT aggregate that for each privacy-unit (identified by a key_hash)
// computes 64 independent counts, where each independent count randomly (50% chance) includes a PU or not.
// The observation is that the 64-bits of a hashed key of the PU are random (50% 0, 50% 1), so we can take
// the 64 bits of the key to make 64 independent decisions.
//
// A COUNT() aggregate in its implementation simply performs total += 1
//
// PAC_COUNT() needs to do for(i=0; i<64; i++) total[i] += (key_hash >> i) & 1; (extract bit i)
//
// We want to do this in a SIMD-friendly way. Therefore, we want to create 64 subtotals of uint8_t (i.e. bytes),
// and perform 64 byte-additions, because in the widest SIMD implementation, AVX512, this means that this
// could be done in a *SINGLE* instruction (AVX512 has 64 lanes of uint8, as 64x8=512)
//
// But, to help auto-vectorizing, we use uint64_t probabilistic_total[8], rather than uint8_t probabilistic_total[64]
// because key_hash is also uint64_t. We apply the below mask to key_hash to extract the lowest bit of each byte:

#define PAC_COUNT_MASK                                                                                                 \
	(1ULL | (1ULL << 8) | (1ULL << 16) | (1ULL << 24) | (1ULL << 32) | (1ULL << 40) | (1ULL << 48) | (1ULL << 56))

// For each of the 8 iterations i, we then do (hash_key>>i) & PAC_COUNT_MASK which selects 8 bits, and then add these
// with a single uint64_t ADD to a uint64 subtotal.
//
// This technique is known as SWAR: SIMD Within A Register
//
// You can only add 255 times before the bytes in this uint64_t start touching each other (causing overflow).
// So after 255 iterations, the probabilistic_total8[64] are added to uint16_t probabilistic_total[64] and reset to 0.
// This repeats possibly in wider total types 16/32/64. We do the extra cascades to 16/32 because often these thinner
// counters are enough to hold the data and that saves memory (we allocate the counter arrays on need only).
//
// The idea is that we get very fast performance 255 times and slower performance once every 256 only.
// This SIMD-friendly implementation can make PAC counting almost as fast as normal counting.
//
// MEMORY OPTIMIZATION: We use lazy allocation with cascading levels (8->16->32->64 bits) to reduce
// memory footprint. For small counts (< 255), only 64 bytes are needed instead of 576+ bytes.
// Define PAC_COUNT_NONCASCADING to use simple fixed uint64_t[64] counters instead.

//#define PAC_COUNT_NONCASCADING 1
//#define PAC_COUNT_NONLAZY 1  // Pre-allocate all levels at initialization

// State for pac_count: lazily allocated cascading counters (or fixed array if NONCASCADING)
// Uses ArenaAllocator for memory management - arena handles cleanup when aggregate completes.
struct PacCountState {
#ifdef PAC_COUNT_NONCASCADING
	uint64_t probabilistic_total[64]; // Simple fixed array of 64 counters

	// NONCASCADING: dummy methods for uniform interface
	void Flush() {
	} // no-op
	void GetTotalsAsDouble(double *dst) const {
		ToDoubleArray(probabilistic_total, dst);
	}
#else
	ArenaAllocator *allocator; // Pointer to DuckDB's arena allocator (set during first update)

	// Lazily allocated levels (nullptr if not allocated)
	uint64_t *probabilistic_total8;  // 8 x uint64_t (64 bytes) - SWAR packed uint8_t subtotals
	uint64_t *probabilistic_total16; // 16 x uint64_t (128 bytes) - SWAR packed uint16_t subtotals
	uint64_t *probabilistic_total32; // 32 x uint64_t (256 bytes) - SWAR packed uint32_t subtotals
	uint64_t *probabilistic_total64; // 64 x uint64_t (512 bytes) - full uint64_t total

	// Exact (non-probabilistic) total for each level - kept to determine when a level's could overflow
	uint64_t exact_total8;  // max 255 before flush to level 16
	uint64_t exact_total16; // max 65535 before flush to level 32
	uint64_t exact_total32; // max ~4G before flush to level 64
	uint64_t exact_total64; // final level (no overflow possible within uint64_t range)

	// Lazily allocate a level's buffer if not yet allocated
	// Returns 0 if newly allocated (and zeroed), otherwise returns exact_total unchanged
	template <typename EXACT_T = int>
	inline EXACT_T EnsureLevelAllocated(uint64_t *&buffer, idx_t count, EXACT_T exact_total = 0) {
		if (!buffer) {
			buffer = reinterpret_cast<uint64_t *>(allocator->Allocate(count * sizeof(uint64_t)));
			memset(buffer, 0, count * sizeof(uint64_t));
			return 0;
		}
		return exact_total;
	}

	// Cascade SWAR-packed counters from one level to the next with proper bit reordering
	// SRC_T/DST_T: element types (uint8_t->uint16_t, uint16_t->uint32_t, uint32_t->uint64_t)
	// SRC_SWAR/DST_SWAR: SWAR widths (8/16/32/64)
	template <typename SRC_T, typename DST_T, int SRC_SWAR, int DST_SWAR>
	static inline void CascadeToNextLevel(const uint64_t *src_buf, uint64_t *dst_buf) {
		const SRC_T *src = reinterpret_cast<const SRC_T *>(src_buf);
		DST_T *dst = reinterpret_cast<DST_T *>(dst_buf);
		constexpr int SRC_PER_U64 = 64 / SRC_SWAR;
		constexpr int DST_PER_U64 = 64 / DST_SWAR;
		for (int bit = 0; bit < 64; bit++) {
			int src_idx = (bit % SRC_SWAR) * SRC_PER_U64 + (bit / SRC_SWAR);
			int dst_idx = (bit % DST_SWAR) * DST_PER_U64 + (bit / DST_SWAR);
			dst[dst_idx] += src[src_idx];
		}
	}

	// Flush32: cascade probabilistic_total32 to probabilistic_total64
	AUTOVECTORIZE inline void Flush32(uint64_t value, bool force) {
		D_ASSERT(probabilistic_total32);
		uint64_t new_total = value + exact_total32;
		bool would_overflow = (new_total > UINT32_MAX);
		if (would_overflow || (force && probabilistic_total64)) {
			if (would_overflow) {
				EnsureLevelAllocated(probabilistic_total64, 64);
			}
			CascadeToNextLevel<uint32_t, uint64_t, 32, 64>(probabilistic_total32, probabilistic_total64);
			memset(probabilistic_total32, 0, 32 * sizeof(uint64_t));
			exact_total32 = value;
		} else {
			exact_total32 = static_cast<uint32_t>(new_total);
		}
	}

	// Flush16: cascade probabilistic_total16 to probabilistic_total32
	AUTOVECTORIZE inline void Flush16(uint32_t value, bool force) {
		D_ASSERT(probabilistic_total16);
		uint32_t new_total = value + exact_total16;
		bool would_overflow = (new_total > UINT16_MAX);
		if (would_overflow || (force && probabilistic_total32)) {
			if (would_overflow) {
				exact_total32 = EnsureLevelAllocated(probabilistic_total32, 32, exact_total32);
			}
			Flush32(exact_total16, force);
			CascadeToNextLevel<uint16_t, uint32_t, 16, 32>(probabilistic_total16, probabilistic_total32);
			memset(probabilistic_total16, 0, 16 * sizeof(uint64_t));
			exact_total16 = value;
		} else {
			exact_total16 = static_cast<uint16_t>(new_total);
		}
	}

	// Flush8: cascade probabilistic_total8 to probabilistic_total16
	AUTOVECTORIZE inline void Flush8(uint32_t value, bool force) {
		D_ASSERT(probabilistic_total8);
		uint32_t new_total = value + exact_total8;
		bool would_overflow = (new_total > UINT8_MAX);
		if (would_overflow || (force && probabilistic_total16)) {
			if (would_overflow) {
				exact_total16 = EnsureLevelAllocated(probabilistic_total16, 16, exact_total16);
			}
			Flush16(exact_total8, force);
			CascadeToNextLevel<uint8_t, uint16_t, 8, 16>(probabilistic_total8, probabilistic_total16);
			memset(probabilistic_total8, 0, 8 * sizeof(uint64_t));
			exact_total8 = value;
		} else {
			exact_total8 = static_cast<uint8_t>(new_total);
		}
	}

	// Flush all levels (called before Combine or Finalize)
	void Flush() {
		if (probabilistic_total8) {
			Flush8(0, true);
		}
	}

	// Convert SWAR packed counters to double[64] for finalization
	template <typename SRC_T>
	static void UnpackSWARToDouble(const uint64_t *swar_data, int swar_size, double *dst) {
		const SRC_T *src = reinterpret_cast<const SRC_T *>(swar_data);
		constexpr int elements_per_u64 = sizeof(uint64_t) / sizeof(SRC_T);
		for (int bit = 0; bit < 64; bit++) {
			int src_idx = (bit % swar_size) * elements_per_u64 + (bit / swar_size);
			dst[bit] = static_cast<double>(src[src_idx]);
		}
	}

	// Get the probabilistic total as doubles, reading from the highest allocated level
	void GetTotalsAsDouble(double *dst) const {
		if (probabilistic_total64) {
			ToDoubleArray(probabilistic_total64, dst);
		} else if (probabilistic_total32) {
			UnpackSWARToDouble<uint32_t>(probabilistic_total32, 32, dst);
		} else if (probabilistic_total16) {
			UnpackSWARToDouble<uint16_t>(probabilistic_total16, 16, dst);
		} else if (probabilistic_total8) {
			UnpackSWARToDouble<uint8_t>(probabilistic_total8, 8, dst);
		} else {
			// No data allocated - return zeros
			memset(dst, 0, 64 * sizeof(double));
		}
	}

	// Pre-allocate all levels (for NONLAZY mode)
	void InitializeAllLevels(ArenaAllocator &alloc) {
		allocator = &alloc;
		EnsureLevelAllocated(probabilistic_total8, 8);
		EnsureLevelAllocated(probabilistic_total16, 16);
		EnsureLevelAllocated(probabilistic_total32, 32);
		EnsureLevelAllocated(probabilistic_total64, 64);
	}
#endif
};

} // namespace duckdb

#endif // PAC_COUNT_HPP
