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
// We want to do this in a SIMD-friendly way. Therefore, we want to create 64 subtotal of uint8_t (i.e. bytes),
// and perform 64 byte-additions, because in the widest SIMD implementation, AVX512, this means that this
// could be done in a *SINGLE* instruction (AVX512 has 64 lanes of uint8, as 64x8=512)
//
// But, to help auto-vectorizing, we use uint64_t probabilistic_total[8], rather than uint8_t probabilistic_total[64]
// because key_hash is also uint64_t. We apply the below mask to key_hash to extract the lowest bit of each byte:

#define PAC_COUNT_MASK                                                                                                 \
	(1ULL | (1ULL << 8) | (1ULL << 16) | (1ULL << 24) | (1ULL << 32) | (1ULL << 40) | (1ULL << 48) | (1ULL << 56))

// For each of the 8 iterations i, we then do (hash_key>>i) & PAC_COUNT_MASK which selects 8 bits, and then add these
// with a single uint64_t ADD to a uint64 subtotal[].
//
// This technique is known as SWAR: SIMD Within A Register
//
// You can only add 255 times before the bytes in this uint64_t start touching each other (causing overflow).
// So after 255 iterations, the probabilistic_total8[64] are added to uint16_t probabilistic_total16 and reset to 0.
// This repeats possibly in wider total types 16/32/64. We do the extra cascades to 16/32 because often these thinner
// counters are enough to hold the data and that saves memory (we allocate the counter arrays on need only).
//
// The idea is that we get very fast performance 255 times and slower performance once every 256 only.
// This SIMD-friendly implementation can make PAC counting almost as fast as normal counting.
//
// MEMORY OPTIMIZATION: We use lazy allocation with cascading levels (8->16->32->64 bits) to reduce
// Define PAC_COUNT_NONCASCADING to use simple fixed uint64_t[64] counters instead.
//
// If we enlarge to uint16_t and allocate uint16_t probabilistic_total16[64] for that, and later enlarge to uint32_t
// and allocate uint32_t probabilistic_total32[64], we are wasting/leaking the previously used uint8 and uint16 memory.
//
// To avoid that, the uint8 memory is now one "bank" (64 bytes). If int16 is needed, a second bank gets allocated
// of again 64 bytes (128 bytes total). And when uint32_t is needed, we allocate two more banks and for 64bits 4 more.
// The count kernels now iterates over banks. Using banked allocations, no memory gets wasted.
//
// Bank layout for total:
//   banks1: 2 banks (128 bytes) - fits uint16_t[64]
//   banks2: 2 banks (128 bytes) - with banks1 = uint32_t[64]
//   banks3: 4 banks (256 bytes) - with banks1+2 = uint64_t[64]

//#define PAC_COUNT_NONLAZY 1  // Pre-allocate all levels at initialization (set via CXXFLAGS)
//#define PAC_COUNT_NONBANKED 1 // will directly aggregate in uint64_t probabilistic_total[64] (no cascading/banking)
//#define PAC_COUNT_NOBUFFERING 1 // Disable input buffering (allocate immediately)

// Two levels: subtotal8 (banks0, uint8_t[64] packed as uint64_t[8]) flushes to banked total (uint16/32/64 depending
// on count magnitude).

struct PacCountState {
#ifdef PAC_COUNT_NONBANKED
	uint64_t probabilistic_total[64];
	void Flush(ArenaAllocator &) {
	}
	void GetTotalsAsDouble(double *dst) const {
		ToDoubleArray(probabilistic_total, dst);
	}
	bool IsBuffering() const {
		return false;
	}
#else
	// Input buffering: buffer first 4 hash values before allocating banks0
	static constexpr uint8_t BUFFER_CAPACITY = 4;

	uint8_t subtotal8_count; // counts in subtotal8 before flush (max 255)
	uint8_t total_level;     // 0=none, 16, 32, or 64
	bool buffering;          // true = buffering mode, false = aggregation mode

	// Union: buffering mode uses buf_hashes, aggregation mode uses banks0
	union {
		uint64_t buf_hashes[BUFFER_CAPACITY]; // Buffering mode: store hash values
		struct {                              // Aggregation mode: bank pointers
			uint8_t *banks0;                  // 1 bank (64 bytes) for subtotal8
			uint8_t *banks1;                  // 2 banks (128 bytes) for total16
			uint8_t *banks2;                  // 2 banks (128 bytes) for total32
			uint8_t *banks3;                  // 4 banks (256 bytes) for total64
		};
	};

	bool IsBuffering() const {
#ifdef PAC_COUNT_NOBUFFERING
		return false;
#else
		return buffering;
#endif
	}

	uint8_t GetBufferCount() const {
		return subtotal8_count; // reuse subtotal8_count for buffer count in buffering mode
	}

	void SetBufferCount(uint8_t count) {
		subtotal8_count = count;
	}

	// Get bank pointer (0-7): bank 0 is banks0, banks 1-2 are banks1, etc.
	inline uint8_t *GetBank(int n) const {
		if (n == 0)
			return banks0;
		if (n < 3)
			return banks1 + (n - 1) * 64;
		if (n < 5)
			return banks2 + (n - 3) * 64;
		return banks3 + (n - 5) * 64;
	}

	// Get subtotal8 as uint64_t array (for SWAR operations)
	inline uint64_t *GetSubtotal8() const {
		return reinterpret_cast<uint64_t *>(banks0);
	}

	// Number of banks for total level: 16->2, 32->4, 64->8 (starting from bank 1)
	static constexpr int BanksForLevel(int level) {
		return level / 8;
	}

	// Allocate banks0 (subtotal8) and transition to aggregation mode
	void AllocateFirstLevel(ArenaAllocator &allocator) {
		// Zero out union first (while still in buffering mode conceptually)
		banks0 = nullptr;
		banks1 = nullptr;
		banks2 = nullptr;
		banks3 = nullptr;
		// Now transition to aggregation mode
		buffering = false;
		banks0 = reinterpret_cast<uint8_t *>(allocator.AllocateAligned(64));
		memset(banks0, 0, 64);
		subtotal8_count = 0;
		total_level = 0;
	}

	// Allocate only banks0 without resetting other banks (for Finalize after Combine)
	void EnsureBanks0Allocated(ArenaAllocator &allocator) {
		// Only call this when NOT buffering
		if (!buffering && !banks0) {
			banks0 = reinterpret_cast<uint8_t *>(allocator.AllocateAligned(64));
			memset(banks0, 0, 64);
		}
	}

	// Allocate banks for total level (only allocates what's new)
	void AllocateBanks(ArenaAllocator &allocator, int level) {
		if (!banks1) {
			banks1 = reinterpret_cast<uint8_t *>(allocator.Allocate(128));
			memset(banks1, 0, 128);
		}
		if (level >= 32 && !banks2) {
			banks2 = reinterpret_cast<uint8_t *>(allocator.Allocate(128));
			memset(banks2, 0, 128);
		}
		if (level >= 64 && !banks3) {
			banks3 = reinterpret_cast<uint8_t *>(allocator.Allocate(256));
			memset(banks3, 0, 256);
		}
	}

	// Upgrade total to wider type, preserving values
	// Total banks start at bank 1 (banks1), not bank 0 (banks0 is subtotal8)
	template <typename SRC_T, typename DST_T>
	void UpgradeTotal(ArenaAllocator &allocator, int new_level) {
		constexpr int SRC_PER_BANK = 64 / sizeof(SRC_T);
		constexpr int DST_PER_BANK = 64 / sizeof(DST_T);
		int old_banks = BanksForLevel(total_level);

		// Gather old values from total banks (starting at bank 1)
		SRC_T old_values[64];
		for (int b = 0; b < old_banks; b++) {
			auto *src = reinterpret_cast<SRC_T *>(GetBank(b + 1)); // +1 to skip banks0
			for (int i = 0; i < SRC_PER_BANK; i++)
				old_values[b * SRC_PER_BANK + i] = src[i];
		}

		// Allocate new banks and scatter
		AllocateBanks(allocator, new_level);
		int new_banks = BanksForLevel(new_level);
		for (int b = 0; b < new_banks; b++) {
			auto *dst = reinterpret_cast<DST_T *>(GetBank(b + 1)); // +1 to skip banks0
			for (int i = 0; i < DST_PER_BANK; i++) {
				int idx = b * DST_PER_BANK + i;
				dst[i] = (idx < 64) ? static_cast<DST_T>(old_values[idx]) : 0;
			}
		}
		total_level = new_level;
	}

	// Add subtotal8 (banks0) to total (banks1+), upgrading if needed
	template <typename T>
	AUTOVECTORIZE void AddSubtotalToTotal() {
		constexpr int ELEMS_PER_BANK = 64 / sizeof(T);
		int num_banks = BanksForLevel(total_level);
		auto *sub = reinterpret_cast<const uint8_t *>(banks0);

		for (int b = 0; b < num_banks; b++) {
			auto *dst = reinterpret_cast<T *>(GetBank(b + 1)); // +1 to skip banks0
			for (int i = 0; i < ELEMS_PER_BANK; i++) {
				dst[i] += static_cast<T>(sub[b * ELEMS_PER_BANK + i]);
			}
		}
	}

	// Flush subtotal8 (banks0) to total (banks1+)
	void Flush(ArenaAllocator &allocator) {
		if (IsBuffering() || subtotal8_count == 0)
			return;

		// Determine required level for new total
		uint64_t new_total =
		    (total_level == 0) ? subtotal8_count : static_cast<uint64_t>(subtotal8_count) + GetMaxPossibleCount();
		int required = (new_total <= UINT16_MAX) ? 16 : (new_total <= UINT32_MAX) ? 32 : 64;

		// Initialize or upgrade if needed
		if (total_level == 0) {
			AllocateBanks(allocator, required);
			total_level = required;
		} else if (required > total_level) {
			if (total_level == 16)
				UpgradeTotal<uint16_t, uint32_t>(allocator, 32);
			if (required == 64 && total_level == 32)
				UpgradeTotal<uint32_t, uint64_t>(allocator, 64);
		}

		// Add subtotal8 to total
		if (total_level == 16)
			AddSubtotalToTotal<uint16_t>();
		else if (total_level == 32)
			AddSubtotalToTotal<uint32_t>();
		else
			AddSubtotalToTotal<uint64_t>();

		memset(banks0, 0, 64);
		subtotal8_count = 0;
	}

	// Estimate max count in any counter (conservative: assumes all bits set)
	uint64_t GetMaxPossibleCount() const {
		if (total_level == 64) {
			uint64_t m = 0;
			for (int i = 0; i < 64; i++)
				m = std::max(m, GetTotalElement<uint64_t>(i));
			return m;
		}
		if (total_level == 32) {
			uint32_t m = 0;
			for (int i = 0; i < 64; i++)
				m = std::max(m, GetTotalElement<uint32_t>(i));
			return m;
		}
		if (total_level == 16) {
			uint16_t m = 0;
			for (int i = 0; i < 64; i++)
				m = std::max(m, GetTotalElement<uint16_t>(i));
			return m;
		}
		return 0;
	}

	// Get element from banked total (total banks start at bank 1)
	template <typename T>
	T GetTotalElement(int i) const {
		constexpr int PER_BANK = 64 / sizeof(T);
		int bank = i / PER_BANK, off = i % PER_BANK;
		return reinterpret_cast<const T *>(GetBank(bank + 1))[off]; // +1 to skip banks0
	}

	void GetTotalsAsDouble(double *dst) const {
		// First get subtotal8 values from banks0
		auto *sub = reinterpret_cast<const uint8_t *>(banks0);
		for (int i = 0; i < 64; i++)
			dst[i] = static_cast<double>(sub[i]);

		// Add total if present
		if (total_level == 16) {
			for (int i = 0; i < 64; i++)
				dst[i] += GetTotalElement<uint16_t>(i);
		} else if (total_level == 32) {
			for (int i = 0; i < 64; i++)
				dst[i] += GetTotalElement<uint32_t>(i);
		} else if (total_level == 64) {
			for (int i = 0; i < 64; i++)
				dst[i] += GetTotalElement<uint64_t>(i);
		}
	}
#endif
};

} // namespace duckdb

#endif // PAC_COUNT_HPP
