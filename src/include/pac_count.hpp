//
// Created by ila on 12/19/25.
//

#ifndef PAC_COUNT_HPP
#define PAC_COUNT_HPP

#include "duckdb.hpp"
#include "pac_aggregate.hpp"

namespace duckdb {

void RegisterPacCountFunctions(ExtensionLoader &);

// PAC_COUNT uses SWAR (SIMD Within A Register) for efficient 64-counter updates.
// Two levels: subtotal8 (inline, uint8_t[64] packed as uint64_t[8]) flushes to
// banked total (uint16/32/64 depending on count magnitude).
//
// Bank layout for total:
//   banks1: 2 banks (128 bytes) - fits uint16_t[64]
//   banks2: 2 banks (128 bytes) - with banks1 = uint32_t[64]
//   banks3: 4 banks (256 bytes) - with banks1+2 = uint64_t[64]

#define PAC_COUNT_MASK                                                                                                 \
	(1ULL | (1ULL << 8) | (1ULL << 16) | (1ULL << 24) | (1ULL << 32) | (1ULL << 40) | (1ULL << 48) | (1ULL << 56))

//#define PAC_COUNT_NONBANKED 1

struct PacCountState {
#ifdef PAC_COUNT_NONBANKED
	uint64_t probabilistic_total[64];
	void Flush(ArenaAllocator &) {
	}
	void GetTotalsAsDouble(double *dst) const {
		ToDoubleArray(probabilistic_total, dst);
	}
#else
	uint8_t subtotal8_count; // counts in subtotal8 before flush (max 255)
	uint8_t total_level;     // 0=none, 16, 32, or 64

	// Level 1: inline subtotal (SWAR uint8_t)
	uint64_t probabilistic_subtotal8[8]; // 64 bytes

	// Level 2: banked total storage (lazily allocated)
	uint8_t *banks1; // 2 banks (128 bytes)
	uint8_t *banks2; // 2 banks (128 bytes)
	uint8_t *banks3; // 4 banks (256 bytes)

	// Get bank pointer (0-7)
	inline uint8_t *GetBank(int n) const {
		if (n < 2)
			return banks1 + n * 64;
		if (n < 4)
			return banks2 + (n - 2) * 64;
		return banks3 + (n - 4) * 64;
	}

	// Number of banks for total level: 16->2, 32->4, 64->8
	static constexpr int BanksForLevel(int level) {
		return level / 8;
	}

	// Allocate banks for level (only allocates what's new)
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
	template <typename SRC_T, typename DST_T>
	void UpgradeTotal(ArenaAllocator &allocator, int new_level) {
		constexpr int SRC_PER_BANK = 64 / sizeof(SRC_T);
		constexpr int DST_PER_BANK = 64 / sizeof(DST_T);
		int old_banks = BanksForLevel(total_level);

		// Gather old values
		SRC_T old_values[64];
		for (int b = 0; b < old_banks; b++) {
			auto *src = reinterpret_cast<SRC_T *>(GetBank(b));
			for (int i = 0; i < SRC_PER_BANK; i++)
				old_values[b * SRC_PER_BANK + i] = src[i];
		}

		// Allocate new banks and scatter
		AllocateBanks(allocator, new_level);
		int new_banks = BanksForLevel(new_level);
		for (int b = 0; b < new_banks; b++) {
			auto *dst = reinterpret_cast<DST_T *>(GetBank(b));
			for (int i = 0; i < DST_PER_BANK; i++) {
				int idx = b * DST_PER_BANK + i;
				dst[i] = (idx < 64) ? static_cast<DST_T>(old_values[idx]) : 0;
			}
		}
		total_level = new_level;
	}

	// Add subtotal8 to total, upgrading if needed
	template <typename T>
	AUTOVECTORIZE void AddSubtotalToTotal() {
		constexpr int ELEMS_PER_BANK = 64 / sizeof(T);
		int num_banks = BanksForLevel(total_level);
		auto *sub = reinterpret_cast<const uint8_t *>(probabilistic_subtotal8);

		for (int b = 0; b < num_banks; b++) {
			auto *dst = reinterpret_cast<T *>(GetBank(b));
			for (int i = 0; i < ELEMS_PER_BANK; i++) {
				dst[i] += static_cast<T>(sub[b * ELEMS_PER_BANK + i]);
			}
		}
	}

	// Check if adding subtotal8 would overflow current level
	bool WouldOverflow(uint8_t add_count) const {
		if (total_level == 0)
			return false;
		uint64_t new_count = static_cast<uint64_t>(subtotal8_count) + add_count;
		if (total_level == 16)
			return new_count > UINT16_MAX;
		if (total_level == 32)
			return new_count > UINT32_MAX;
		return false; // uint64 can't overflow from uint8 additions
	}

	// Flush subtotal8 to total
	void Flush(ArenaAllocator &allocator) {
		if (subtotal8_count == 0)
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

		memset(probabilistic_subtotal8, 0, 64);
		subtotal8_count = 0;
	}

	// Estimate max count in any counter (conservative: assumes all bits set)
	uint64_t GetMaxPossibleCount() const {
		if (total_level == 64) {
			auto *p = reinterpret_cast<const uint64_t *>(banks1);
			uint64_t m = p[0];
			for (int i = 1; i < 64; i++)
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

	// Get element from banked total
	template <typename T>
	T GetTotalElement(int i) const {
		constexpr int PER_BANK = 64 / sizeof(T);
		int bank = i / PER_BANK, off = i % PER_BANK;
		return reinterpret_cast<const T *>(GetBank(bank))[off];
	}

	void GetTotalsAsDouble(double *dst) const {
		// First get subtotal8 values
		auto *sub = reinterpret_cast<const uint8_t *>(probabilistic_subtotal8);
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
