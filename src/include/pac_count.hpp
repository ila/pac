//
// Created by ila on 12/19/25.
//

#ifndef PAC_COUNT_HPP
#define PAC_COUNT_HPP

#include "duckdb.hpp"
#include "pac_aggregate.hpp"
#include <random>

namespace duckdb {

// PAC_COUNT(key_hash) implements a COUNT aggregate that for each privacy-unit (identified by a key_hash)
// computes 64 independent counts, where each independent count randomly (50% chance) includes a PU or not.
// The observation is that the 64-bits of a hashed key of the PU are random (50% 0, 50% 1), so we can take
// the 64 bits of the key to make 64 independent decisions.
//
// A COUNT() aggregate in its implementation simply performs total += 1
//
// PAC_COUNT() needs to do for(i=0; i<64; i++) total[i] += (key_hash >> i) & 1; (extract bit i)
//
// We want to do this in a SIMD-friendly way. Therefore, we want to create 64 totals of uint8_t (i.e. bytes),
// and perform 64 byte-additions, because in the widest SIMD implementation, AVX512, this means that this
// could be done in a *SINGLE* instruction (AVX512 has 64 lanes of uint8, as 64x8=512)
//
// However, to help auto-vectorizing compilers get this right, we use not 64 x uint8_t totals, but 8 x uint64_t
// totals because key_hash is already uint64_t. We apply this mask to key_hash:

#define PAC_COUNT_MASK  \
   ((1ULL << 0) | (1ULL << 8) | (1ULL << 16) | (1ULL << 24) | (1ULL << 32) | (1ULL << 40) | (1ULL << 48) | (1ULL << 56))

// for each of the 8 iterations i, we then do (hash_key>>i) & PAC_COUNT_MASK which selects 8 bits, and then add these
// with a single uint64_t add to a uint64 total8. You can only do that 255 times before overflow starts to happen.
// So after 255 iterations, the totals8  are added to full (uint64_t) totals64 and reset to 0.
//
// This means we get very fast performance 254 times and slower performance once every 255 only.
// This SIMD-friendly implementation should make PAC counting almost as fast as normal counting.

// State for pac_count: 64 counters and totals8 intermediate accumulators
struct PacCountState {
	uint64_t totals8[8];   // SIMD-friendly intermediate accumulators (8 x 8 bytes)
	uint64_t totals64[64]; // Final counters (64 x 8 bytes)
	uint8_t update_count;  // Counts updates, flushes when wraps to 0

	AUTOVECTORIZE void inline Flush() {
		const uint8_t *small = reinterpret_cast<const uint8_t *>(totals8);
		for (int i = 0; i < 64; i++) {
			totals64[i] += small[i];
		}
		memset(totals8, 0, sizeof(totals8));
	}
};

// Register the pac_count aggregate functions with the loader
void RegisterPacCountFunctions(ExtensionLoader &);

} // namespace duckdb

#endif // PAC_COUNT_HPP
