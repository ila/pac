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

// PAC_COUNT uses SWAR (SIMD Within A Register) for fast probabilistic counting.
// 8 uint64_t hold 64 packed uint8_t counters. On overflow (255), flush to uint64_t[64].
#define PAC_COUNT_MASK                                                                                                 \
	(1ULL | (1ULL << 8) | (1ULL << 16) | (1ULL << 24) | (1ULL << 32) | (1ULL << 40) | (1ULL << 48) | (1ULL << 56))
// For each of the 8 iterations i, we then do (hash_key>>i) & PAC_COUNT_MASK which selects 8 bits, and then add these
// with a single uint64_t ADD to a uint64 subtotal.
//
// This technique is known as SWAR: SIMD Within A Register
//
// You can only add 255 times before the bytes in this uint64_t start touching each other (causing overflow).
// So after 255 iterations, the probabilistic_total8[64] are added to uint64_t probabilistic_total[64] and reset to 0.
//
// The idea is that we get very fast performance 255 times and slower performance once every 256 only.
// This SIMD-friendly implementation can make PAC counting almost as fast as normal counting.
//
// Define PAC_COUNT_NOCASCADING for a naive implementation that directly updates uint64_t[64] counters.
// This is slower but simpler and useful for benchmarking the SWAR optimization.

// #define PAC_COUNT_NOBUFFERING 1
// #define PAC_COUNT_NOCASCADING 1
// #define PAC_COUNT_NOSIMD 1
#if defined(PAC_SUMAVG_NOSIMD) && !defined(PAC_SUMAVG_NOCASCADING)
PAC_COUNT_NOSIMD only makes sense in combination with PAC_COUNT_NOCASCADING
#endif

    // SWAR-optimized state
    struct PacCountState {
#ifndef PAC_COUNT_NOCASCADING
	uint64_t probabilistic_total8[8]; // SWAR packed uint8_t counters
	uint8_t exact_total8;             // counts updates, flush at 255
#endif
	uint64_t probabilistic_total[64]; // final totals

	void FlushLevel() {
#ifndef PAC_COUNT_NOCASCADING
		if (exact_total8 == 0) {
			return;
		}
		const uint8_t *src = reinterpret_cast<const uint8_t *>(probabilistic_total8);
		for (int bit = 0; bit < 64; bit++) {
			probabilistic_total[bit] += src[bit]; // we ignore SWAR order-changing here (does not matter for noising)
		}
		memset(probabilistic_total8, 0, sizeof(probabilistic_total8));
		exact_total8 = 0;
#endif
	}
	void GetTotalsAsDouble(double *dst) const {
		ToDoubleArray(probabilistic_total, dst);
	}
	PacCountState *GetState() {
		return this;
	}
	PacCountState *EnsureState(ArenaAllocator &) {
		return this;
	}
};

#ifndef PAC_COUNT_NOCASCADING
AUTOVECTORIZE static inline void PacCountUpdateSWAR(PacCountState &state, uint64_t key_hash) {
	for (int j = 0; j < 8; j++) { // just 8, not 64 iterations (SWAR: we count 8 bits every iteration)
		state.probabilistic_total8[j] += (key_hash >> j) & PAC_COUNT_MASK;
	}
}
#endif

// Update one hash - direct for PacCountState (always available)
template <typename S>
static inline void PacCountUpdateOne(S &agg, uint64_t hash, ArenaAllocator &a);

#ifndef PAC_COUNT_NOCASCADING
template <>
inline void PacCountUpdateOne(PacCountState &agg, uint64_t hash, ArenaAllocator &) {
	PacCountUpdateSWAR(agg, hash);
	if (++agg.exact_total8 == 255) {
		agg.FlushLevel();
	}
}
#else
// NOCASCADING: simple direct update to uint64_t[64]
template <>
inline void PacCountUpdateOne(PacCountState &agg, uint64_t hash, ArenaAllocator &) {
	for (int i = 0; i < 64; i++) {
#ifdef PAC_COUNT_NOSIMD
		if ((hash >> i) & 1) {
			agg.probabilistic_total[i]++;
		}
#else
		agg.probabilistic_total[i] += (hash >> i) & 1;
#endif
	}
}
#endif

#if !defined(PAC_COUNT_NOBUFFERING) && !defined(PAC_COUNT_NOCASCADING)
// Wrapped aggregation state: buffers 3 hashes before allocating PacCountState.
// Uses pointer tagging: lower 3 bits store n_buffered (0-3), upper 61 bits store PacCountState*.
// This is intended for situations where DuckDB is abandoning hash tables when it struggles to
// find duplicate keys. In this case, we really only want to allocate state memory in the dst
// of a Combine(src,dst). During the scatter-update state we survive with this small state (32 bytes).
struct PacCountStateWrapper {
	uint64_t hash_buf[3];
	union {
		uint64_t n_buffered;  // we only look at the lowest 3 bits
		PacCountState *state; // is a 8-byte aligned pointer (we misuse the lowest 3 bits as counter)
	};

	PacCountState *GetState() const {
		return reinterpret_cast<PacCountState *>(reinterpret_cast<uintptr_t>(state) & ~7ULL);
	}

	PacCountState *EnsureState(ArenaAllocator &a) {
		PacCountState *s = GetState();
		if (!s) {
			s = reinterpret_cast<PacCountState *>(a.Allocate(sizeof(PacCountState)));
			memset(s, 0, sizeof(PacCountState));
			state = s;
		}
		return s;
	}

	// Flush buffered hashes + key_hash into dst state
	AUTOVECTORIZE inline void FlushBufferInternal(PacCountState &dst, uint64_t cnt) {
		if (cnt + dst.exact_total8 >= 255) {
			dst.FlushLevel(); // make room
		}
		for (uint64_t i = 0; i < cnt; i++) {
			PacCountUpdateSWAR(dst, hash_buf[i]);
		}
	}

	inline void FlushBuffer(PacCountStateWrapper &agg, ArenaAllocator &a) {
		uint64_t cnt = agg.n_buffered & 7;
		if (cnt > 0) {
			auto dst = *agg.EnsureState(a);
			FlushBufferInternal(dst, cnt);
			agg.n_buffered &= ~7ULL;
			dst.exact_total8 += cnt;
		}
	}
};

template <>
AUTOVECTORIZE inline void PacCountUpdateOne(PacCountStateWrapper &agg, uint64_t key_hash, ArenaAllocator &a) {
	uint64_t cnt = agg.n_buffered & 7;
	if (DUCKDB_UNLIKELY(cnt == 3)) {
		auto &dst = *agg.EnsureState(a);
		auto n_buffered = agg.n_buffered & ~7ULL;
		agg.n_buffered = key_hash;       // hack: overwrite pointer temporarily
		agg.FlushBufferInternal(dst, 4); // we now have a buffer of 4
		agg.n_buffered = n_buffered;
		dst.exact_total8 += 4; // 3 buffered plus the new one (key_hash)
	} else {
		agg.hash_buf[cnt] = key_hash;
		agg.n_buffered++; // increments cnt
	}
}
#endif

} // namespace duckdb

#endif // PAC_COUNT_HPP
