#include "fips202.h"
#include "fips202x4.h"
#include "keccak4x/KeccakP-1600-times4-SnP.h"
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define NROUNDS 24

/* Keccak round constants */
static const uint64_t KeccakF_RoundConstants[NROUNDS] = {
        (uint64_t) 0x0000000000000001ULL,
        (uint64_t) 0x0000000000008082ULL,
        (uint64_t) 0x800000000000808aULL,
        (uint64_t) 0x8000000080008000ULL,
        (uint64_t) 0x000000000000808bULL,
        (uint64_t) 0x0000000080000001ULL,
        (uint64_t) 0x8000000080008081ULL,
        (uint64_t) 0x8000000000008009ULL,
        (uint64_t) 0x000000000000008aULL,
        (uint64_t) 0x0000000000000088ULL,
        (uint64_t) 0x0000000080008009ULL,
        (uint64_t) 0x000000008000000aULL,
        (uint64_t) 0x000000008000808bULL,
        (uint64_t) 0x800000000000008bULL,
        (uint64_t) 0x8000000000008089ULL,
        (uint64_t) 0x8000000000008003ULL,
        (uint64_t) 0x8000000000008002ULL,
        (uint64_t) 0x8000000000000080ULL,
        (uint64_t) 0x000000000000800aULL,
        (uint64_t) 0x800000008000000aULL,
        (uint64_t) 0x8000000080008081ULL,
        (uint64_t) 0x8000000000008080ULL,
        (uint64_t) 0x0000000080000001ULL,
        (uint64_t) 0x8000000080008008ULL
};

static void keccakx4_absorb_once(__m256i s[25],
                                 unsigned int r,
                                 const uint8_t *in0,
                                 const uint8_t *in1,
                                 const uint8_t *in2,
                                 const uint8_t *in3,
                                 size_t inlen,
                                 uint8_t p) {
    size_t i;
    uint64_t pos = 0;
    __m256i t, idx;

    for (i = 0; i < 25; ++i) {
        s[i] = _mm256_setzero_si256();
    }

    idx = _mm256_set_epi64x((long long) in3, (long long) in2, (long long) in1, (long long) in0);
    while (inlen >= r) {
        for (i = 0; i < r / 8; ++i) {
            t = _mm256_i64gather_epi64((long long *) pos, idx, 1);
            s[i] = _mm256_xor_si256(s[i], t);
            pos += 8;
        }
        inlen -= r;

        PQCLEAN_DILITHIUM3_AVX2_f1600x4(s, KeccakF_RoundConstants);
    }

    for (i = 0; i < inlen / 8; ++i) {
        t = _mm256_i64gather_epi64((long long *) pos, idx, 1);
        s[i] = _mm256_xor_si256(s[i], t);
        pos += 8;
    }
    inlen -= 8 * i;

    if (inlen) {
        t = _mm256_i64gather_epi64((long long *) pos, idx, 1);
        idx = _mm256_set1_epi64x((long long) ((1ULL << (8 * inlen)) - 1));
        t = _mm256_and_si256(t, idx);
        s[i] = _mm256_xor_si256(s[i], t);
    }

    t = _mm256_set1_epi64x((uint64_t) p << 8 * inlen);
    s[i] = _mm256_xor_si256(s[i], t);
    t = _mm256_set1_epi64x((long long) (1ULL << 63));
    s[r / 8 - 1] = _mm256_xor_si256(s[r / 8 - 1], t);
}

static void keccakx4_squeezeblocks(uint8_t *out0,
                                   uint8_t *out1,
                                   uint8_t *out2,
                                   uint8_t *out3,
                                   size_t nblocks,
                                   unsigned int r,
                                   __m256i s[25]) {
    unsigned int i;
    double temp0, temp1;
    __m128d t;

    while (nblocks > 0) {
        PQCLEAN_DILITHIUM3_AVX2_f1600x4(s, KeccakF_RoundConstants);
        for (i = 0; i < r / 8; ++i) {
            t = _mm_castsi128_pd(_mm256_castsi256_si128(s[i]));
            _mm_storel_pd(&temp0, t);
            _mm_storeh_pd(&temp1, t);
            memcpy(&out0[8 * i], &temp0, sizeof(double));
            memcpy(&out1[8 * i], &temp1, sizeof(double));
            t = _mm_castsi128_pd(_mm256_extracti128_si256(s[i], 1));
            _mm_storel_pd(&temp0, t);
            _mm_storeh_pd(&temp1, t);
            memcpy(&out2[8 * i], &temp0, sizeof(double));
            memcpy(&out3[8 * i], &temp1, sizeof(double));
        }

        out0 += r;
        out1 += r;
        out2 += r;
        out3 += r;
        --nblocks;
    }
}

void PQCLEAN_DILITHIUM3_AVX2_shake128x4_absorb_once(keccakx4_state *state,
                                                    const uint8_t *in0,
                                                    const uint8_t *in1,
                                                    const uint8_t *in2,
                                                    const uint8_t *in3,
                                                    size_t inlen) {
    keccakx4_absorb_once(state->s, SHAKE128_RATE, in0, in1, in2, in3, inlen, 0x1F);
}

void PQCLEAN_DILITHIUM3_AVX2_shake128x4_squeezeblocks(uint8_t *out0,
                                                      uint8_t *out1,
                                                      uint8_t *out2,
                                                      uint8_t *out3,
                                                      size_t nblocks,
                                                      keccakx4_state *state) {
    keccakx4_squeezeblocks(out0, out1, out2, out3, nblocks, SHAKE128_RATE, state->s);
}


void PQCLEAN_DILITHIUM3_AVX2_shake256x4_absorb_once(keccakx4_state *state,
                                                    const uint8_t *in0,
                                                    const uint8_t *in1,
                                                    const uint8_t *in2,
                                                    const uint8_t *in3,
                                                    size_t inlen) {
    keccakx4_absorb_once(state->s, SHAKE256_RATE, in0, in1, in2, in3, inlen, 0x1F);
}

void PQCLEAN_DILITHIUM3_AVX2_shake256x4_squeezeblocks(uint8_t *out0,
                                                      uint8_t *out1,
                                                      uint8_t *out2,
                                                      uint8_t *out3,
                                                      size_t nblocks,
                                                      keccakx4_state *state) {
    keccakx4_squeezeblocks(out0, out1, out2, out3, nblocks, SHAKE256_RATE, state->s);
}


void XURQ_AVX2_shake128x4_squeezeblocks(uint8_t *out0,
                                        uint8_t *out1,
                                        uint8_t *out2,
                                        uint8_t *out3,
                                        int nblocks,
                                        keccakx4_state *state) {
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    __m256i f0, f1, f2, f3, f4, f5, f6, f7;
    __m128i t;

    for (int i = 0; i < nblocks; ++i) {
        KeccakP1600times4_PermuteAll_24rounds(state->s);

        t0 = _mm256_unpacklo_epi64(state->s[0], state->s[1]);
        t1 = _mm256_unpackhi_epi64(state->s[0], state->s[1]);
        t2 = _mm256_unpacklo_epi64(state->s[2], state->s[3]);
        t3 = _mm256_unpackhi_epi64(state->s[2], state->s[3]);

        t4 = _mm256_unpacklo_epi64(state->s[4], state->s[5]);
        t5 = _mm256_unpackhi_epi64(state->s[4], state->s[5]);
        t6 = _mm256_unpacklo_epi64(state->s[6], state->s[7]);
        t7 = _mm256_unpackhi_epi64(state->s[6], state->s[7]);

        f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
        f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
        f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
        f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

        f4 = _mm256_permute2x128_si256(t4, t6, 0x20);
        f5 = _mm256_permute2x128_si256(t5, t7, 0x20);
        f6 = _mm256_permute2x128_si256(t4, t6, 0x31);
        f7 = _mm256_permute2x128_si256(t5, t7, 0x31);


        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) out3, f3);
        _mm256_storeu_si256((__m256i *) (out0 + 32), f4);
        _mm256_storeu_si256((__m256i *) (out1 + 32), f5);
        _mm256_storeu_si256((__m256i *) (out2 + 32), f6);
        _mm256_storeu_si256((__m256i *) (out3 + 32), f7);

        out0 += 64;
        out1 += 64;
        out2 += 64;
        out3 += 64;

        t0 = _mm256_unpacklo_epi64(state->s[8], state->s[9]);
        t1 = _mm256_unpackhi_epi64(state->s[8], state->s[9]);
        t2 = _mm256_unpacklo_epi64(state->s[10], state->s[11]);
        t3 = _mm256_unpackhi_epi64(state->s[10], state->s[11]);

        t4 = _mm256_unpacklo_epi64(state->s[12], state->s[13]);
        t5 = _mm256_unpackhi_epi64(state->s[12], state->s[13]);
        t6 = _mm256_unpacklo_epi64(state->s[14], state->s[15]);
        t7 = _mm256_unpackhi_epi64(state->s[14], state->s[15]);

        f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
        f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
        f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
        f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

        f4 = _mm256_permute2x128_si256(t4, t6, 0x20);
        f5 = _mm256_permute2x128_si256(t5, t7, 0x20);
        f6 = _mm256_permute2x128_si256(t4, t6, 0x31);
        f7 = _mm256_permute2x128_si256(t5, t7, 0x31);

        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) out3, f3);
        _mm256_storeu_si256((__m256i *) (out0 + 32), f4);
        _mm256_storeu_si256((__m256i *) (out1 + 32), f5);
        _mm256_storeu_si256((__m256i *) (out2 + 32), f6);
        _mm256_storeu_si256((__m256i *) (out3 + 32), f7);

        out0 += 64;
        out1 += 64;
        out2 += 64;
        out3 += 64;

        t0 = _mm256_unpacklo_epi64(state->s[16], state->s[17]);
        t1 = _mm256_unpackhi_epi64(state->s[16], state->s[17]);
        t2 = _mm256_unpacklo_epi64(state->s[18], state->s[19]);
        t3 = _mm256_unpackhi_epi64(state->s[18], state->s[19]);

        f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
        f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
        f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
        f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) out3, f3);

        out0 += 32;
        out1 += 32;
        out2 += 32;
        out3 += 32;

        t = _mm256_castsi256_si128(state->s[20]);
        *(uint64_t *) out0 = _mm_extract_epi64(t, 0);
        *(uint64_t *) out1 = _mm_extract_epi64(t, 1);
        t = _mm256_extracti128_si256(state->s[20], 1);
        *(uint64_t *) out2 = _mm_extract_epi64(t, 0);
        *(uint64_t *) out3 = _mm_extract_epi64(t, 1);

        out0 += 8;
        out1 += 8;
        out2 += 8;
        out3 += 8;

    }


}

void XURQ_AVX2_shake256x4_squeezeblocks(uint8_t *out0,
                                        uint8_t *out1,
                                        uint8_t *out2,
                                        uint8_t *out3,
                                        int nblocks,
                                        keccakx4_state *state) {
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    __m256i f0, f1, f2, f3, f4, f5, f6, f7;
    __m128i t;

    for (int i = 0; i < nblocks; ++i) {
        KeccakP1600times4_PermuteAll_24rounds(state->s);

        t0 = _mm256_unpacklo_epi64(state->s[0], state->s[1]);
        t1 = _mm256_unpackhi_epi64(state->s[0], state->s[1]);
        t2 = _mm256_unpacklo_epi64(state->s[2], state->s[3]);
        t3 = _mm256_unpackhi_epi64(state->s[2], state->s[3]);

        t4 = _mm256_unpacklo_epi64(state->s[4], state->s[5]);
        t5 = _mm256_unpackhi_epi64(state->s[4], state->s[5]);
        t6 = _mm256_unpacklo_epi64(state->s[6], state->s[7]);
        t7 = _mm256_unpackhi_epi64(state->s[6], state->s[7]);

        f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
        f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
        f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
        f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

        f4 = _mm256_permute2x128_si256(t4, t6, 0x20);
        f5 = _mm256_permute2x128_si256(t5, t7, 0x20);
        f6 = _mm256_permute2x128_si256(t4, t6, 0x31);
        f7 = _mm256_permute2x128_si256(t5, t7, 0x31);

        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) out3, f3);
        _mm256_storeu_si256((__m256i *) (out0 + 32), f4);
        _mm256_storeu_si256((__m256i *) (out1 + 32), f5);
        _mm256_storeu_si256((__m256i *) (out2 + 32), f6);
        _mm256_storeu_si256((__m256i *) (out3 + 32), f7);

        out0 += 64;
        out1 += 64;
        out2 += 64;
        out3 += 64;

        t0 = _mm256_unpacklo_epi64(state->s[8], state->s[9]);
        t1 = _mm256_unpackhi_epi64(state->s[8], state->s[9]);
        t2 = _mm256_unpacklo_epi64(state->s[10], state->s[11]);
        t3 = _mm256_unpackhi_epi64(state->s[10], state->s[11]);

        t4 = _mm256_unpacklo_epi64(state->s[12], state->s[13]);
        t5 = _mm256_unpackhi_epi64(state->s[12], state->s[13]);
        t6 = _mm256_unpacklo_epi64(state->s[14], state->s[15]);
        t7 = _mm256_unpackhi_epi64(state->s[14], state->s[15]);

        f0 = _mm256_permute2x128_si256(t0, t2, 0x20);
        f1 = _mm256_permute2x128_si256(t1, t3, 0x20);
        f2 = _mm256_permute2x128_si256(t0, t2, 0x31);
        f3 = _mm256_permute2x128_si256(t1, t3, 0x31);

        f4 = _mm256_permute2x128_si256(t4, t6, 0x20);
        f5 = _mm256_permute2x128_si256(t5, t7, 0x20);
        f6 = _mm256_permute2x128_si256(t4, t6, 0x31);
        f7 = _mm256_permute2x128_si256(t5, t7, 0x31);

        _mm256_storeu_si256((__m256i *) out0, f0);
        _mm256_storeu_si256((__m256i *) out1, f1);
        _mm256_storeu_si256((__m256i *) out2, f2);
        _mm256_storeu_si256((__m256i *) out3, f3);
        _mm256_storeu_si256((__m256i *) (out0 + 32), f4);
        _mm256_storeu_si256((__m256i *) (out1 + 32), f5);
        _mm256_storeu_si256((__m256i *) (out2 + 32), f6);
        _mm256_storeu_si256((__m256i *) (out3 + 32), f7);

        out0 += 64;
        out1 += 64;
        out2 += 64;
        out3 += 64;


        t = _mm256_castsi256_si128(state->s[16]);
        *(uint64_t *) out0 = _mm_extract_epi64(t, 0);
        *(uint64_t *) out1 = _mm_extract_epi64(t, 1);
        t = _mm256_extracti128_si256(state->s[16], 1);
        *(uint64_t *) out2 = _mm_extract_epi64(t, 0);
        *(uint64_t *) out3 = _mm_extract_epi64(t, 1);

        out0 += 8;
        out1 += 8;
        out2 += 8;
        out3 += 8;
    }


}
//For test
void PQCLEAN_DILITHIUM3_AVX2_shake128x4(uint8_t *out0,
                                        uint8_t *out1,
                                        uint8_t *out2,
                                        uint8_t *out3,
                                        const uint8_t *seed,
                                        uint16_t nonce0,
                                        uint16_t nonce1,
                                        uint16_t nonce2,
                                        uint16_t nonce3) {
    keccakx4_state state;

    __m256i f;

    f = _mm256_loadu_si256((__m256i *) seed);
    _mm256_store_si256(out0, f);
    _mm256_store_si256(out1, f);
    _mm256_store_si256(out2, f);
    _mm256_store_si256(out3, f);

    out0[32 + 0] = nonce0;
    out0[32 + 1] = nonce0 >> 8;
    out1[32 + 0] = nonce1;
    out1[32 + 1] = nonce1 >> 8;
    out2[32 + 0] = nonce2;
    out2[32 + 1] = nonce2 >> 8;
    out3[32 + 0] = nonce3;
    out3[32 + 1] = nonce3 >> 8;

    PQCLEAN_DILITHIUM3_AVX2_shake128x4_absorb_once(&state, out0, out1, out2, out3, 34);
    PQCLEAN_DILITHIUM3_AVX2_shake128x4_squeezeblocks(out0, out1, out2, out3, 1, &state);

}
//For test
void PQCLEAN_DILITHIUM3_AVX2_shake128x4_5round_squeeze(uint8_t *out0,
                                                       uint8_t *out1,
                                                       uint8_t *out2,
                                                       uint8_t *out3,
                                                       const uint8_t *seed,
                                                       uint16_t nonce0,
                                                       uint16_t nonce1,
                                                       uint16_t nonce2,
                                                       uint16_t nonce3) {
    keccakx4_state state;

    __m256i f;

    f = _mm256_loadu_si256((__m256i *) seed);
    _mm256_store_si256(out0, f);
    _mm256_store_si256(out1, f);
    _mm256_store_si256(out2, f);
    _mm256_store_si256(out3, f);

    out0[32 + 0] = nonce0;
    out0[32 + 1] = nonce0 >> 8;
    out1[32 + 0] = nonce1;
    out1[32 + 1] = nonce1 >> 8;
    out2[32 + 0] = nonce2;
    out2[32 + 1] = nonce2 >> 8;
    out3[32 + 0] = nonce3;
    out3[32 + 1] = nonce3 >> 8;

    PQCLEAN_DILITHIUM3_AVX2_shake128x4_absorb_once(&state, out0, out1, out2, out3, 34);
    PQCLEAN_DILITHIUM3_AVX2_shake128x4_squeezeblocks(out0, out1, out2, out3, 5, &state);

}
//For test
void PQCLEAN_DILITHIUM3_AVX2_shake256x4(uint8_t *out0,
                                        uint8_t *out1,
                                        uint8_t *out2,
                                        uint8_t *out3,
                                        const uint8_t *seed,
                                        uint16_t nonce0,
                                        uint16_t nonce1,
                                        uint16_t nonce2,
                                        uint16_t nonce3) {
    unsigned int i;

    keccakx4_state state;

    __m256i f;

    f = _mm256_loadu_si256((__m256i *) seed);
    _mm256_store_si256(out0, f);
    _mm256_store_si256(out1, f);
    _mm256_store_si256(out2, f);
    _mm256_store_si256(out3, f);

    out0[32 + 0] = nonce0;
    out0[32 + 1] = nonce0 >> 8;
    out1[32 + 0] = nonce1;
    out1[32 + 1] = nonce1 >> 8;
    out2[32 + 0] = nonce2;
    out2[32 + 1] = nonce2 >> 8;
    out3[32 + 0] = nonce3;
    out3[32 + 1] = nonce3 >> 8;


    PQCLEAN_DILITHIUM3_AVX2_shake256x4_absorb_once(&state, out0, out1, out2, out3, 34);
    PQCLEAN_DILITHIUM3_AVX2_shake256x4_squeezeblocks(out0, out1, out2, out3, 1, &state);

}


//For test
void PQCLEAN_DILITHIUM3_AVX2_shake256x4_5round_squeeze(uint8_t *out0,
                                                       uint8_t *out1,
                                                       uint8_t *out2,
                                                       uint8_t *out3,
                                                       const uint8_t *seed,
                                                       uint16_t nonce0,
                                                       uint16_t nonce1,
                                                       uint16_t nonce2,
                                                       uint16_t nonce3) {
    unsigned int i;

    keccakx4_state state;

    __m256i f;

    f = _mm256_loadu_si256((__m256i *) seed);
    _mm256_store_si256(out0, f);
    _mm256_store_si256(out1, f);
    _mm256_store_si256(out2, f);
    _mm256_store_si256(out3, f);

    out0[32 + 0] = nonce0;
    out0[32 + 1] = nonce0 >> 8;
    out1[32 + 0] = nonce1;
    out1[32 + 1] = nonce1 >> 8;
    out2[32 + 0] = nonce2;
    out2[32 + 1] = nonce2 >> 8;
    out3[32 + 0] = nonce3;
    out3[32 + 1] = nonce3 >> 8;


    PQCLEAN_DILITHIUM3_AVX2_shake256x4_absorb_once(&state, out0, out1, out2, out3, 34);
    PQCLEAN_DILITHIUM3_AVX2_shake256x4_squeezeblocks(out0, out1, out2, out3, 5, &state);

}


//For test
void XURQ_AVX2_shake128x4(uint8_t *out0,
                          uint8_t *out1,
                          uint8_t *out2,
                          uint8_t *out3,
                          const uint8_t *seed,
                          uint16_t nonce0,
                          uint16_t nonce1,
                          uint16_t nonce2,
                          uint16_t nonce3) {
    keccakx4_state state;

    uint64_t *seed64 = (uint64_t *) seed;

    state.s[0] = _mm256_set1_epi64x(seed64[0]);
    state.s[1] = _mm256_set1_epi64x(seed64[1]);
    state.s[2] = _mm256_set1_epi64x(seed64[2]);
    state.s[3] = _mm256_set1_epi64x(seed64[3]);
    state.s[4] = _mm256_set_epi64x((0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2,
                                   (0x1f << 16) ^ nonce1, (0x1f << 16) ^ nonce0);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm256_setzero_si256();

    state.s[20] = _mm256_set1_epi64x(0x1ULL << 63);

    XURQ_AVX2_shake128x4_squeezeblocks(out0, out1, out2, out3, 1, &state);
}

//For test
void XURQ_AVX2_shake128x4_5round_squeeze(uint8_t *out0,
                                         uint8_t *out1,
                                         uint8_t *out2,
                                         uint8_t *out3,
                                         const uint8_t *seed,
                                         uint16_t nonce0,
                                         uint16_t nonce1,
                                         uint16_t nonce2,
                                         uint16_t nonce3) {
    keccakx4_state state;

    uint64_t *seed64 = (uint64_t *) seed;

    state.s[0] = _mm256_set1_epi64x(seed64[0]);
    state.s[1] = _mm256_set1_epi64x(seed64[1]);
    state.s[2] = _mm256_set1_epi64x(seed64[2]);
    state.s[3] = _mm256_set1_epi64x(seed64[3]);
    state.s[4] = _mm256_set_epi64x((0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2,
                                   (0x1f << 16) ^ nonce1, (0x1f << 16) ^ nonce0);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm256_setzero_si256();

    state.s[20] = _mm256_set1_epi64x(0x1ULL << 63);

    XURQ_AVX2_shake128x4_squeezeblocks(out0, out1, out2, out3, 5, &state);
}


//For test
void XURQ_AVX2_shake256x4(uint8_t *out0,
                          uint8_t *out1,
                          uint8_t *out2,
                          uint8_t *out3,
                          const uint8_t *seed,
                          uint16_t nonce0,
                          uint16_t nonce1,
                          uint16_t nonce2,
                          uint16_t nonce3) {

    keccakx4_state state;

    uint64_t *seed64 = (uint64_t *) seed;

    state.s[0] = _mm256_set1_epi64x(seed64[0]);
    state.s[1] = _mm256_set1_epi64x(seed64[1]);
    state.s[2] = _mm256_set1_epi64x(seed64[2]);
    state.s[3] = _mm256_set1_epi64x(seed64[3]);
    state.s[4] = _mm256_set_epi64x((0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2,
                                   (0x1f << 16) ^ nonce1, (0x1f << 16) ^ nonce0);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm256_setzero_si256();

    state.s[16] = _mm256_set1_epi64x(0x1ULL << 63);

    XURQ_AVX2_shake256x4_squeezeblocks(out0, out1, out2, out3, 1, &state);

}

//For test
void XURQ_AVX2_shake256x4_5round_squeeze(uint8_t *out0,
                                         uint8_t *out1,
                                         uint8_t *out2,
                                         uint8_t *out3,
                                         const uint8_t *seed,
                                         uint16_t nonce0,
                                         uint16_t nonce1,
                                         uint16_t nonce2,
                                         uint16_t nonce3) {

    keccakx4_state state;

    uint64_t *seed64 = (uint64_t *) seed;

    state.s[0] = _mm256_set1_epi64x(seed64[0]);
    state.s[1] = _mm256_set1_epi64x(seed64[1]);
    state.s[2] = _mm256_set1_epi64x(seed64[2]);
    state.s[3] = _mm256_set1_epi64x(seed64[3]);
    state.s[4] = _mm256_set_epi64x((0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2,
                                   (0x1f << 16) ^ nonce1, (0x1f << 16) ^ nonce0);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm256_setzero_si256();

    state.s[16] = _mm256_set1_epi64x(0x1ULL << 63);

    XURQ_AVX2_shake128x4_squeezeblocks(out0, out1, out2, out3, 5, &state);
}