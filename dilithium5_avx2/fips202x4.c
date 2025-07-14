#include "fips202.h"
#include "fips202x4.h"
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define NROUNDS 24

/* Keccak round constants */
static const uint64_t KeccakF_RoundConstants[NROUNDS] = {
    (uint64_t)0x0000000000000001ULL,
    (uint64_t)0x0000000000008082ULL,
    (uint64_t)0x800000000000808aULL,
    (uint64_t)0x8000000080008000ULL,
    (uint64_t)0x000000000000808bULL,
    (uint64_t)0x0000000080000001ULL,
    (uint64_t)0x8000000080008081ULL,
    (uint64_t)0x8000000000008009ULL,
    (uint64_t)0x000000000000008aULL,
    (uint64_t)0x0000000000000088ULL,
    (uint64_t)0x0000000080008009ULL,
    (uint64_t)0x000000008000000aULL,
    (uint64_t)0x000000008000808bULL,
    (uint64_t)0x800000000000008bULL,
    (uint64_t)0x8000000000008089ULL,
    (uint64_t)0x8000000000008003ULL,
    (uint64_t)0x8000000000008002ULL,
    (uint64_t)0x8000000000000080ULL,
    (uint64_t)0x000000000000800aULL,
    (uint64_t)0x800000008000000aULL,
    (uint64_t)0x8000000080008081ULL,
    (uint64_t)0x8000000000008080ULL,
    (uint64_t)0x0000000080000001ULL,
    (uint64_t)0x8000000080008008ULL
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

    idx = _mm256_set_epi64x((long long)in3, (long long)in2, (long long)in1, (long long)in0);
    while (inlen >= r) {
        for (i = 0; i < r / 8; ++i) {
            t = _mm256_i64gather_epi64((long long *)pos, idx, 1);
            s[i] = _mm256_xor_si256(s[i], t);
            pos += 8;
        }
        inlen -= r;

        PQCLEAN_DILITHIUM5_AVX2_f1600x4(s, KeccakF_RoundConstants);
    }

    for (i = 0; i < inlen / 8; ++i) {
        t = _mm256_i64gather_epi64((long long *)pos, idx, 1);
        s[i] = _mm256_xor_si256(s[i], t);
        pos += 8;
    }
    inlen -= 8 * i;

    if (inlen) {
        t = _mm256_i64gather_epi64((long long *)pos, idx, 1);
        idx = _mm256_set1_epi64x((long long)((1ULL << (8 * inlen)) - 1));
        t = _mm256_and_si256(t, idx);
        s[i] = _mm256_xor_si256(s[i], t);
    }

    t = _mm256_set1_epi64x((uint64_t)p << 8 * inlen);
    s[i] = _mm256_xor_si256(s[i], t);
    t = _mm256_set1_epi64x((long long)(1ULL << 63));
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
        PQCLEAN_DILITHIUM5_AVX2_f1600x4(s, KeccakF_RoundConstants);
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

void PQCLEAN_DILITHIUM5_AVX2_shake128x4_absorb_once(keccakx4_state *state,
        const uint8_t *in0,
        const uint8_t *in1,
        const uint8_t *in2,
        const uint8_t *in3,
        size_t inlen) {
    keccakx4_absorb_once(state->s, SHAKE128_RATE, in0, in1, in2, in3, inlen, 0x1F);
}

void PQCLEAN_DILITHIUM5_AVX2_shake128x4_squeezeblocks(uint8_t *out0,
        uint8_t *out1,
        uint8_t *out2,
        uint8_t *out3,
        size_t nblocks,
        keccakx4_state *state) {
    keccakx4_squeezeblocks(out0, out1, out2, out3, nblocks, SHAKE128_RATE, state->s);
}

void PQCLEAN_DILITHIUM5_AVX2_shake256x4_absorb_once(keccakx4_state *state,
        const uint8_t *in0,
        const uint8_t *in1,
        const uint8_t *in2,
        const uint8_t *in3,
        size_t inlen) {
    keccakx4_absorb_once(state->s, SHAKE256_RATE, in0, in1, in2, in3, inlen, 0x1F);
}

void PQCLEAN_DILITHIUM5_AVX2_shake256x4_squeezeblocks(uint8_t *out0,
        uint8_t *out1,
        uint8_t *out2,
        uint8_t *out3,
        size_t nblocks,
        keccakx4_state *state) {
    keccakx4_squeezeblocks(out0, out1, out2, out3, nblocks, SHAKE256_RATE, state->s);
}

void PQCLEAN_DILITHIUM5_AVX2_shake128x4(uint8_t *out0,
                                        uint8_t *out1,
                                        uint8_t *out2,
                                        uint8_t *out3,
                                        size_t outlen,
                                        const uint8_t *in0,
                                        const uint8_t *in1,
                                        const uint8_t *in2,
                                        const uint8_t *in3,
                                        size_t inlen) {
    unsigned int i;
    size_t nblocks = outlen / SHAKE128_RATE;
    uint8_t t[4][SHAKE128_RATE];
    keccakx4_state state;

    PQCLEAN_DILITHIUM5_AVX2_shake128x4_absorb_once(&state, in0, in1, in2, in3, inlen);
    PQCLEAN_DILITHIUM5_AVX2_shake128x4_squeezeblocks(out0, out1, out2, out3, nblocks, &state);

    out0 += nblocks * SHAKE128_RATE;
    out1 += nblocks * SHAKE128_RATE;
    out2 += nblocks * SHAKE128_RATE;
    out3 += nblocks * SHAKE128_RATE;
    outlen -= nblocks * SHAKE128_RATE;

    if (outlen) {
        PQCLEAN_DILITHIUM5_AVX2_shake128x4_squeezeblocks(t[0], t[1], t[2], t[3], 1, &state);
        for (i = 0; i < outlen; ++i) {
            out0[i] = t[0][i];
            out1[i] = t[1][i];
            out2[i] = t[2][i];
            out3[i] = t[3][i];
        }
    }
}

void PQCLEAN_DILITHIUM5_AVX2_shake256x4(uint8_t *out0,
                                        uint8_t *out1,
                                        uint8_t *out2,
                                        uint8_t *out3,
                                        size_t outlen,
                                        const uint8_t *in0,
                                        const uint8_t *in1,
                                        const uint8_t *in2,
                                        const uint8_t *in3,
                                        size_t inlen) {
    unsigned int i;
    size_t nblocks = outlen / SHAKE256_RATE;
    uint8_t t[4][SHAKE256_RATE];
    keccakx4_state state;

    PQCLEAN_DILITHIUM5_AVX2_shake256x4_absorb_once(&state, in0, in1, in2, in3, inlen);
    PQCLEAN_DILITHIUM5_AVX2_shake256x4_squeezeblocks(out0, out1, out2, out3, nblocks, &state);

    out0 += nblocks * SHAKE256_RATE;
    out1 += nblocks * SHAKE256_RATE;
    out2 += nblocks * SHAKE256_RATE;
    out3 += nblocks * SHAKE256_RATE;
    outlen -= nblocks * SHAKE256_RATE;

    if (outlen) {
        PQCLEAN_DILITHIUM5_AVX2_shake256x4_squeezeblocks(t[0], t[1], t[2], t[3], 1, &state);
        for (i = 0; i < outlen; ++i) {
            out0[i] = t[0][i];
            out1[i] = t[1][i];
            out2[i] = t[2][i];
            out3[i] = t[3][i];
        }
    }
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