#include "params.h"
#include "rejsample.h"
#include "symmetric.h"
#include "align.h"
#include "polyvec.h"
#include "fips202x8.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>


ALIGN(64) static const uint8_t idx_u[64] = {
        0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80,
        12, 13, 14, 0x80, 15, 16, 17, 0x80, 18, 19, 20, 0x80, 21, 22, 23, 0x80,
        24, 25, 26, 0x80, 27, 28, 29, 0x80, 30, 31, 32, 0x80, 33, 34, 35, 0x80,
        36, 37, 38, 0x80, 39, 40, 41, 0x80, 42, 43, 44, 0x80, 45, 46, 47, 0x80,
};


void XURQ_AVX512_rej_uniform8x(int32_t *r0,
                               int32_t *r1,
                               int32_t *r2,
                               int32_t *r3,
                               int32_t *r4,
                               int32_t *r5,
                               int32_t *r6,
                               int32_t *r7,
                               uint32_t ctr[8],
                               uint8_t buf[8][864]) {
    uint32_t pos = 0;
    uint16_t g0, g1, g2, g3, g4, g5, g6, g7;
    const __m512i bound = _mm512_set1_epi32(Q);
    const __m512i mask = _mm512_set1_epi32(0x7FFFFF);
    const __m512i index = _mm512_load_si512(idx_u);
    __m512i d0, d1, d2, d3, d4, d5, d6, d7;


    for (int i = 0; i < 16; ++i) {
        d0 = _mm512_loadu_si512(&buf[0][pos]);
        d1 = _mm512_loadu_si512(&buf[1][pos]);
        d2 = _mm512_loadu_si512(&buf[2][pos]);
        d3 = _mm512_loadu_si512(&buf[3][pos]);
        d4 = _mm512_loadu_si512(&buf[4][pos]);
        d5 = _mm512_loadu_si512(&buf[5][pos]);
        d6 = _mm512_loadu_si512(&buf[6][pos]);
        d7 = _mm512_loadu_si512(&buf[7][pos]);

        d0 = _mm512_permutexvar_epi8(index, d0);
        d1 = _mm512_permutexvar_epi8(index, d1);
        d2 = _mm512_permutexvar_epi8(index, d2);
        d3 = _mm512_permutexvar_epi8(index, d3);
        d4 = _mm512_permutexvar_epi8(index, d4);
        d5 = _mm512_permutexvar_epi8(index, d5);
        d6 = _mm512_permutexvar_epi8(index, d6);
        d7 = _mm512_permutexvar_epi8(index, d7);

        d0 &= mask;
        d1 &= mask;
        d2 &= mask;
        d3 &= mask;
        d4 &= mask;
        d5 &= mask;
        d6 &= mask;
        d7 &= mask;

        g0 = _mm512_cmplt_epi32_mask(d0, bound);
        g1 = _mm512_cmplt_epi32_mask(d1, bound);
        g2 = _mm512_cmplt_epi32_mask(d2, bound);
        g3 = _mm512_cmplt_epi32_mask(d3, bound);
        g4 = _mm512_cmplt_epi32_mask(d4, bound);
        g5 = _mm512_cmplt_epi32_mask(d5, bound);
        g6 = _mm512_cmplt_epi32_mask(d6, bound);
        g7 = _mm512_cmplt_epi32_mask(d7, bound);

        _mm512_mask_compressstoreu_epi32(r0 + ctr[0], g0, d0);
        _mm512_mask_compressstoreu_epi32(r1 + ctr[1], g1, d1);
        _mm512_mask_compressstoreu_epi32(r2 + ctr[2], g2, d2);
        _mm512_mask_compressstoreu_epi32(r3 + ctr[3], g3, d3);
        _mm512_mask_compressstoreu_epi32(r4 + ctr[4], g4, d4);
        _mm512_mask_compressstoreu_epi32(r5 + ctr[5], g5, d5);
        _mm512_mask_compressstoreu_epi32(r6 + ctr[6], g6, d6);
        _mm512_mask_compressstoreu_epi32(r7 + ctr[7], g7, d7);

        ctr[0] += _mm_popcnt_u32(g0);
        ctr[1] += _mm_popcnt_u32(g1);
        ctr[2] += _mm_popcnt_u32(g2);
        ctr[3] += _mm_popcnt_u32(g3);
        ctr[4] += _mm_popcnt_u32(g4);
        ctr[5] += _mm_popcnt_u32(g5);
        ctr[6] += _mm_popcnt_u32(g6);
        ctr[7] += _mm_popcnt_u32(g7);

        pos += 48;
    }

    uint32_t t;
    int32_t *rr[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
    for (int i = 0; i < 8; ++i) {
        pos = 768;
        while (ctr[i] < N && pos < 840) {
            t = buf[i][pos++];
            t |= (uint32_t) buf[i][pos++] << 8;
            t |= (uint32_t) buf[i][pos++] << 16;
            t &= 0x7FFFFF;

            if (t < Q) {
                rr[i][ctr[i]] = t;
                ctr[i]++;
            }
        }
    }

}


int XURQ_rej_uniform_avx512(int32_t *r, uint8_t *buf) {
    uint32_t pos = 0;
    uint16_t g0, g1, g2, g3;
    const __m512i bound = _mm512_set1_epi32(Q);
    const __m512i mask = _mm512_set1_epi32(0x7FFFFF);
    const __m512i index = _mm512_load_si512(idx_u);
    __m512i d0, d1, d2, d3;
    int ctr0, ctr1, ctr2, ctr3;
    ctr0 = 0;

    for (int i = 0; i < 4; ++i) {
        d0 = _mm512_loadu_si512(&buf[pos]);
        d1 = _mm512_loadu_si512(&buf[pos + 48]);
        d2 = _mm512_loadu_si512(&buf[pos + 96]);
        d3 = _mm512_loadu_si512(&buf[pos + 144]);

        d0 = _mm512_permutexvar_epi8(index, d0);
        d1 = _mm512_permutexvar_epi8(index, d1);
        d2 = _mm512_permutexvar_epi8(index, d2);
        d3 = _mm512_permutexvar_epi8(index, d3);

        d0 &= mask;
        d1 &= mask;
        d2 &= mask;
        d3 &= mask;

        g0 = _mm512_cmplt_epi32_mask(d0, bound);
        g1 = _mm512_cmplt_epi32_mask(d1, bound);
        g2 = _mm512_cmplt_epi32_mask(d2, bound);
        g3 = _mm512_cmplt_epi32_mask(d3, bound);

        _mm512_mask_compressstoreu_epi32(r + ctr0, g0, d0);
        ctr0 += _mm_popcnt_u32(g0);
        _mm512_mask_compressstoreu_epi32(r + ctr0, g1, d1);
        ctr0 += _mm_popcnt_u32(g1);
        _mm512_mask_compressstoreu_epi32(r + ctr0, g2, d2);
        ctr0 += _mm_popcnt_u32(g2);
        _mm512_mask_compressstoreu_epi32(r + ctr0, g3, d3);
        ctr0 += _mm_popcnt_u32(g3);

        pos += 192;
    }

    uint32_t t;
    while (ctr0 < N && pos < 840) {
        t = buf[pos++];
        t |= (uint32_t) buf[pos++] << 8;
        t |= (uint32_t) buf[pos++] << 16;
        t &= 0x7FFFFF;

        if (t < Q) {
            r[ctr0] = t;
            ctr0++;
        }
    }

    return ctr0;
}

ALIGN(64) static const uint8_t z_up[64] = {
        0, 1, 2, 0,
        2, 3, 4, 0,
        5, 6, 7, 0,
        7, 8, 9, 0,
        10, 11, 12, 0,
        12, 13, 14, 0,
        15, 16, 17, 0,
        17, 18, 19, 0,
        20, 21, 22, 0,
        22, 23, 24, 0,
        25, 26, 27, 0,
        27, 28, 29, 0,
        30, 31, 32, 0,
        32, 33, 34, 0,
        35, 36, 37, 0,
        37, 38, 39, 0
};

void polyz_unpack_avx512(poly *r, const uint8_t *buf) {
    __m512i a0, a1, a2, a3;
    int pos = 0;
    int ctr = 0;
    const __m512i mask = _mm512_set1_epi32(0xFFFFF);
    const __m512i gamma = _mm512_set1_epi32(GAMMA1);
    const __m512i index = _mm512_load_si512(z_up);
    const __m512i index5 = _mm512_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4);


    for (int i = 0; i < 4; ++i) {
        a0 = _mm512_loadu_si512(buf + pos);
        a1 = _mm512_loadu_si512(buf + pos + 40);
        a2 = _mm512_loadu_si512(buf + pos + 80);
        a3 = _mm512_loadu_si512(buf + pos + 120);

        a0 = _mm512_permutexvar_epi8(index, a0);
        a1 = _mm512_permutexvar_epi8(index, a1);
        a2 = _mm512_permutexvar_epi8(index, a2);
        a3 = _mm512_permutexvar_epi8(index, a3);

        a0 = _mm512_srlv_epi32(a0, index5);
        a1 = _mm512_srlv_epi32(a1, index5);
        a2 = _mm512_srlv_epi32(a2, index5);
        a3 = _mm512_srlv_epi32(a3, index5);

        a0 &= mask;
        a1 &= mask;
        a2 &= mask;
        a3 &= mask;

        a0 = _mm512_sub_epi32(gamma, a0);
        a1 = _mm512_sub_epi32(gamma, a1);
        a2 = _mm512_sub_epi32(gamma, a2);
        a3 = _mm512_sub_epi32(gamma, a3);

        _mm512_storeu_epi32(r->coeffs + ctr, a0);
        _mm512_storeu_epi32(r->coeffs + ctr + 16, a1);
        _mm512_storeu_epi32(r->coeffs + ctr + 32, a2);
        _mm512_storeu_epi32(r->coeffs + ctr + 48, a3);

        pos += 160;
        ctr += 64;
    }

}

void XURQ_AVX512_precompute_gamma1_8x(uint8_t buf[8][840], const uint64_t seed[6],
                                      uint16_t nonce0, uint16_t nonce1, uint16_t nonce2, uint16_t nonce3,
                                      uint16_t nonce4, uint16_t nonce5, uint16_t nonce6, uint16_t nonce7) {
    keccakx8_state state;

    state.s[0] = _mm512_set1_epi64(seed[0]);
    state.s[1] = _mm512_set1_epi64(seed[1]);
    state.s[2] = _mm512_set1_epi64(seed[2]);
    state.s[3] = _mm512_set1_epi64(seed[3]);
    state.s[4] = _mm512_set1_epi64(seed[4]);
    state.s[5] = _mm512_set1_epi64(seed[5]);
    state.s[6] = _mm512_set_epi64((0x1fULL << 16) ^ nonce7, (0x1fULL << 16) ^ nonce6, (0x1fULL << 16) ^ nonce5,
                                  (0x1fULL << 16) ^ nonce4,
                                  (0x1fULL << 16) ^ nonce3, (0x1fULL << 16) ^ nonce2, (0x1fULL << 16) ^ nonce1,
                                  (0x1fULL << 16) ^ nonce0);

    for (int j = 7; j < 25; ++j)
        state.s[j] = _mm512_setzero_si512();

    state.s[16] = _mm512_set1_epi64(0x1ULL << 63);
    XURQ_AVX512_shake256x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 6);

}


void ExpandMask(polyvecl *z,
                const uint8_t seed[64],
                uint16_t nonce) {
    ALIGN(64) uint8_t buf[8][840];
    XURQ_AVX512_precompute_gamma1_8x(buf, (uint64_t *) seed,
                                     nonce, nonce + 1, nonce + 2, nonce + 3,
                                     nonce + 4, nonce + 5, nonce + 6, 0);

    polyz_unpack_avx512(&z->vec[0], buf[0]);
    polyz_unpack_avx512(&z->vec[1], buf[1]);
    polyz_unpack_avx512(&z->vec[2], buf[2]);
    polyz_unpack_avx512(&z->vec[3], buf[3]);
    polyz_unpack_avx512(&z->vec[4], buf[4]);
    polyz_unpack_avx512(&z->vec[5], buf[5]);
    polyz_unpack_avx512(&z->vec[6], buf[6]);
}
