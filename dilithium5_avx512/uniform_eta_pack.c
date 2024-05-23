//
// Created by xurq on 2023/3/4.
//
#include <stdio.h>
#include "uniform_eta_pack.h"
#include "consts.h"
#include "params.h"
#include "align.h"
#include "fips202x8.h"

// XURQ_AVX512_poly_uniform_eta8x_with_pack(
//         sk + 3 * SEEDBYTES + 0 * POLYETA_PACKEDBYTES,
// sk + 3 * SEEDBYTES + 1 * POLYETA_PACKEDBYTES,
// sk + 3 * SEEDBYTES + 2 * POLYETA_PACKEDBYTES,
// sk + 3 * SEEDBYTES + 3 * POLYETA_PACKEDBYTES,
// sk + 3 * SEEDBYTES + (L + 0) * POLYETA_PACKEDBYTES,
// sk + 3 * SEEDBYTES + (L + 1) * POLYETA_PACKEDBYTES,
// sk + 3 * SEEDBYTES + (L + 2) * POLYETA_PACKEDBYTES,
// sk + 3 * SEEDBYTES + (L + 3) * POLYETA_PACKEDBYTES,
// &s1, &s2, rhoprime);




ALIGN(64) static const uint8_t idx_eta1[64] = {
        0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26, 32, 33, 34, 40,
        41, 42, 48, 49, 50, 56, 57, 58, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};


static void eta_pack8x(uint8_t r[POLYETA_PACKEDBYTES], const uint8_t *p) {
    __m512i b0, b1, b2, b3;

    uint64_t mmask = 0xffffff;
    const __m512i idx1 = _mm512_load_si512(idx_eta1);
    const __m512i mask0 = _mm512_set1_epi16(0xff);
    const __m512i mask1 = _mm512_set1_epi32(0xffff);
    const __m512i mask2 = _mm512_set1_epi64(0xffffffff);

    b0 = _mm512_loadu_si512(p);
    b1 = _mm512_loadu_si512(p + 64);
    b2 = _mm512_loadu_si512(p + 128);
    b3 = _mm512_loadu_si512(p + 192);

    b0 ^= _mm512_srli_epi16(b0, 5);
    b1 ^= _mm512_srli_epi16(b1, 5);
    b2 ^= _mm512_srli_epi16(b2, 5);
    b3 ^= _mm512_srli_epi16(b3, 5);

    b0 &= mask0;
    b1 &= mask0;
    b2 &= mask0;
    b3 &= mask0;

    b0 ^= _mm512_srli_epi32(b0, 10);
    b1 ^= _mm512_srli_epi32(b1, 10);
    b2 ^= _mm512_srli_epi32(b2, 10);
    b3 ^= _mm512_srli_epi32(b3, 10);

    b0 &= mask1;
    b1 &= mask1;
    b2 &= mask1;
    b3 &= mask1;

    b0 ^= _mm512_srli_epi64(b0, 20);
    b1 ^= _mm512_srli_epi64(b1, 20);
    b2 ^= _mm512_srli_epi64(b2, 20);
    b3 ^= _mm512_srli_epi64(b3, 20);

    b0 &= mask2;
    b1 &= mask2;
    b2 &= mask2;
    b3 &= mask2;

    b0 = _mm512_maskz_permutexvar_epi8(mmask, idx1, b0);
    b1 = _mm512_maskz_permutexvar_epi8(mmask, idx1, b1);
    b2 = _mm512_maskz_permutexvar_epi8(mmask, idx1, b2);
    b3 = _mm512_maskz_permutexvar_epi8(mmask, idx1, b3);

    _mm512_mask_storeu_epi8(r, mmask, b0);
    _mm512_mask_storeu_epi8(r + 24, mmask, b1);
    _mm512_mask_storeu_epi8(r + 48, mmask, b2);
    _mm512_mask_storeu_epi8(r + 72, mmask, b3);
}


int rej_eta_with_pipe_avx512(poly *s, uint8_t *p, const uint8_t *buf) {
    uint32_t pos = 0;
    int ctr = 0;
    uint64_t g0, g1, g2, g3;
    const __m512i bound = _mm512_set1_epi8(0x0f);
    __m512i c0, c1, c2, c3;
    __m512i b0, b1, b2, b3;

    const __m512i mask = _mm512_set1_epi8(0xc0);
    const __m512i num13 = _mm512_set1_epi16(13);
    const __m512i num5 = _mm512_set1_epi16(5);
    const __m512i eta = _mm512_set1_epi8(2);

    b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[pos]));
    b1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[pos + 32]));
    b2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[pos + 64]));
    b3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[pos + 96]));

    c0 = _mm512_slli_epi16(b0, 4);
    c1 = _mm512_slli_epi16(b1, 4);
    c2 = _mm512_slli_epi16(b2, 4);
    c3 = _mm512_slli_epi16(b3, 4);

    c0 = _mm512_ternarylogic_epi32(b0, c0, bound, 0xa8);
    c1 = _mm512_ternarylogic_epi32(b1, c1, bound, 0xa8);
    c2 = _mm512_ternarylogic_epi32(b2, c2, bound, 0xa8);
    c3 = _mm512_ternarylogic_epi32(b3, c3, bound, 0xa8);

    g0 = _mm512_cmp_epi8_mask(c0, bound, _MM_CMPINT_LT);
    g1 = _mm512_cmp_epi8_mask(c1, bound, _MM_CMPINT_LT);
    g2 = _mm512_cmp_epi8_mask(c2, bound, _MM_CMPINT_LT);
    g3 = _mm512_cmp_epi8_mask(c3, bound, _MM_CMPINT_LT);

    b0 = _mm512_mullo_epi16(c0, num13);
    b1 = _mm512_mullo_epi16(c1, num13);
    b2 = _mm512_mullo_epi16(c2, num13);
    b3 = _mm512_mullo_epi16(c3, num13);

    b0 &= mask;
    b1 &= mask;
    b2 &= mask;
    b3 &= mask;

    b0 = _mm512_srli_epi16(b0, 6);
    b1 = _mm512_srli_epi16(b1, 6);
    b2 = _mm512_srli_epi16(b2, 6);
    b3 = _mm512_srli_epi16(b3, 6);

    b0 = _mm512_mullo_epi16(b0, num5);
    b1 = _mm512_mullo_epi16(b1, num5);
    b2 = _mm512_mullo_epi16(b2, num5);
    b3 = _mm512_mullo_epi16(b3, num5);

    b0 = _mm512_sub_epi8(c0, b0);
    b1 = _mm512_sub_epi8(c1, b1);
    b2 = _mm512_sub_epi8(c2, b2);
    b3 = _mm512_sub_epi8(c3, b3);

    _mm512_mask_compressstoreu_epi8(p, g0, b0);
    p += _mm_popcnt_u64(g0);
    _mm512_mask_compressstoreu_epi8(p, g1, b1);
    p += _mm_popcnt_u64(g1);
    _mm512_mask_compressstoreu_epi8(p, g2, b2);
    p += _mm_popcnt_u64(g2);
    _mm512_mask_compressstoreu_epi8(p, g3, b3);
    p += _mm_popcnt_u64(g3);


    b0 = _mm512_sub_epi8(eta, b0);
    b1 = _mm512_sub_epi8(eta, b1);
    b2 = _mm512_sub_epi8(eta, b2);
    b3 = _mm512_sub_epi8(eta, b3);

    c0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b0, 0));
    c1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b0, 1));
    c2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b0, 2));
    c3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b0, 3));

    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, g0 & 0xffff, c0);
    ctr += _mm_popcnt_u64(g0 & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g0 >> 16) & 0xffff, c1);
    ctr += _mm_popcnt_u64((g0 >> 16) & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g0 >> 32) & 0xffff, c2);
    ctr += _mm_popcnt_u64((g0 >> 32) & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g0 >> 48) & 0xffff, c3);
    ctr += _mm_popcnt_u64((g0 >> 48) & 0xffff);

    c0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b1, 0));
    c1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b1, 1));
    c2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b1, 2));
    c3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b1, 3));

    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, g1 & 0xffff, c0);
    ctr += _mm_popcnt_u64(g1 & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g1 >> 16) & 0xffff, c1);
    ctr += _mm_popcnt_u64((g1 >> 16) & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g1 >> 32) & 0xffff, c2);
    ctr += _mm_popcnt_u64((g1 >> 32) & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g1 >> 48) & 0xffff, c3);
    ctr += _mm_popcnt_u64((g1 >> 48) & 0xffff);

    c0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b2, 0));
    c1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b2, 1));
    c2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b2, 2));
    c3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b2, 3));

    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, g2 & 0xffff, c0);
    ctr += _mm_popcnt_u64(g2 & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g2 >> 16) & 0xffff, c1);
    ctr += _mm_popcnt_u64((g2 >> 16) & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g2 >> 32) & 0xffff, c2);
    ctr += _mm_popcnt_u64((g2 >> 32) & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g2 >> 48) & 0xffff, c3);
    ctr += _mm_popcnt_u64((g2 >> 48) & 0xffff);


    c0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b3, 0));
    c1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b3, 1));
    c2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b3, 2));
    c3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b3, 3));

    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, g3 & 0xffff, c0);
    ctr += _mm_popcnt_u64(g3 & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g3 >> 16) & 0xffff, c1);
    ctr += _mm_popcnt_u64((g3 >> 16) & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g3 >> 32) & 0xffff, c2);
    ctr += _mm_popcnt_u64((g3 >> 32) & 0xffff);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, (g3 >> 48) & 0xffff, c3);
    ctr += _mm_popcnt_u64((g3 >> 48) & 0xffff);


    __m128i b, c;
    __m128i ibound = _mm_set1_epi16(0x0F0F);
    const __m128i inum13 = _mm_set1_epi16(13);
    const __m128i inum5 = _mm_set1_epi16(5);
    const __m128i ieta = _mm_set1_epi8(2);
    const __m128i imask = _mm_set1_epi8(0xc0);

    b = _mm_cvtepu8_epi16(_mm_loadu_epi8(buf + 128));
    c = _mm_slli_epi16(b, 4);
    c = (b | c) & ibound;
    g0 = _mm_cmp_epi8_mask(c, ibound, _MM_CMPINT_LT);
    b = _mm_mullo_epi16(c, inum13);
    b &= imask;
    b = _mm_srli_epi16(b, 6);
    b = _mm_mullo_epi16(b, inum5);
    b = _mm_sub_epi16(c, b);
    _mm_mask_compressstoreu_epi8(p , g0, b);
    b = _mm_sub_epi8(ieta, b);
    c0 = _mm512_cvtepi8_epi32(b);
    _mm512_mask_compressstoreu_epi32(s->coeffs + ctr, g0, c0);

    return ctr + _mm_popcnt_u64(g0);
}


static unsigned int rej_eta_with_pipe(int32_t *a,
                                      int ctr,
                                      int len,
                                      uint8_t *p,
                                      const uint8_t *buf,
                                      int buflen) {
    uint32_t t0, t1;
    int pos = 0;
    while (ctr < len && pos < buflen) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

        if (t0 < 15) {
            t0 = t0 - (205 * t0 >> 10) * 5;
            p[ctr] = t0;
            a[ctr++] = 2 - t0;
        }
        if (t1 < 15 && ctr < len) {
            t1 = t1 - (205 * t1 >> 10) * 5;
            p[ctr] = t1;
            a[ctr++] = 2 - t1;
        }
    }

    return ctr;
}


void poly_uniform_eta8x_with_pack(
        uint8_t *r,
        polyvecl *s1,
        polyveck *s2,
        const uint64_t seed[4]) {

    uint32_t ctr[8] = {0};
    keccakx8_state state;
    ALIGN(64) uint8_t buf[8][136];
    ALIGN(64) uint8_t p[8][256];

    state.s[0] = _mm512_set1_epi64(seed[0]);
    state.s[1] = _mm512_set1_epi64(seed[1]);
    state.s[2] = _mm512_set1_epi64(seed[2]);
    state.s[3] = _mm512_set1_epi64(seed[3]);
    state.s[4] = _mm512_set_epi64((0x1f << 16) ^ 7, (0x1f << 16) ^ 6, (0x1f << 16) ^ 5, (0x1f << 16) ^ 4,
                                  (0x1f << 16) ^ 3, (0x1f << 16) ^ 2, (0x1f << 16) ^ 1, (0x1f << 16) ^ 0);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm512_setzero_si512();

    state.s[16] = _mm512_set1_epi64(0x1ULL << 63);

    XURQ_AVX512_shake256x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 1);

    // rej_eta8x_with_pipe(s1, s2, ctr, p, buf);
    ctr[0] = rej_eta_with_pipe_avx512(&s1->vec[0], p[0], buf[0]);
    ctr[1] = rej_eta_with_pipe_avx512(&s1->vec[1], p[1], buf[1]);
    ctr[2] = rej_eta_with_pipe_avx512(&s1->vec[2], p[2], buf[2]);
    ctr[3] = rej_eta_with_pipe_avx512(&s1->vec[3], p[3], buf[3]);
    ctr[4] = rej_eta_with_pipe_avx512(&s1->vec[4], p[4], buf[4]);
    ctr[5] = rej_eta_with_pipe_avx512(&s1->vec[5], p[5], buf[5]);
    ctr[6] = rej_eta_with_pipe_avx512(&s1->vec[6], p[6], buf[6]);
    ctr[7] = rej_eta_with_pipe_avx512(&s2->vec[0], p[7], buf[7]);

    while (ctr[0] < N || ctr[1] < N || ctr[2] < N || ctr[3] < N || ctr[4] < N || ctr[5] < N || ctr[6] < N ||
           ctr[7] < N) {
        XURQ_AVX512_shake256x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 1);

        ctr[0] += rej_eta_with_pipe(s1->vec[0].coeffs, ctr[0], N, p[0], buf[0], SHAKE256_RATE);
        ctr[1] += rej_eta_with_pipe(s1->vec[1].coeffs, ctr[1], N, p[1], buf[1], SHAKE256_RATE);
        ctr[2] += rej_eta_with_pipe(s1->vec[2].coeffs, ctr[2], N, p[2], buf[2], SHAKE256_RATE);
        ctr[3] += rej_eta_with_pipe(s1->vec[3].coeffs, ctr[3], N, p[3], buf[3], SHAKE256_RATE);
        ctr[4] += rej_eta_with_pipe(s1->vec[4].coeffs, ctr[4], N, p[4], buf[4], SHAKE256_RATE);
        ctr[5] += rej_eta_with_pipe(s1->vec[5].coeffs, ctr[5], N, p[5], buf[5], SHAKE256_RATE);
        ctr[6] += rej_eta_with_pipe(s1->vec[6].coeffs, ctr[6], N, p[6], buf[6], SHAKE256_RATE);
        ctr[7] += rej_eta_with_pipe(s2->vec[0].coeffs, ctr[7], N, p[7], buf[7], SHAKE256_RATE);
    }

    eta_pack8x(r + 3 * SEEDBYTES + 0 * POLYETA_PACKEDBYTES, p[0]);
    eta_pack8x(r + 3 * SEEDBYTES + 1 * POLYETA_PACKEDBYTES, p[1]);
    eta_pack8x(r + 3 * SEEDBYTES + 2 * POLYETA_PACKEDBYTES, p[2]);
    eta_pack8x(r + 3 * SEEDBYTES + 3 * POLYETA_PACKEDBYTES, p[3]);
    eta_pack8x(r + 3 * SEEDBYTES + 4 * POLYETA_PACKEDBYTES, p[4]);
    eta_pack8x(r + 3 * SEEDBYTES + 5 * POLYETA_PACKEDBYTES, p[5]);
    eta_pack8x(r + 3 * SEEDBYTES + 6 * POLYETA_PACKEDBYTES, p[6]);
    eta_pack8x(r + 3 * SEEDBYTES + 7 * POLYETA_PACKEDBYTES, p[7]);


    state.s[0] = _mm512_set1_epi64(seed[0]);
    state.s[1] = _mm512_set1_epi64(seed[1]);
    state.s[2] = _mm512_set1_epi64(seed[2]);
    state.s[3] = _mm512_set1_epi64(seed[3]);
    state.s[4] = _mm512_set_epi64((0x1f << 16) ^ 15, (0x1f << 16) ^ 14, (0x1f << 16) ^ 13, (0x1f << 16) ^ 12,
                                  (0x1f << 16) ^ 11, (0x1f << 16) ^ 10, (0x1f << 16) ^ 9, (0x1f << 16) ^ 8);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm512_setzero_si512();

    state.s[16] = _mm512_set1_epi64(0x1ULL << 63);

    XURQ_AVX512_shake256x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 1);

    for (int i = 0; i < 8; ++i) {
        ctr[i] = 0;
    }

    // rej_eta7x_with_pipe(s2, ctr, p, buf);
    ctr[0] = rej_eta_with_pipe_avx512(&s2->vec[1], p[0], buf[0]);
    ctr[1] = rej_eta_with_pipe_avx512(&s2->vec[2], p[1], buf[1]);
    ctr[2] = rej_eta_with_pipe_avx512(&s2->vec[3], p[2], buf[2]);
    ctr[3] = rej_eta_with_pipe_avx512(&s2->vec[4], p[3], buf[3]);
    ctr[4] = rej_eta_with_pipe_avx512(&s2->vec[5], p[4], buf[4]);
    ctr[5] = rej_eta_with_pipe_avx512(&s2->vec[6], p[5], buf[5]);
    ctr[6] = rej_eta_with_pipe_avx512(&s2->vec[7], p[6], buf[6]);

    while (ctr[0] < N || ctr[1] < N || ctr[2] < N || ctr[3] < N || ctr[4] < N || ctr[5] < N || ctr[6] < N) {
        XURQ_AVX512_shake256x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 1);

        ctr[0] += rej_eta_with_pipe(s2->vec[1].coeffs, ctr[0], N, p[0], buf[0], SHAKE256_RATE);
        ctr[1] += rej_eta_with_pipe(s2->vec[2].coeffs, ctr[1], N, p[1], buf[1], SHAKE256_RATE);
        ctr[2] += rej_eta_with_pipe(s2->vec[3].coeffs, ctr[2], N, p[2], buf[2], SHAKE256_RATE);
        ctr[3] += rej_eta_with_pipe(s2->vec[4].coeffs, ctr[3], N, p[3], buf[3], SHAKE256_RATE);
        ctr[4] += rej_eta_with_pipe(s2->vec[5].coeffs, ctr[4], N, p[4], buf[4], SHAKE256_RATE);
        ctr[5] += rej_eta_with_pipe(s2->vec[6].coeffs, ctr[5], N, p[5], buf[5], SHAKE256_RATE);
        ctr[6] += rej_eta_with_pipe(s2->vec[7].coeffs, ctr[6], N, p[6], buf[6], SHAKE256_RATE);
    }

    eta_pack8x(r + 3 * SEEDBYTES + 8 * POLYETA_PACKEDBYTES, p[0]);
    eta_pack8x(r + 3 * SEEDBYTES + 9 * POLYETA_PACKEDBYTES, p[1]);
    eta_pack8x(r + 3 * SEEDBYTES + 10 * POLYETA_PACKEDBYTES, p[2]);
    eta_pack8x(r + 3 * SEEDBYTES + 11 * POLYETA_PACKEDBYTES, p[3]);
    eta_pack8x(r + 3 * SEEDBYTES + 12 * POLYETA_PACKEDBYTES, p[4]);
    eta_pack8x(r + 3 * SEEDBYTES + 13 * POLYETA_PACKEDBYTES, p[5]);
    eta_pack8x(r + 3 * SEEDBYTES + 14 * POLYETA_PACKEDBYTES, p[6]);

}



