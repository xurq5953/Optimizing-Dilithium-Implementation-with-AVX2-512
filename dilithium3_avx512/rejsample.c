#include "params.h"
#include "rejsample.h"
#include "symmetric.h"
#include "align.h"
#include "polyvec.h"
#include "fips202x8.h"
#include <immintrin.h>
#include <stdint.h>

const uint8_t PQCLEAN_DILITHIUM3_AVX2_idxlut[256][8] = {
    { 0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0},
    { 1,  0,  0,  0,  0,  0,  0,  0},
    { 0,  1,  0,  0,  0,  0,  0,  0},
    { 2,  0,  0,  0,  0,  0,  0,  0},
    { 0,  2,  0,  0,  0,  0,  0,  0},
    { 1,  2,  0,  0,  0,  0,  0,  0},
    { 0,  1,  2,  0,  0,  0,  0,  0},
    { 3,  0,  0,  0,  0,  0,  0,  0},
    { 0,  3,  0,  0,  0,  0,  0,  0},
    { 1,  3,  0,  0,  0,  0,  0,  0},
    { 0,  1,  3,  0,  0,  0,  0,  0},
    { 2,  3,  0,  0,  0,  0,  0,  0},
    { 0,  2,  3,  0,  0,  0,  0,  0},
    { 1,  2,  3,  0,  0,  0,  0,  0},
    { 0,  1,  2,  3,  0,  0,  0,  0},
    { 4,  0,  0,  0,  0,  0,  0,  0},
    { 0,  4,  0,  0,  0,  0,  0,  0},
    { 1,  4,  0,  0,  0,  0,  0,  0},
    { 0,  1,  4,  0,  0,  0,  0,  0},
    { 2,  4,  0,  0,  0,  0,  0,  0},
    { 0,  2,  4,  0,  0,  0,  0,  0},
    { 1,  2,  4,  0,  0,  0,  0,  0},
    { 0,  1,  2,  4,  0,  0,  0,  0},
    { 3,  4,  0,  0,  0,  0,  0,  0},
    { 0,  3,  4,  0,  0,  0,  0,  0},
    { 1,  3,  4,  0,  0,  0,  0,  0},
    { 0,  1,  3,  4,  0,  0,  0,  0},
    { 2,  3,  4,  0,  0,  0,  0,  0},
    { 0,  2,  3,  4,  0,  0,  0,  0},
    { 1,  2,  3,  4,  0,  0,  0,  0},
    { 0,  1,  2,  3,  4,  0,  0,  0},
    { 5,  0,  0,  0,  0,  0,  0,  0},
    { 0,  5,  0,  0,  0,  0,  0,  0},
    { 1,  5,  0,  0,  0,  0,  0,  0},
    { 0,  1,  5,  0,  0,  0,  0,  0},
    { 2,  5,  0,  0,  0,  0,  0,  0},
    { 0,  2,  5,  0,  0,  0,  0,  0},
    { 1,  2,  5,  0,  0,  0,  0,  0},
    { 0,  1,  2,  5,  0,  0,  0,  0},
    { 3,  5,  0,  0,  0,  0,  0,  0},
    { 0,  3,  5,  0,  0,  0,  0,  0},
    { 1,  3,  5,  0,  0,  0,  0,  0},
    { 0,  1,  3,  5,  0,  0,  0,  0},
    { 2,  3,  5,  0,  0,  0,  0,  0},
    { 0,  2,  3,  5,  0,  0,  0,  0},
    { 1,  2,  3,  5,  0,  0,  0,  0},
    { 0,  1,  2,  3,  5,  0,  0,  0},
    { 4,  5,  0,  0,  0,  0,  0,  0},
    { 0,  4,  5,  0,  0,  0,  0,  0},
    { 1,  4,  5,  0,  0,  0,  0,  0},
    { 0,  1,  4,  5,  0,  0,  0,  0},
    { 2,  4,  5,  0,  0,  0,  0,  0},
    { 0,  2,  4,  5,  0,  0,  0,  0},
    { 1,  2,  4,  5,  0,  0,  0,  0},
    { 0,  1,  2,  4,  5,  0,  0,  0},
    { 3,  4,  5,  0,  0,  0,  0,  0},
    { 0,  3,  4,  5,  0,  0,  0,  0},
    { 1,  3,  4,  5,  0,  0,  0,  0},
    { 0,  1,  3,  4,  5,  0,  0,  0},
    { 2,  3,  4,  5,  0,  0,  0,  0},
    { 0,  2,  3,  4,  5,  0,  0,  0},
    { 1,  2,  3,  4,  5,  0,  0,  0},
    { 0,  1,  2,  3,  4,  5,  0,  0},
    { 6,  0,  0,  0,  0,  0,  0,  0},
    { 0,  6,  0,  0,  0,  0,  0,  0},
    { 1,  6,  0,  0,  0,  0,  0,  0},
    { 0,  1,  6,  0,  0,  0,  0,  0},
    { 2,  6,  0,  0,  0,  0,  0,  0},
    { 0,  2,  6,  0,  0,  0,  0,  0},
    { 1,  2,  6,  0,  0,  0,  0,  0},
    { 0,  1,  2,  6,  0,  0,  0,  0},
    { 3,  6,  0,  0,  0,  0,  0,  0},
    { 0,  3,  6,  0,  0,  0,  0,  0},
    { 1,  3,  6,  0,  0,  0,  0,  0},
    { 0,  1,  3,  6,  0,  0,  0,  0},
    { 2,  3,  6,  0,  0,  0,  0,  0},
    { 0,  2,  3,  6,  0,  0,  0,  0},
    { 1,  2,  3,  6,  0,  0,  0,  0},
    { 0,  1,  2,  3,  6,  0,  0,  0},
    { 4,  6,  0,  0,  0,  0,  0,  0},
    { 0,  4,  6,  0,  0,  0,  0,  0},
    { 1,  4,  6,  0,  0,  0,  0,  0},
    { 0,  1,  4,  6,  0,  0,  0,  0},
    { 2,  4,  6,  0,  0,  0,  0,  0},
    { 0,  2,  4,  6,  0,  0,  0,  0},
    { 1,  2,  4,  6,  0,  0,  0,  0},
    { 0,  1,  2,  4,  6,  0,  0,  0},
    { 3,  4,  6,  0,  0,  0,  0,  0},
    { 0,  3,  4,  6,  0,  0,  0,  0},
    { 1,  3,  4,  6,  0,  0,  0,  0},
    { 0,  1,  3,  4,  6,  0,  0,  0},
    { 2,  3,  4,  6,  0,  0,  0,  0},
    { 0,  2,  3,  4,  6,  0,  0,  0},
    { 1,  2,  3,  4,  6,  0,  0,  0},
    { 0,  1,  2,  3,  4,  6,  0,  0},
    { 5,  6,  0,  0,  0,  0,  0,  0},
    { 0,  5,  6,  0,  0,  0,  0,  0},
    { 1,  5,  6,  0,  0,  0,  0,  0},
    { 0,  1,  5,  6,  0,  0,  0,  0},
    { 2,  5,  6,  0,  0,  0,  0,  0},
    { 0,  2,  5,  6,  0,  0,  0,  0},
    { 1,  2,  5,  6,  0,  0,  0,  0},
    { 0,  1,  2,  5,  6,  0,  0,  0},
    { 3,  5,  6,  0,  0,  0,  0,  0},
    { 0,  3,  5,  6,  0,  0,  0,  0},
    { 1,  3,  5,  6,  0,  0,  0,  0},
    { 0,  1,  3,  5,  6,  0,  0,  0},
    { 2,  3,  5,  6,  0,  0,  0,  0},
    { 0,  2,  3,  5,  6,  0,  0,  0},
    { 1,  2,  3,  5,  6,  0,  0,  0},
    { 0,  1,  2,  3,  5,  6,  0,  0},
    { 4,  5,  6,  0,  0,  0,  0,  0},
    { 0,  4,  5,  6,  0,  0,  0,  0},
    { 1,  4,  5,  6,  0,  0,  0,  0},
    { 0,  1,  4,  5,  6,  0,  0,  0},
    { 2,  4,  5,  6,  0,  0,  0,  0},
    { 0,  2,  4,  5,  6,  0,  0,  0},
    { 1,  2,  4,  5,  6,  0,  0,  0},
    { 0,  1,  2,  4,  5,  6,  0,  0},
    { 3,  4,  5,  6,  0,  0,  0,  0},
    { 0,  3,  4,  5,  6,  0,  0,  0},
    { 1,  3,  4,  5,  6,  0,  0,  0},
    { 0,  1,  3,  4,  5,  6,  0,  0},
    { 2,  3,  4,  5,  6,  0,  0,  0},
    { 0,  2,  3,  4,  5,  6,  0,  0},
    { 1,  2,  3,  4,  5,  6,  0,  0},
    { 0,  1,  2,  3,  4,  5,  6,  0},
    { 7,  0,  0,  0,  0,  0,  0,  0},
    { 0,  7,  0,  0,  0,  0,  0,  0},
    { 1,  7,  0,  0,  0,  0,  0,  0},
    { 0,  1,  7,  0,  0,  0,  0,  0},
    { 2,  7,  0,  0,  0,  0,  0,  0},
    { 0,  2,  7,  0,  0,  0,  0,  0},
    { 1,  2,  7,  0,  0,  0,  0,  0},
    { 0,  1,  2,  7,  0,  0,  0,  0},
    { 3,  7,  0,  0,  0,  0,  0,  0},
    { 0,  3,  7,  0,  0,  0,  0,  0},
    { 1,  3,  7,  0,  0,  0,  0,  0},
    { 0,  1,  3,  7,  0,  0,  0,  0},
    { 2,  3,  7,  0,  0,  0,  0,  0},
    { 0,  2,  3,  7,  0,  0,  0,  0},
    { 1,  2,  3,  7,  0,  0,  0,  0},
    { 0,  1,  2,  3,  7,  0,  0,  0},
    { 4,  7,  0,  0,  0,  0,  0,  0},
    { 0,  4,  7,  0,  0,  0,  0,  0},
    { 1,  4,  7,  0,  0,  0,  0,  0},
    { 0,  1,  4,  7,  0,  0,  0,  0},
    { 2,  4,  7,  0,  0,  0,  0,  0},
    { 0,  2,  4,  7,  0,  0,  0,  0},
    { 1,  2,  4,  7,  0,  0,  0,  0},
    { 0,  1,  2,  4,  7,  0,  0,  0},
    { 3,  4,  7,  0,  0,  0,  0,  0},
    { 0,  3,  4,  7,  0,  0,  0,  0},
    { 1,  3,  4,  7,  0,  0,  0,  0},
    { 0,  1,  3,  4,  7,  0,  0,  0},
    { 2,  3,  4,  7,  0,  0,  0,  0},
    { 0,  2,  3,  4,  7,  0,  0,  0},
    { 1,  2,  3,  4,  7,  0,  0,  0},
    { 0,  1,  2,  3,  4,  7,  0,  0},
    { 5,  7,  0,  0,  0,  0,  0,  0},
    { 0,  5,  7,  0,  0,  0,  0,  0},
    { 1,  5,  7,  0,  0,  0,  0,  0},
    { 0,  1,  5,  7,  0,  0,  0,  0},
    { 2,  5,  7,  0,  0,  0,  0,  0},
    { 0,  2,  5,  7,  0,  0,  0,  0},
    { 1,  2,  5,  7,  0,  0,  0,  0},
    { 0,  1,  2,  5,  7,  0,  0,  0},
    { 3,  5,  7,  0,  0,  0,  0,  0},
    { 0,  3,  5,  7,  0,  0,  0,  0},
    { 1,  3,  5,  7,  0,  0,  0,  0},
    { 0,  1,  3,  5,  7,  0,  0,  0},
    { 2,  3,  5,  7,  0,  0,  0,  0},
    { 0,  2,  3,  5,  7,  0,  0,  0},
    { 1,  2,  3,  5,  7,  0,  0,  0},
    { 0,  1,  2,  3,  5,  7,  0,  0},
    { 4,  5,  7,  0,  0,  0,  0,  0},
    { 0,  4,  5,  7,  0,  0,  0,  0},
    { 1,  4,  5,  7,  0,  0,  0,  0},
    { 0,  1,  4,  5,  7,  0,  0,  0},
    { 2,  4,  5,  7,  0,  0,  0,  0},
    { 0,  2,  4,  5,  7,  0,  0,  0},
    { 1,  2,  4,  5,  7,  0,  0,  0},
    { 0,  1,  2,  4,  5,  7,  0,  0},
    { 3,  4,  5,  7,  0,  0,  0,  0},
    { 0,  3,  4,  5,  7,  0,  0,  0},
    { 1,  3,  4,  5,  7,  0,  0,  0},
    { 0,  1,  3,  4,  5,  7,  0,  0},
    { 2,  3,  4,  5,  7,  0,  0,  0},
    { 0,  2,  3,  4,  5,  7,  0,  0},
    { 1,  2,  3,  4,  5,  7,  0,  0},
    { 0,  1,  2,  3,  4,  5,  7,  0},
    { 6,  7,  0,  0,  0,  0,  0,  0},
    { 0,  6,  7,  0,  0,  0,  0,  0},
    { 1,  6,  7,  0,  0,  0,  0,  0},
    { 0,  1,  6,  7,  0,  0,  0,  0},
    { 2,  6,  7,  0,  0,  0,  0,  0},
    { 0,  2,  6,  7,  0,  0,  0,  0},
    { 1,  2,  6,  7,  0,  0,  0,  0},
    { 0,  1,  2,  6,  7,  0,  0,  0},
    { 3,  6,  7,  0,  0,  0,  0,  0},
    { 0,  3,  6,  7,  0,  0,  0,  0},
    { 1,  3,  6,  7,  0,  0,  0,  0},
    { 0,  1,  3,  6,  7,  0,  0,  0},
    { 2,  3,  6,  7,  0,  0,  0,  0},
    { 0,  2,  3,  6,  7,  0,  0,  0},
    { 1,  2,  3,  6,  7,  0,  0,  0},
    { 0,  1,  2,  3,  6,  7,  0,  0},
    { 4,  6,  7,  0,  0,  0,  0,  0},
    { 0,  4,  6,  7,  0,  0,  0,  0},
    { 1,  4,  6,  7,  0,  0,  0,  0},
    { 0,  1,  4,  6,  7,  0,  0,  0},
    { 2,  4,  6,  7,  0,  0,  0,  0},
    { 0,  2,  4,  6,  7,  0,  0,  0},
    { 1,  2,  4,  6,  7,  0,  0,  0},
    { 0,  1,  2,  4,  6,  7,  0,  0},
    { 3,  4,  6,  7,  0,  0,  0,  0},
    { 0,  3,  4,  6,  7,  0,  0,  0},
    { 1,  3,  4,  6,  7,  0,  0,  0},
    { 0,  1,  3,  4,  6,  7,  0,  0},
    { 2,  3,  4,  6,  7,  0,  0,  0},
    { 0,  2,  3,  4,  6,  7,  0,  0},
    { 1,  2,  3,  4,  6,  7,  0,  0},
    { 0,  1,  2,  3,  4,  6,  7,  0},
    { 5,  6,  7,  0,  0,  0,  0,  0},
    { 0,  5,  6,  7,  0,  0,  0,  0},
    { 1,  5,  6,  7,  0,  0,  0,  0},
    { 0,  1,  5,  6,  7,  0,  0,  0},
    { 2,  5,  6,  7,  0,  0,  0,  0},
    { 0,  2,  5,  6,  7,  0,  0,  0},
    { 1,  2,  5,  6,  7,  0,  0,  0},
    { 0,  1,  2,  5,  6,  7,  0,  0},
    { 3,  5,  6,  7,  0,  0,  0,  0},
    { 0,  3,  5,  6,  7,  0,  0,  0},
    { 1,  3,  5,  6,  7,  0,  0,  0},
    { 0,  1,  3,  5,  6,  7,  0,  0},
    { 2,  3,  5,  6,  7,  0,  0,  0},
    { 0,  2,  3,  5,  6,  7,  0,  0},
    { 1,  2,  3,  5,  6,  7,  0,  0},
    { 0,  1,  2,  3,  5,  6,  7,  0},
    { 4,  5,  6,  7,  0,  0,  0,  0},
    { 0,  4,  5,  6,  7,  0,  0,  0},
    { 1,  4,  5,  6,  7,  0,  0,  0},
    { 0,  1,  4,  5,  6,  7,  0,  0},
    { 2,  4,  5,  6,  7,  0,  0,  0},
    { 0,  2,  4,  5,  6,  7,  0,  0},
    { 1,  2,  4,  5,  6,  7,  0,  0},
    { 0,  1,  2,  4,  5,  6,  7,  0},
    { 3,  4,  5,  6,  7,  0,  0,  0},
    { 0,  3,  4,  5,  6,  7,  0,  0},
    { 1,  3,  4,  5,  6,  7,  0,  0},
    { 0,  1,  3,  4,  5,  6,  7,  0},
    { 2,  3,  4,  5,  6,  7,  0,  0},
    { 0,  2,  3,  4,  5,  6,  7,  0},
    { 1,  2,  3,  4,  5,  6,  7,  0},
    { 0,  1,  2,  3,  4,  5,  6,  7}
};

unsigned int PQCLEAN_DILITHIUM3_AVX2_rej_uniform_avx(int32_t *restrict r, const uint8_t buf[REJ_UNIFORM_BUFLEN + 8]) {
    unsigned int ctr, pos;
    uint32_t good;
    __m256i d, tmp;
    const __m256i bound = _mm256_set1_epi32(Q);
    const __m256i mask  = _mm256_set1_epi32(0x7FFFFF);
    const __m256i idx8  = _mm256_set_epi8(-1, 15, 14, 13, -1, 12, 11, 10,
                                          -1, 9, 8, 7, -1, 6, 5, 4,
                                          -1, 11, 10, 9, -1, 8, 7, 6,
                                          -1, 5, 4, 3, -1, 2, 1, 0);

    ctr = pos = 0;
    while (pos <= REJ_UNIFORM_BUFLEN - 24) {
        d = _mm256_loadu_si256((__m256i *)&buf[pos]);
        d = _mm256_permute4x64_epi64(d, 0x94);
        d = _mm256_shuffle_epi8(d, idx8);
        d = _mm256_and_si256(d, mask);
        pos += 24;

        tmp = _mm256_sub_epi32(d, bound);
        good = _mm256_movemask_ps((__m256)tmp);
        tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)&PQCLEAN_DILITHIUM3_AVX2_idxlut[good]));
        d = _mm256_permutevar8x32_epi32(d, tmp);

        _mm256_storeu_si256((__m256i *)&r[ctr], d);
        ctr += _mm_popcnt_u32(good);

        if (ctr > N - 8) {
            break;
        }
    }

    uint32_t t;
    while (ctr < N && pos <= REJ_UNIFORM_BUFLEN - 3) {
        t  = buf[pos++];
        t |= (uint32_t)buf[pos++] << 8;
        t |= (uint32_t)buf[pos++] << 16;
        t &= 0x7FFFFF;

        if (t < Q) {
            r[ctr++] = t;
        }
    }

    return ctr;
}

unsigned int PQCLEAN_DILITHIUM3_AVX2_rej_eta_avx(int32_t *restrict r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]) {
    unsigned int ctr, pos;
    uint32_t good;
    __m256i f0, f1;
    __m128i g0, g1;
    const __m256i mask = _mm256_set1_epi8(15);
    const __m256i eta = _mm256_set1_epi8(4);
    const __m256i bound = _mm256_set1_epi8(9);

    ctr = pos = 0;
    while (ctr <= N - 8 && pos <= REJ_UNIFORM_ETA_BUFLEN - 16) {
        f0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)&buf[pos]));
        f1 = _mm256_slli_epi16(f0, 4);
        f0 = _mm256_or_si256(f0, f1);
        f0 = _mm256_and_si256(f0, mask);

        f1 = _mm256_sub_epi8(f0, bound);
        f0 = _mm256_sub_epi8(eta, f0);
        good = _mm256_movemask_epi8(f1);

        g0 = _mm256_castsi256_si128(f0);
        g1 = _mm_loadl_epi64((__m128i *)&PQCLEAN_DILITHIUM3_AVX2_idxlut[good & 0xFF]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        _mm256_storeu_si256((__m256i *)&r[ctr], f1);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) {
            break;
        }
        g0 = _mm_bsrli_si128(g0, 8);
        g1 = _mm_loadl_epi64((__m128i *)&PQCLEAN_DILITHIUM3_AVX2_idxlut[good & 0xFF]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        _mm256_storeu_si256((__m256i *)&r[ctr], f1);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) {
            break;
        }
        g0 = _mm256_extracti128_si256(f0, 1);
        g1 = _mm_loadl_epi64((__m128i *)&PQCLEAN_DILITHIUM3_AVX2_idxlut[good & 0xFF]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        _mm256_storeu_si256((__m256i *)&r[ctr], f1);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) {
            break;
        }
        g0 = _mm_bsrli_si128(g0, 8);
        g1 = _mm_loadl_epi64((__m128i *)&PQCLEAN_DILITHIUM3_AVX2_idxlut[good]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        _mm256_storeu_si256((__m256i *)&r[ctr], f1);
        ctr += _mm_popcnt_u32(good);
        pos += 4;
    }

    uint32_t t0, t1;
    while (ctr < N && pos < REJ_UNIFORM_ETA_BUFLEN) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

        if (t0 < 9) {
            r[ctr++] = 4 - t0;
        }
        if (t1 < 9 && ctr < N) {
            r[ctr++] = 4 - t1;
        }
    }

    return ctr;
}


unsigned int XURQ_AVX2_rej_uniform_avx_s1s3(int32_t *restrict r, const uint8_t *buf, unsigned int num) {
    unsigned int ctr, pos;
    uint32_t good;
    __m256i d, tmp;
    const __m256i bound = _mm256_set1_epi32(Q);
    const __m256i mask = _mm256_set1_epi32(0x7FFFFF);
    const __m256i idx8 = _mm256_set_epi8(-1, 15, 14, 13, -1, 12, 11, 10,
                                         -1, 9, 8, 7, -1, 6, 5, 4,
                                         -1, 11, 10, 9, -1, 8, 7, 6,
                                         -1, 5, 4, 3, -1, 2, 1, 0);

    ctr = num;
    pos = 0;
    for (int i = 0; i < 7; ++i) {
        d = _mm256_loadu_si256((__m256i *) &buf[pos]);
        d = _mm256_permute4x64_epi64(d, 0x94);
        d = _mm256_shuffle_epi8(d, idx8);
        d = _mm256_and_si256(d, mask);

        tmp = _mm256_sub_epi32(d, bound);
        good = _mm256_movemask_ps((__m256) tmp);
        if (good == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], d);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &PQCLEAN_DILITHIUM3_AVX2_idxlut[good]));
            d = _mm256_permutevar8x32_epi32(d, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], d);
            ctr += _mm_popcnt_u32(good);
        }
    }

    return ctr;
}

unsigned int XURQ_AVX2_rej_uniform_avx_s1s3_final(int32_t *restrict r, const uint8_t *buf, unsigned int num) {
    unsigned int ctr, pos;
    uint32_t good;
    __m256i d, tmp;
    const __m256i bound = _mm256_set1_epi32(Q);
    const __m256i mask = _mm256_set1_epi32(0x7FFFFF);
    const __m256i idx8 = _mm256_set_epi8(-1, 15, 14, 13, -1, 12, 11, 10,
                                         -1, 9, 8, 7, -1, 6, 5, 4,
                                         -1, 11, 10, 9, -1, 8, 7, 6,
                                         -1, 5, 4, 3, -1, 2, 1, 0);

    ctr = num;

    pos = 0;

    for (int i = 0; i < 3; ++i) {
        d = _mm256_loadu_si256((__m256i *) &buf[pos]);
        d = _mm256_permute4x64_epi64(d, 0x94);
        d = _mm256_shuffle_epi8(d, idx8);
        d = _mm256_and_si256(d, mask);
        pos += 24;

        tmp = _mm256_sub_epi32(d, bound);
        good = _mm256_movemask_ps((__m256) tmp);
        if (good == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], d);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &PQCLEAN_DILITHIUM3_AVX2_idxlut[good]));
            d = _mm256_permutevar8x32_epi32(d, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], d);
            ctr += _mm_popcnt_u32(good);
        }
    }

    pos = 72;

    while (ctr < 248 && pos <= 144) {
        d = _mm256_loadu_si256((__m256i *) &buf[pos]);
        d = _mm256_permute4x64_epi64(d, 0x94);
        d = _mm256_shuffle_epi8(d, idx8);
        d = _mm256_and_si256(d, mask);
        pos += 24;

        tmp = _mm256_sub_epi32(d, bound);
        good = _mm256_movemask_ps((__m256) tmp);
        if (good == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], d);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &PQCLEAN_DILITHIUM3_AVX2_idxlut[good]));
            d = _mm256_permutevar8x32_epi32(d, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], d);
            ctr += _mm_popcnt_u32(good);
        }
    }

    uint32_t t;
    while (ctr < N && pos <= 168) {
        t = buf[pos++];
        t |= (uint32_t) buf[pos++] << 8;
        t |= (uint32_t) buf[pos++] << 16;
        t &= 0x7FFFFF;

        if (t < Q) {
            r[ctr++] = t;
        }
    }

    return ctr;
}

unsigned int rej_eta_avx512(int32_t * restrict r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]) {
    unsigned int ctr, pos;
    __mmask16 good0, good1;
    __m512i f0, f1, f2, f3;
    const __m512i mask = _mm512_set1_epi32(15);
    const __m512i eta = _mm512_set1_epi32(ETA);
    const __m512i bound = _mm512_set1_epi32(9);
    const __m512i idx16  = _mm512_set_epi32(15, 7, 14, 6,13, 5,12, 4,
                                            11, 3, 10, 2, 9, 1, 8, 0);
    ctr = pos = 0;
    while(ctr <= N - 32 && pos <= REJ_UNIFORM_ETA_BUFLEN - 16) {
        f0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i *)&buf[pos]));
        f1 = _mm512_srai_epi32(f0,4);
        f0 = _mm512_and_si512(f0,mask);
        // to get the right order
        f2 = _mm512_shuffle_i32x4(f0, f1, 0x44);
        f3 = _mm512_shuffle_i32x4(f0, f1, 0xEE);
        f0 = _mm512_permutexvar_epi32(idx16, f2);
        f1 = _mm512_permutexvar_epi32(idx16, f3);
        good0 = _mm512_cmp_epi32_mask(f0, bound, 1);
        good1 = _mm512_cmp_epi32_mask(f1, bound, 1);
        //store
        f0 = _mm512_sub_epi32(eta,f0);
        _mm512_mask_compressstoreu_epi32(&r[ctr], good0, f0);
        ctr += _mm_popcnt_u32((int32_t)good0);

        f1 = _mm512_sub_epi32(eta,f1);
        _mm512_mask_compressstoreu_epi32(&r[ctr], good1, f1);
        ctr += _mm_popcnt_u32((int32_t)good1);
        pos += 16;
        if(ctr > N - 32) break;
    }
    uint32_t t0, t1;
    while(ctr < N && pos < REJ_UNIFORM_ETA_BUFLEN) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

        if(t0 < 9)
            r[ctr++] = 4 - t0;
        if(t1 < 9 && ctr < N)
            r[ctr++] = 4 - t1;
    }

    return ctr;
}

unsigned int XURQ_AVX2_rej_eta_avx_with_pack(int32_t *restrict r, uint8_t *pipe, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]) {
    unsigned int ctr, pos;
    uint32_t good;
    __m256i f0, f1;
    __m128i g0, g1;
    __m128i d0, d1, d2, d3;
    const __m256i mask = _mm256_set1_epi8(15);
    const __m256i eta = _mm256_set1_epi8(4);
    const __m128i etas = _mm_set1_epi8(ETA);
    const __m256i bound = _mm256_set1_epi8(9);

    ctr = pos = 0;
    while (ctr <= N - 8 && pos <= REJ_UNIFORM_ETA_BUFLEN - 16) {
        f0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)&buf[pos]));
        f1 = _mm256_slli_epi16(f0, 4);
        f0 = _mm256_or_si256(f0, f1);
        f0 = _mm256_and_si256(f0, mask);

        f1 = _mm256_sub_epi8(f0, bound);
        f0 = _mm256_sub_epi8(eta, f0);
        good = _mm256_movemask_epi8(f1);

        g0 = _mm256_castsi256_si128(f0);
        g1 = _mm_loadl_epi64((__m128i *)&PQCLEAN_DILITHIUM3_AVX2_idxlut[good & 0xFF]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);

        _mm256_storeu_si256((__m256i *)&r[ctr], f1);
        _mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(etas, g1));
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) {
            break;
        }
        g0 = _mm_bsrli_si128(g0, 8);
        g1 = _mm_loadl_epi64((__m128i *)&PQCLEAN_DILITHIUM3_AVX2_idxlut[good & 0xFF]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        _mm256_storeu_si256((__m256i *)&r[ctr], f1);
        _mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(etas, g1));
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) {
            break;
        }
        g0 = _mm256_extracti128_si256(f0, 1);
        g1 = _mm_loadl_epi64((__m128i *)&PQCLEAN_DILITHIUM3_AVX2_idxlut[good & 0xFF]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        _mm256_storeu_si256((__m256i *)&r[ctr], f1);
        _mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(etas, g1));
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) {
            break;
        }
        g0 = _mm_bsrli_si128(g0, 8);
        g1 = _mm_loadl_epi64((__m128i *)&PQCLEAN_DILITHIUM3_AVX2_idxlut[good]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        _mm256_storeu_si256((__m256i *)&r[ctr], f1);
        _mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(etas, g1));
        ctr += _mm_popcnt_u32(good);
        pos += 4;
    }

    uint32_t t0, t1;
    while (ctr < N && pos < REJ_UNIFORM_ETA_BUFLEN) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

        if (t0 < 9) {
            r[ctr] = 4 - t0;
            pipe[ctr] = t0;
            ctr++;
        }
        if (t1 < 9 && ctr < N) {
            r[ctr] = 4 - t1;
            pipe[ctr] = t1;
            ctr++;
        }
    }

    return ctr;
}

ALIGN(64) static const uint8_t idx_u[64] = {
        0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80,
        12, 13, 14, 0x80, 15, 16, 17, 0x80, 18, 19, 20, 0x80, 21, 22, 23, 0x80,
        24, 25, 26, 0x80, 27, 28, 29, 0x80, 30, 31, 32, 0x80, 33, 34, 35, 0x80,
        36, 37, 38, 0x80, 39, 40, 41, 0x80, 42, 43, 44, 0x80, 45, 46, 47, 0x80,
};

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

void XURQ_D3_AVX512_polyz_unpack(poly *r, const uint8_t *buf) {
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

void XURQ_poly_uniform_gamma1(polyvecl *z,
                              const uint8_t seed[64],
                              uint16_t nonce) {
    ALIGN(64) uint8_t buf[8][840];
    XURQ_AVX512_precompute_gamma1_8x(buf, (uint64_t *) seed,
                                     nonce, nonce + 1, nonce + 2, nonce + 3,
                                     nonce + 4, nonce + 5, nonce + 6, 0);

    XURQ_D3_AVX512_polyz_unpack(&z->vec[0], buf[0]);
    XURQ_D3_AVX512_polyz_unpack(&z->vec[1], buf[1]);
    XURQ_D3_AVX512_polyz_unpack(&z->vec[2], buf[2]);
    XURQ_D3_AVX512_polyz_unpack(&z->vec[3], buf[3]);
    XURQ_D3_AVX512_polyz_unpack(&z->vec[4], buf[4]);

}