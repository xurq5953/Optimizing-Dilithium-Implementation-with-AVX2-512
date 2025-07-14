#include "params.h"
#include "rejsample.h"
#include "align.h"
#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>
#include <stdio.h>

const uint8_t DILITHIUM2_AVX2_idxlut[256][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
        {2, 0, 0, 0, 0, 0, 0, 0},
        {0, 2, 0, 0, 0, 0, 0, 0},
        {1, 2, 0, 0, 0, 0, 0, 0},
        {0, 1, 2, 0, 0, 0, 0, 0},
        {3, 0, 0, 0, 0, 0, 0, 0},
        {0, 3, 0, 0, 0, 0, 0, 0},
        {1, 3, 0, 0, 0, 0, 0, 0},
        {0, 1, 3, 0, 0, 0, 0, 0},
        {2, 3, 0, 0, 0, 0, 0, 0},
        {0, 2, 3, 0, 0, 0, 0, 0},
        {1, 2, 3, 0, 0, 0, 0, 0},
        {0, 1, 2, 3, 0, 0, 0, 0},
        {4, 0, 0, 0, 0, 0, 0, 0},
        {0, 4, 0, 0, 0, 0, 0, 0},
        {1, 4, 0, 0, 0, 0, 0, 0},
        {0, 1, 4, 0, 0, 0, 0, 0},
        {2, 4, 0, 0, 0, 0, 0, 0},
        {0, 2, 4, 0, 0, 0, 0, 0},
        {1, 2, 4, 0, 0, 0, 0, 0},
        {0, 1, 2, 4, 0, 0, 0, 0},
        {3, 4, 0, 0, 0, 0, 0, 0},
        {0, 3, 4, 0, 0, 0, 0, 0},
        {1, 3, 4, 0, 0, 0, 0, 0},
        {0, 1, 3, 4, 0, 0, 0, 0},
        {2, 3, 4, 0, 0, 0, 0, 0},
        {0, 2, 3, 4, 0, 0, 0, 0},
        {1, 2, 3, 4, 0, 0, 0, 0},
        {0, 1, 2, 3, 4, 0, 0, 0},
        {5, 0, 0, 0, 0, 0, 0, 0},
        {0, 5, 0, 0, 0, 0, 0, 0},
        {1, 5, 0, 0, 0, 0, 0, 0},
        {0, 1, 5, 0, 0, 0, 0, 0},
        {2, 5, 0, 0, 0, 0, 0, 0},
        {0, 2, 5, 0, 0, 0, 0, 0},
        {1, 2, 5, 0, 0, 0, 0, 0},
        {0, 1, 2, 5, 0, 0, 0, 0},
        {3, 5, 0, 0, 0, 0, 0, 0},
        {0, 3, 5, 0, 0, 0, 0, 0},
        {1, 3, 5, 0, 0, 0, 0, 0},
        {0, 1, 3, 5, 0, 0, 0, 0},
        {2, 3, 5, 0, 0, 0, 0, 0},
        {0, 2, 3, 5, 0, 0, 0, 0},
        {1, 2, 3, 5, 0, 0, 0, 0},
        {0, 1, 2, 3, 5, 0, 0, 0},
        {4, 5, 0, 0, 0, 0, 0, 0},
        {0, 4, 5, 0, 0, 0, 0, 0},
        {1, 4, 5, 0, 0, 0, 0, 0},
        {0, 1, 4, 5, 0, 0, 0, 0},
        {2, 4, 5, 0, 0, 0, 0, 0},
        {0, 2, 4, 5, 0, 0, 0, 0},
        {1, 2, 4, 5, 0, 0, 0, 0},
        {0, 1, 2, 4, 5, 0, 0, 0},
        {3, 4, 5, 0, 0, 0, 0, 0},
        {0, 3, 4, 5, 0, 0, 0, 0},
        {1, 3, 4, 5, 0, 0, 0, 0},
        {0, 1, 3, 4, 5, 0, 0, 0},
        {2, 3, 4, 5, 0, 0, 0, 0},
        {0, 2, 3, 4, 5, 0, 0, 0},
        {1, 2, 3, 4, 5, 0, 0, 0},
        {0, 1, 2, 3, 4, 5, 0, 0},
        {6, 0, 0, 0, 0, 0, 0, 0},
        {0, 6, 0, 0, 0, 0, 0, 0},
        {1, 6, 0, 0, 0, 0, 0, 0},
        {0, 1, 6, 0, 0, 0, 0, 0},
        {2, 6, 0, 0, 0, 0, 0, 0},
        {0, 2, 6, 0, 0, 0, 0, 0},
        {1, 2, 6, 0, 0, 0, 0, 0},
        {0, 1, 2, 6, 0, 0, 0, 0},
        {3, 6, 0, 0, 0, 0, 0, 0},
        {0, 3, 6, 0, 0, 0, 0, 0},
        {1, 3, 6, 0, 0, 0, 0, 0},
        {0, 1, 3, 6, 0, 0, 0, 0},
        {2, 3, 6, 0, 0, 0, 0, 0},
        {0, 2, 3, 6, 0, 0, 0, 0},
        {1, 2, 3, 6, 0, 0, 0, 0},
        {0, 1, 2, 3, 6, 0, 0, 0},
        {4, 6, 0, 0, 0, 0, 0, 0},
        {0, 4, 6, 0, 0, 0, 0, 0},
        {1, 4, 6, 0, 0, 0, 0, 0},
        {0, 1, 4, 6, 0, 0, 0, 0},
        {2, 4, 6, 0, 0, 0, 0, 0},
        {0, 2, 4, 6, 0, 0, 0, 0},
        {1, 2, 4, 6, 0, 0, 0, 0},
        {0, 1, 2, 4, 6, 0, 0, 0},
        {3, 4, 6, 0, 0, 0, 0, 0},
        {0, 3, 4, 6, 0, 0, 0, 0},
        {1, 3, 4, 6, 0, 0, 0, 0},
        {0, 1, 3, 4, 6, 0, 0, 0},
        {2, 3, 4, 6, 0, 0, 0, 0},
        {0, 2, 3, 4, 6, 0, 0, 0},
        {1, 2, 3, 4, 6, 0, 0, 0},
        {0, 1, 2, 3, 4, 6, 0, 0},
        {5, 6, 0, 0, 0, 0, 0, 0},
        {0, 5, 6, 0, 0, 0, 0, 0},
        {1, 5, 6, 0, 0, 0, 0, 0},
        {0, 1, 5, 6, 0, 0, 0, 0},
        {2, 5, 6, 0, 0, 0, 0, 0},
        {0, 2, 5, 6, 0, 0, 0, 0},
        {1, 2, 5, 6, 0, 0, 0, 0},
        {0, 1, 2, 5, 6, 0, 0, 0},
        {3, 5, 6, 0, 0, 0, 0, 0},
        {0, 3, 5, 6, 0, 0, 0, 0},
        {1, 3, 5, 6, 0, 0, 0, 0},
        {0, 1, 3, 5, 6, 0, 0, 0},
        {2, 3, 5, 6, 0, 0, 0, 0},
        {0, 2, 3, 5, 6, 0, 0, 0},
        {1, 2, 3, 5, 6, 0, 0, 0},
        {0, 1, 2, 3, 5, 6, 0, 0},
        {4, 5, 6, 0, 0, 0, 0, 0},
        {0, 4, 5, 6, 0, 0, 0, 0},
        {1, 4, 5, 6, 0, 0, 0, 0},
        {0, 1, 4, 5, 6, 0, 0, 0},
        {2, 4, 5, 6, 0, 0, 0, 0},
        {0, 2, 4, 5, 6, 0, 0, 0},
        {1, 2, 4, 5, 6, 0, 0, 0},
        {0, 1, 2, 4, 5, 6, 0, 0},
        {3, 4, 5, 6, 0, 0, 0, 0},
        {0, 3, 4, 5, 6, 0, 0, 0},
        {1, 3, 4, 5, 6, 0, 0, 0},
        {0, 1, 3, 4, 5, 6, 0, 0},
        {2, 3, 4, 5, 6, 0, 0, 0},
        {0, 2, 3, 4, 5, 6, 0, 0},
        {1, 2, 3, 4, 5, 6, 0, 0},
        {0, 1, 2, 3, 4, 5, 6, 0},
        {7, 0, 0, 0, 0, 0, 0, 0},
        {0, 7, 0, 0, 0, 0, 0, 0},
        {1, 7, 0, 0, 0, 0, 0, 0},
        {0, 1, 7, 0, 0, 0, 0, 0},
        {2, 7, 0, 0, 0, 0, 0, 0},
        {0, 2, 7, 0, 0, 0, 0, 0},
        {1, 2, 7, 0, 0, 0, 0, 0},
        {0, 1, 2, 7, 0, 0, 0, 0},
        {3, 7, 0, 0, 0, 0, 0, 0},
        {0, 3, 7, 0, 0, 0, 0, 0},
        {1, 3, 7, 0, 0, 0, 0, 0},
        {0, 1, 3, 7, 0, 0, 0, 0},
        {2, 3, 7, 0, 0, 0, 0, 0},
        {0, 2, 3, 7, 0, 0, 0, 0},
        {1, 2, 3, 7, 0, 0, 0, 0},
        {0, 1, 2, 3, 7, 0, 0, 0},
        {4, 7, 0, 0, 0, 0, 0, 0},
        {0, 4, 7, 0, 0, 0, 0, 0},
        {1, 4, 7, 0, 0, 0, 0, 0},
        {0, 1, 4, 7, 0, 0, 0, 0},
        {2, 4, 7, 0, 0, 0, 0, 0},
        {0, 2, 4, 7, 0, 0, 0, 0},
        {1, 2, 4, 7, 0, 0, 0, 0},
        {0, 1, 2, 4, 7, 0, 0, 0},
        {3, 4, 7, 0, 0, 0, 0, 0},
        {0, 3, 4, 7, 0, 0, 0, 0},
        {1, 3, 4, 7, 0, 0, 0, 0},
        {0, 1, 3, 4, 7, 0, 0, 0},
        {2, 3, 4, 7, 0, 0, 0, 0},
        {0, 2, 3, 4, 7, 0, 0, 0},
        {1, 2, 3, 4, 7, 0, 0, 0},
        {0, 1, 2, 3, 4, 7, 0, 0},
        {5, 7, 0, 0, 0, 0, 0, 0},
        {0, 5, 7, 0, 0, 0, 0, 0},
        {1, 5, 7, 0, 0, 0, 0, 0},
        {0, 1, 5, 7, 0, 0, 0, 0},
        {2, 5, 7, 0, 0, 0, 0, 0},
        {0, 2, 5, 7, 0, 0, 0, 0},
        {1, 2, 5, 7, 0, 0, 0, 0},
        {0, 1, 2, 5, 7, 0, 0, 0},
        {3, 5, 7, 0, 0, 0, 0, 0},
        {0, 3, 5, 7, 0, 0, 0, 0},
        {1, 3, 5, 7, 0, 0, 0, 0},
        {0, 1, 3, 5, 7, 0, 0, 0},
        {2, 3, 5, 7, 0, 0, 0, 0},
        {0, 2, 3, 5, 7, 0, 0, 0},
        {1, 2, 3, 5, 7, 0, 0, 0},
        {0, 1, 2, 3, 5, 7, 0, 0},
        {4, 5, 7, 0, 0, 0, 0, 0},
        {0, 4, 5, 7, 0, 0, 0, 0},
        {1, 4, 5, 7, 0, 0, 0, 0},
        {0, 1, 4, 5, 7, 0, 0, 0},
        {2, 4, 5, 7, 0, 0, 0, 0},
        {0, 2, 4, 5, 7, 0, 0, 0},
        {1, 2, 4, 5, 7, 0, 0, 0},
        {0, 1, 2, 4, 5, 7, 0, 0},
        {3, 4, 5, 7, 0, 0, 0, 0},
        {0, 3, 4, 5, 7, 0, 0, 0},
        {1, 3, 4, 5, 7, 0, 0, 0},
        {0, 1, 3, 4, 5, 7, 0, 0},
        {2, 3, 4, 5, 7, 0, 0, 0},
        {0, 2, 3, 4, 5, 7, 0, 0},
        {1, 2, 3, 4, 5, 7, 0, 0},
        {0, 1, 2, 3, 4, 5, 7, 0},
        {6, 7, 0, 0, 0, 0, 0, 0},
        {0, 6, 7, 0, 0, 0, 0, 0},
        {1, 6, 7, 0, 0, 0, 0, 0},
        {0, 1, 6, 7, 0, 0, 0, 0},
        {2, 6, 7, 0, 0, 0, 0, 0},
        {0, 2, 6, 7, 0, 0, 0, 0},
        {1, 2, 6, 7, 0, 0, 0, 0},
        {0, 1, 2, 6, 7, 0, 0, 0},
        {3, 6, 7, 0, 0, 0, 0, 0},
        {0, 3, 6, 7, 0, 0, 0, 0},
        {1, 3, 6, 7, 0, 0, 0, 0},
        {0, 1, 3, 6, 7, 0, 0, 0},
        {2, 3, 6, 7, 0, 0, 0, 0},
        {0, 2, 3, 6, 7, 0, 0, 0},
        {1, 2, 3, 6, 7, 0, 0, 0},
        {0, 1, 2, 3, 6, 7, 0, 0},
        {4, 6, 7, 0, 0, 0, 0, 0},
        {0, 4, 6, 7, 0, 0, 0, 0},
        {1, 4, 6, 7, 0, 0, 0, 0},
        {0, 1, 4, 6, 7, 0, 0, 0},
        {2, 4, 6, 7, 0, 0, 0, 0},
        {0, 2, 4, 6, 7, 0, 0, 0},
        {1, 2, 4, 6, 7, 0, 0, 0},
        {0, 1, 2, 4, 6, 7, 0, 0},
        {3, 4, 6, 7, 0, 0, 0, 0},
        {0, 3, 4, 6, 7, 0, 0, 0},
        {1, 3, 4, 6, 7, 0, 0, 0},
        {0, 1, 3, 4, 6, 7, 0, 0},
        {2, 3, 4, 6, 7, 0, 0, 0},
        {0, 2, 3, 4, 6, 7, 0, 0},
        {1, 2, 3, 4, 6, 7, 0, 0},
        {0, 1, 2, 3, 4, 6, 7, 0},
        {5, 6, 7, 0, 0, 0, 0, 0},
        {0, 5, 6, 7, 0, 0, 0, 0},
        {1, 5, 6, 7, 0, 0, 0, 0},
        {0, 1, 5, 6, 7, 0, 0, 0},
        {2, 5, 6, 7, 0, 0, 0, 0},
        {0, 2, 5, 6, 7, 0, 0, 0},
        {1, 2, 5, 6, 7, 0, 0, 0},
        {0, 1, 2, 5, 6, 7, 0, 0},
        {3, 5, 6, 7, 0, 0, 0, 0},
        {0, 3, 5, 6, 7, 0, 0, 0},
        {1, 3, 5, 6, 7, 0, 0, 0},
        {0, 1, 3, 5, 6, 7, 0, 0},
        {2, 3, 5, 6, 7, 0, 0, 0},
        {0, 2, 3, 5, 6, 7, 0, 0},
        {1, 2, 3, 5, 6, 7, 0, 0},
        {0, 1, 2, 3, 5, 6, 7, 0},
        {4, 5, 6, 7, 0, 0, 0, 0},
        {0, 4, 5, 6, 7, 0, 0, 0},
        {1, 4, 5, 6, 7, 0, 0, 0},
        {0, 1, 4, 5, 6, 7, 0, 0},
        {2, 4, 5, 6, 7, 0, 0, 0},
        {0, 2, 4, 5, 6, 7, 0, 0},
        {1, 2, 4, 5, 6, 7, 0, 0},
        {0, 1, 2, 4, 5, 6, 7, 0},
        {3, 4, 5, 6, 7, 0, 0, 0},
        {0, 3, 4, 5, 6, 7, 0, 0},
        {1, 3, 4, 5, 6, 7, 0, 0},
        {0, 1, 3, 4, 5, 6, 7, 0},
        {2, 3, 4, 5, 6, 7, 0, 0},
        {0, 2, 3, 4, 5, 6, 7, 0},
        {1, 2, 3, 4, 5, 6, 7, 0},
        {0, 1, 2, 3, 4, 5, 6, 7}
};


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
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good]));
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
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good]));
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
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good]));
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





#define REJ(n) \
g0 = _mm256_castsi256_si128(f##n);\
g1 = _mm_bsrli_si128(g0, 8);\
g2 = _mm256_extracti128_si256(f##n, 1);\
g3 = _mm_bsrli_si128(g2, 8);\
\
d0 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good##n & 0xFF]);\
d1 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[(good##n >> 8) & 0xFF]);\
d2 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[(good##n >> 16) & 0xFF]);\
d3 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[(good##n >> 24) & 0xFF]);\
\
d0 = _mm_shuffle_epi8(g0,d0);\
d1 = _mm_shuffle_epi8(g1,d1);\
d2 = _mm_shuffle_epi8(g2,d2);\
d3 = _mm_shuffle_epi8(g3,d3);\
\
f4 = _mm256_cvtepi8_epi32(d0);\
f5 = _mm256_cvtepi8_epi32(d1);\
f6 = _mm256_cvtepi8_epi32(d2);\
f7 = _mm256_cvtepi8_epi32(d3);\
\
_mm256_storeu_si256((__m256i *) &r[ctr], f4);                                     \
_mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(etas, d0));\
ctr += _mm_popcnt_u32(good##n & 0xFF);\
_mm256_storeu_si256((__m256i *) &r[ctr], f5);                                     \
_mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(etas, d1));\
ctr += _mm_popcnt_u32((good##n >> 8) & 0xFF);\
_mm256_storeu_si256((__m256i *) &r[ctr], f6);                                     \
_mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(etas, d2));\
ctr += _mm_popcnt_u32((good##n >> 16) & 0xFF);\
_mm256_storeu_si256((__m256i *) &r[ctr], f7);                                     \
_mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(etas, d3));\
ctr += _mm_popcnt_u32((good##n >> 24) & 0xFF);\
\



#define REJ2(n) \
g0 = _mm256_castsi256_si128(f##n);\
g1 = _mm_bsrli_si128(g0, 8);\
g2 = _mm256_extracti128_si256(f##n, 1);\
g3 = _mm_bsrli_si128(g2, 8);\
\
t4 = _mm256_cvtepu8_epi32(g0);\
t5 = _mm256_cvtepu8_epi32(g1);\
t6 = _mm256_cvtepu8_epi32(g2);\
t7 = _mm256_cvtepu8_epi32(g3);\
\
_mm256_storeu_si256((__m256i *) &r[ctr], t4);\
ctr += 8;\
_mm256_storeu_si256((__m256i *) &r[ctr], t5);\
ctr += 8;\
_mm256_storeu_si256((__m256i *) &r[ctr], t6);\
ctr += 8;\
_mm256_storeu_si256((__m256i *) &r[ctr], t7);\
ctr += 8;\
\


static uint32_t rej_eta_final2(int32_t *restrict r, uint8_t *pipe, uint32_t ctr, const uint8_t *buf) {
    __m256i f0, f1;
    __m128i g0, g1;
    __m128i d0, d1;
    uint32_t good0;

    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i mask2 = _mm_set1_epi8(0x03);
    const __m128i eta = _mm_set1_epi8(ETA);
    const __m128i bound = mask;
    const __m128i num13 = _mm_set1_epi16(13);
    const __m128i num5 = _mm_set1_epi16(5);

    g0 = _mm_loadl_epi64((__m128i*)buf);
    g0 = _mm_cvtepu8_epi16(g0);
    g1 = _mm_slli_epi16(g0,4);
    g0 = (g0 | g1) & mask;

    good0 = _mm_movemask_epi8(_mm_sub_epi8(g0,bound));

    g1 = _mm_mullo_epi16(g0, num13);
    g1 = _mm_srli_epi16(g1,6);
    g1 = g1 & mask2;
    g1 = _mm_mullo_epi16(g1, num5);

    g0 = _mm_sub_epi8(g0, g1);
    g0 = _mm_sub_epi8(eta, g0);

    //ctr <= 240
    if (ctr <= (N - 16)) {
        g1 = _mm_bsrli_si128(g0, 8);

        d0 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0 & 0xFF]);
        d1 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[(good0 >> 8) & 0xFF]);
        g0 = _mm_shuffle_epi8(g0,d0);
        g1 = _mm_shuffle_epi8(g1,d1);
        f0 = _mm256_cvtepi8_epi32(g0);
        f1 = _mm256_cvtepi8_epi32(g1);

        _mm256_storeu_si256((__m256i *) &r[ctr], f0);
        _mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(eta, g0));
        ctr += _mm_popcnt_u32(good0 & 0xFF);
        _mm256_storeu_si256((__m256i *) &r[ctr], f1);
        _mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(eta, g1));
        ctr += _mm_popcnt_u32((good0 >> 8) & 0xFF);

        return ctr;
    }

    // 248 >= ctr > 240
    if (ctr <= (N - 8)) {
        g1 = _mm_bsrli_si128(g0, 8);

        d0 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0 & 0xFF]);
        d1 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[(good0 >> 8) & 0xFF]);
        g0 = _mm_shuffle_epi8(g0,d0);
        g1 = _mm_shuffle_epi8(g1,d1);
        f0 = _mm256_cvtepi8_epi32(g0);
        f1 = _mm256_cvtepi8_epi32(g1);

        _mm256_storeu_si256((__m256i *) &r[ctr], f0);
        _mm_storeu_si128((__m128i *)&pipe[ctr], _mm_sub_epi8(eta, g0));
        ctr += _mm_popcnt_u32(good0 & 0xFF);

        ALIGN(32) int32_t t[8];
        _mm256_storeu_si256((__m256i *) t, f1);
        int count = _mm_popcnt_u32((good0 >> 8) & 0xFF);
        int i = 0;
        while(count != 0 && ctr < N) {
            r[ctr] = t[i];
            pipe[ctr] = ETA - t[i];
            i++;
            count--;
            ctr++;
        }

        return ctr;
    }

    //ctr > 248
    ALIGN(32) int32_t t[8];

    d0 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0 & 0xFF]);
    d0 = _mm_shuffle_epi8(g0,d0);
    f0 = _mm256_cvtepi8_epi32(d0);

    _mm256_storeu_si256((__m256i *) t, f0);

    int count = _mm_popcnt_u32((good0 >> 8) & 0xFF);
    int i = 0;
    while(count > 0 && ctr < N) {
        r[ctr] = t[i];
        pipe[ctr] = ETA - t[i];
        i++;
        count--;
        ctr++;
    }

    return ctr;
}

unsigned int XURQ_AVX2_rej_eta_avx_with_pack(int32_t *restrict r, uint8_t *pipe, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]) {
    unsigned int ctr, pos;
    uint32_t good0, good1, good2, good3;
    __m256i f0, f1, f2, f3, f4, f5, f6, f7;
    __m128i g0, g1, g2, g3;
    __m128i d0, d1, d2, d3;
    const __m256i mask = _mm256_set1_epi8(0x0f);
    const __m256i mask2 = _mm256_set1_epi8(0x03);
    const __m256i eta = _mm256_set1_epi8(ETA);
    const __m256i bound = mask;//15
    const __m256i num5 = _mm256_set1_epi16(5);
    const __m256i num13 = _mm256_set1_epi16(13);
    const __m128i etas = _mm_set1_epi8(ETA);

    ctr = 0;

    f1 = _mm256_loadu_si256((__m256i *) (buf));
    f3 = _mm256_loadu_si256((__m256i *) (buf + 32));

    f0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(f1));
    f1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(f1, 1));
    f2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(f3));
    f3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(f3, 1));

    f4 = _mm256_slli_epi16(f0, 4);
    f5 = _mm256_slli_epi16(f1, 4);
    f6 = _mm256_slli_epi16(f2, 4);
    f7 = _mm256_slli_epi16(f3, 4);

    f0 = _mm256_or_si256(f0, f4);
    f1 = _mm256_or_si256(f1, f5);
    f2 = _mm256_or_si256(f2, f6);
    f3 = _mm256_or_si256(f3, f7);

    f0 = f0 & mask;
    f1 = f1 & mask;
    f2 = f2 & mask;
    f3 = f3 & mask;

    good0 = _mm256_movemask_epi8(_mm256_sub_epi8(f0,bound));
    good1 = _mm256_movemask_epi8(_mm256_sub_epi8(f1,bound));
    good2 = _mm256_movemask_epi8(_mm256_sub_epi8(f2,bound));
    good3 = _mm256_movemask_epi8(_mm256_sub_epi8(f3,bound));

    f4 =_mm256_mullo_epi16(f0,num13);
    f5 =_mm256_mullo_epi16(f1,num13);
    f6 =_mm256_mullo_epi16(f2,num13);
    f7 =_mm256_mullo_epi16(f3,num13);

    f4 = _mm256_srli_epi32(f4, 6);
    f5 = _mm256_srli_epi32(f5, 6);
    f6 = _mm256_srli_epi32(f6, 6);
    f7 = _mm256_srli_epi32(f7, 6);

    f4 = f4 & mask2;
    f5 = f5 & mask2;
    f6 = f6 & mask2;
    f7 = f7 & mask2;

    f4 = _mm256_mullo_epi16(f4,num5);
    f5 = _mm256_mullo_epi16(f5,num5);
    f6 = _mm256_mullo_epi16(f6,num5);
    f7 = _mm256_mullo_epi16(f7,num5);

    f0 = _mm256_sub_epi8(f0, f4);
    f1 = _mm256_sub_epi8(f1, f5);
    f2 = _mm256_sub_epi8(f2, f6);
    f3 = _mm256_sub_epi8(f3, f7);

    f0 = _mm256_sub_epi8(eta, f0);
    f1 = _mm256_sub_epi8(eta, f1);
    f2 = _mm256_sub_epi8(eta, f2);
    f3 = _mm256_sub_epi8(eta, f3);

    REJ(0)
    REJ(1)
    REJ(2)
    REJ(3)

    f1 = _mm256_loadu_si256((__m256i *) (buf + 64));
    f3 = _mm256_loadu_si256((__m256i *) (buf + 96));

    f0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(f1));
    f1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(f1, 1));
    f2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(f3));
    f3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(f3, 1));

    f4 = _mm256_slli_epi16(f0, 4);
    f5 = _mm256_slli_epi16(f1, 4);
    f6 = _mm256_slli_epi16(f2, 4);
    f7 = _mm256_slli_epi16(f3, 4);

    f0 = _mm256_or_si256(f0, f4);
    f1 = _mm256_or_si256(f1, f5);
    f2 = _mm256_or_si256(f2, f6);
    f3 = _mm256_or_si256(f3, f7);

    f0 = f0 & mask;
    f1 = f1 & mask;
    f2 = f2 & mask;
    f3 = f3 & mask;

    good0 = _mm256_movemask_epi8(_mm256_sub_epi8(f0,bound));
    good1 = _mm256_movemask_epi8(_mm256_sub_epi8(f1,bound));
    good2 = _mm256_movemask_epi8(_mm256_sub_epi8(f2,bound));
    good3 = _mm256_movemask_epi8(_mm256_sub_epi8(f3,bound));

    f4 =_mm256_mullo_epi16(f0,num13);
    f5 =_mm256_mullo_epi16(f1,num13);
    f6 =_mm256_mullo_epi16(f2,num13);
    f7 =_mm256_mullo_epi16(f3,num13);

    f4 = _mm256_srli_epi32(f4, 6);
    f5 = _mm256_srli_epi32(f5, 6);
    f6 = _mm256_srli_epi32(f6, 6);
    f7 = _mm256_srli_epi32(f7, 6);

    f4 = f4 & mask2;
    f5 = f5 & mask2;
    f6 = f6 & mask2;
    f7 = f7 & mask2;

    f4 = _mm256_mullo_epi16(f4,num5);
    f5 = _mm256_mullo_epi16(f5,num5);
    f6 = _mm256_mullo_epi16(f6,num5);
    f7 = _mm256_mullo_epi16(f7,num5);

    f0 = _mm256_sub_epi8(f0, f4);
    f1 = _mm256_sub_epi8(f1, f5);
    f2 = _mm256_sub_epi8(f2, f6);
    f3 = _mm256_sub_epi8(f3, f7);

    f0 = _mm256_sub_epi8(eta, f0);
    f1 = _mm256_sub_epi8(eta, f1);
    f2 = _mm256_sub_epi8(eta, f2);
    f3 = _mm256_sub_epi8(eta, f3);

    REJ(0)
    REJ(1)
    REJ(2)
    REJ(3)


    if (ctr < N)
        ctr = rej_eta_final2(r, pipe, ctr,&buf[128]);

    return ctr;
}



