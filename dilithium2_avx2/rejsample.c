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


unsigned int XURQ_AVX2_rej_uniform_avx_with_s2(int32_t *restrict r, const uint8_t buf[REJ_UNIFORM_BUFLEN + 8]) {
    unsigned int ctr, pos;
    uint32_t good0, good1, good2, good3;
    __m256i d0, d1, d2, d3;
    __m256i tmp0, tmp1, tmp2, tmp3;
    const __m256i bound = _mm256_set1_epi32(Q);
    const __m256i mask = _mm256_set1_epi32(0x7FFFFF);
    const __m256i idx8 = _mm256_set_epi8(-1, 15, 14, 13, -1, 12, 11, 10,
                                         -1, 9, 8, 7, -1, 6, 5, 4,
                                         -1, 11, 10, 9, -1, 8, 7, 6,
                                         -1, 5, 4, 3, -1, 2, 1, 0);

    ctr = pos = 0;

    for (int i = 0; i < 8; ++i) {
        d0 = _mm256_loadu_si256((__m256i *) &buf[pos]);
        d1 = _mm256_loadu_si256((__m256i *) &buf[pos + 24]);
        d2 = _mm256_loadu_si256((__m256i *) &buf[pos + 48]);
        d3 = _mm256_loadu_si256((__m256i *) &buf[pos + 72]);
        pos += 96;

        d0 = _mm256_permute4x64_epi64(d0, 0x94);
        d1 = _mm256_permute4x64_epi64(d1, 0x94);
        d2 = _mm256_permute4x64_epi64(d2, 0x94);
        d3 = _mm256_permute4x64_epi64(d3, 0x94);

        d0 = _mm256_shuffle_epi8(d0, idx8);
        d1 = _mm256_shuffle_epi8(d1, idx8);
        d2 = _mm256_shuffle_epi8(d2, idx8);
        d3 = _mm256_shuffle_epi8(d3, idx8);

        d0 = _mm256_and_si256(d0, mask);
        d1 = _mm256_and_si256(d1, mask);
        d2 = _mm256_and_si256(d2, mask);
        d3 = _mm256_and_si256(d3, mask);

        tmp0 = _mm256_sub_epi32(d0, bound);
        tmp1 = _mm256_sub_epi32(d1, bound);
        tmp2 = _mm256_sub_epi32(d2, bound);
        tmp3 = _mm256_sub_epi32(d3, bound);

        good0 = _mm256_movemask_ps((__m256) tmp0);
        good1 = _mm256_movemask_ps((__m256) tmp1);
        good2 = _mm256_movemask_ps((__m256) tmp2);
        good3 = _mm256_movemask_ps((__m256) tmp3);

        tmp0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0]));
        tmp1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good1]));
        tmp2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good2]));
        tmp3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good3]));

        d0 = _mm256_permutevar8x32_epi32(d0, tmp0);
        d1 = _mm256_permutevar8x32_epi32(d1, tmp1);
        d2 = _mm256_permutevar8x32_epi32(d2, tmp2);
        d3 = _mm256_permutevar8x32_epi32(d3, tmp3);

        _mm256_storeu_si256((__m256i *) &r[ctr], d0);
        ctr += _mm_popcnt_u32(good0);
        _mm256_storeu_si256((__m256i *) &r[ctr], d1);
        ctr += _mm_popcnt_u32(good1);
        _mm256_storeu_si256((__m256i *) &r[ctr], d2);
        ctr += _mm_popcnt_u32(good2);
        _mm256_storeu_si256((__m256i *) &r[ctr], d3);
        ctr += _mm_popcnt_u32(good3);
    }

    uint32_t t;
    while (ctr < N && pos <= REJ_UNIFORM_BUFLEN - 3) {
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

unsigned int dep_XURQ_AVX2_rej_uniform_avx_s1s2(int32_t *restrict r, const uint8_t *buf, unsigned int num) {
    unsigned int ctr;
    uint32_t good0, good1, good2, good3;
    __m256i d0, d1, d2, d3;
    __m256i tmp0, tmp1, tmp2, tmp3;
    const __m256i bound = _mm256_set1_epi32(Q);
    const __m256i mask = _mm256_set1_epi32(0x7FFFFF);
    const __m256i idx8 = _mm256_set_epi8(-1, 15, 14, 13, -1, 12, 11, 10,
                                         -1, 9, 8, 7, -1, 6, 5, 4,
                                         -1, 11, 10, 9, -1, 8, 7, 6,
                                         -1, 5, 4, 3, -1, 2, 1, 0);

    ctr = num;

    d0 = _mm256_loadu_si256((__m256i *) &buf[0]);
    d1 = _mm256_loadu_si256((__m256i *) &buf[24]);
    d2 = _mm256_loadu_si256((__m256i *) &buf[48]);
    d3 = _mm256_loadu_si256((__m256i *) &buf[72]);

    d0 = _mm256_permute4x64_epi64(d0, 0x94);
    d1 = _mm256_permute4x64_epi64(d1, 0x94);
    d2 = _mm256_permute4x64_epi64(d2, 0x94);
    d3 = _mm256_permute4x64_epi64(d3, 0x94);

    d0 = _mm256_shuffle_epi8(d0, idx8);
    d1 = _mm256_shuffle_epi8(d1, idx8);
    d2 = _mm256_shuffle_epi8(d2, idx8);
    d3 = _mm256_shuffle_epi8(d3, idx8);

    d0 = _mm256_and_si256(d0, mask);
    d1 = _mm256_and_si256(d1, mask);
    d2 = _mm256_and_si256(d2, mask);
    d3 = _mm256_and_si256(d3, mask);

    tmp0 = _mm256_sub_epi32(d0, bound);
    tmp1 = _mm256_sub_epi32(d1, bound);
    tmp2 = _mm256_sub_epi32(d2, bound);
    tmp3 = _mm256_sub_epi32(d3, bound);

    good0 = _mm256_movemask_ps((__m256) tmp0);
    good1 = _mm256_movemask_ps((__m256) tmp1);
    good2 = _mm256_movemask_ps((__m256) tmp2);
    good3 = _mm256_movemask_ps((__m256) tmp3);

    tmp0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0]));
    tmp1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good1]));
    tmp2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good2]));
    tmp3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good3]));

    d0 = _mm256_permutevar8x32_epi32(d0, tmp0);
    d1 = _mm256_permutevar8x32_epi32(d1, tmp1);
    d2 = _mm256_permutevar8x32_epi32(d2, tmp2);
    d3 = _mm256_permutevar8x32_epi32(d3, tmp3);

    _mm256_storeu_si256((__m256i *) &r[ctr], d0);
    ctr += _mm_popcnt_u32(good0);
    _mm256_storeu_si256((__m256i *) &r[ctr], d1);
    ctr += _mm_popcnt_u32(good1);
    _mm256_storeu_si256((__m256i *) &r[ctr], d2);
    ctr += _mm_popcnt_u32(good2);
    _mm256_storeu_si256((__m256i *) &r[ctr], d3);
    ctr += _mm_popcnt_u32(good3);

    d0 = _mm256_loadu_si256((__m256i *) &buf[96]);
    d1 = _mm256_loadu_si256((__m256i *) &buf[120]);
    d2 = _mm256_loadu_si256((__m256i *) &buf[144]);

    d0 = _mm256_permute4x64_epi64(d0, 0x94);
    d1 = _mm256_permute4x64_epi64(d1, 0x94);
    d2 = _mm256_permute4x64_epi64(d2, 0x94);

    d0 = _mm256_shuffle_epi8(d0, idx8);
    d1 = _mm256_shuffle_epi8(d1, idx8);
    d2 = _mm256_shuffle_epi8(d2, idx8);

    d0 = _mm256_and_si256(d0, mask);
    d1 = _mm256_and_si256(d1, mask);
    d2 = _mm256_and_si256(d2, mask);

    tmp0 = _mm256_sub_epi32(d0, bound);
    tmp1 = _mm256_sub_epi32(d1, bound);
    tmp2 = _mm256_sub_epi32(d2, bound);

    good0 = _mm256_movemask_ps((__m256) tmp0);
    good1 = _mm256_movemask_ps((__m256) tmp1);
    good2 = _mm256_movemask_ps((__m256) tmp2);

    tmp0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0]));
    tmp1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good1]));
    tmp2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good2]));

    d0 = _mm256_permutevar8x32_epi32(d0, tmp0);
    d1 = _mm256_permutevar8x32_epi32(d1, tmp1);
    d2 = _mm256_permutevar8x32_epi32(d2, tmp2);

    _mm256_storeu_si256((__m256i *) &r[ctr], d0);
    ctr += _mm_popcnt_u32(good0);
    _mm256_storeu_si256((__m256i *) &r[ctr], d1);
    ctr += _mm_popcnt_u32(good1);
    _mm256_storeu_si256((__m256i *) &r[ctr], d2);
    ctr += _mm_popcnt_u32(good2);

    return ctr;
}

#define SAMPLE_H(a, b, c, d, m) \
d0 = _mm256_loadu_si256((__m256i *) &buf[a][(m)]);\
d1 = _mm256_loadu_si256((__m256i *) &buf[b][(m)]);\
d2 = _mm256_loadu_si256((__m256i *) &buf[c][(m)]);\
d3 = _mm256_loadu_si256((__m256i *) &buf[d][(m)]);\
\
d0 = _mm256_permute4x64_epi64(d0, 0x94);\
d1 = _mm256_permute4x64_epi64(d1, 0x94);\
d2 = _mm256_permute4x64_epi64(d2, 0x94);\
d3 = _mm256_permute4x64_epi64(d3, 0x94);\
\
d0 = _mm256_shuffle_epi8(d0, idx8);\
d1 = _mm256_shuffle_epi8(d1, idx8);\
d2 = _mm256_shuffle_epi8(d2, idx8);\
d3 = _mm256_shuffle_epi8(d3, idx8);\
\
d0 = _mm256_and_si256(d0, mask);\
d1 = _mm256_and_si256(d1, mask);\
d2 = _mm256_and_si256(d2, mask);\
d3 = _mm256_and_si256(d3, mask);\
\
tmp0 = _mm256_sub_epi32(d0, bound);\
tmp1 = _mm256_sub_epi32(d1, bound);\
tmp2 = _mm256_sub_epi32(d2, bound);\
tmp3 = _mm256_sub_epi32(d3, bound);\
\
good0 = _mm256_movemask_ps((__m256) tmp0);\
good1 = _mm256_movemask_ps((__m256) tmp1);\
good2 = _mm256_movemask_ps((__m256) tmp2);\
good3 = _mm256_movemask_ps((__m256) tmp3);\
\
tmp0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0]));\
tmp1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good1]));\
tmp2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good2]));\
tmp3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good3]));\
\
d0 = _mm256_permutevar8x32_epi32(d0, tmp0);\
d1 = _mm256_permutevar8x32_epi32(d1, tmp1);\
d2 = _mm256_permutevar8x32_epi32(d2, tmp2);\
d3 = _mm256_permutevar8x32_epi32(d3, tmp3);\
\
_mm256_storeu_si256((__m256i *) &r##a[ctr[a]], d0);\
ctr[a] += _mm_popcnt_u32(good0);\
_mm256_storeu_si256((__m256i *) &r##b[ctr[b]], d1);\
ctr[b] += _mm_popcnt_u32(good1);\
_mm256_storeu_si256((__m256i *) &r##c[ctr[c]], d2);\
ctr[c] += _mm_popcnt_u32(good2);\
_mm256_storeu_si256((__m256i *) &r##d[ctr[d]], d3);\
ctr[d] += _mm_popcnt_u32(good3);\



void XURQ_AVX2_rej_uniform_avx_s1s2_h(int32_t *restrict r0,
                                      int32_t *restrict r1,
                                      int32_t *restrict r2,
                                      int32_t *restrict r3,
                                      int32_t *restrict r4,
                                      int32_t *restrict r5,
                                      int32_t *restrict r6,
                                      int32_t *restrict r7,
                                      const uint8_t buf[8][176], unsigned int *ctr) {

    uint32_t good0, good1, good2, good3;
    __m256i d0, d1, d2, d3;
    __m256i tmp0, tmp1, tmp2, tmp3;
    const __m256i bound = _mm256_set1_epi32(Q);
    const __m256i mask = _mm256_set1_epi32(0x7FFFFF);
    const __m256i idx8 = _mm256_set_epi8(-1, 15, 14, 13, -1, 12, 11, 10,
                                         -1, 9, 8, 7, -1, 6, 5, 4,
                                         -1, 11, 10, 9, -1, 8, 7, 6,
                                         -1, 5, 4, 3, -1, 2, 1, 0);


    for (int i = 0; i < 7; ++i) {
        SAMPLE_H(0, 1, 2, 3, i * 24)
    }

    for (int i = 0; i < 7; ++i) {
        SAMPLE_H(4, 5, 6, 7, i * 24)
    }

}

#define SAMPLE_B(n) \
d0 = _mm256_loadu_si256((__m256i *) &buf[n][0]);\
d1 = _mm256_loadu_si256((__m256i *) &buf[n][24]);\
d2 = _mm256_loadu_si256((__m256i *) &buf[n][48]);\
d3 = _mm256_loadu_si256((__m256i *) &buf[n][72]);\
\
d0 = _mm256_permute4x64_epi64(d0, 0x94);\
d1 = _mm256_permute4x64_epi64(d1, 0x94);\
d2 = _mm256_permute4x64_epi64(d2, 0x94);\
d3 = _mm256_permute4x64_epi64(d3, 0x94);\
\
d0 = _mm256_shuffle_epi8(d0, idx8);\
d1 = _mm256_shuffle_epi8(d1, idx8);\
d2 = _mm256_shuffle_epi8(d2, idx8);\
d3 = _mm256_shuffle_epi8(d3, idx8);\
\
d0 = _mm256_and_si256(d0, mask);\
d1 = _mm256_and_si256(d1, mask);\
d2 = _mm256_and_si256(d2, mask);\
d3 = _mm256_and_si256(d3, mask);\
\
tmp0 = _mm256_sub_epi32(d0, bound);\
tmp1 = _mm256_sub_epi32(d1, bound);\
tmp2 = _mm256_sub_epi32(d2, bound);\
tmp3 = _mm256_sub_epi32(d3, bound);\
\
good0 = _mm256_movemask_ps((__m256) tmp0);\
good1 = _mm256_movemask_ps((__m256) tmp1);\
good2 = _mm256_movemask_ps((__m256) tmp2);\
good3 = _mm256_movemask_ps((__m256) tmp3);\
\
tmp0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0]));\
tmp1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good1]));\
tmp2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good2]));\
tmp3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good3]));\
\
d0 = _mm256_permutevar8x32_epi32(d0, tmp0);\
d1 = _mm256_permutevar8x32_epi32(d1, tmp1);\
d2 = _mm256_permutevar8x32_epi32(d2, tmp2);\
d3 = _mm256_permutevar8x32_epi32(d3, tmp3);\
\
_mm256_storeu_si256((__m256i *) &r##n[ctr[n]], d0);\
ctr[n] += _mm_popcnt_u32(good0);\
_mm256_storeu_si256((__m256i *) &r##n[ctr[n]], d1);\
ctr[n] += _mm_popcnt_u32(good1);\
_mm256_storeu_si256((__m256i *) &r##n[ctr[n]], d2);\
ctr[n] += _mm_popcnt_u32(good2);\
_mm256_storeu_si256((__m256i *) &r##n[ctr[n]], d3);\
ctr[n] += _mm_popcnt_u32(good3);\
\
d0 = _mm256_loadu_si256((__m256i *) &buf[n][96]);\
d1 = _mm256_loadu_si256((__m256i *) &buf[n][120]);\
d2 = _mm256_loadu_si256((__m256i *) &buf[n][144]);\
\
d0 = _mm256_permute4x64_epi64(d0, 0x94);\
d1 = _mm256_permute4x64_epi64(d1, 0x94);\
d2 = _mm256_permute4x64_epi64(d2, 0x94);\
\
d0 = _mm256_shuffle_epi8(d0, idx8);\
d1 = _mm256_shuffle_epi8(d1, idx8);\
d2 = _mm256_shuffle_epi8(d2, idx8);\
\
d0 = _mm256_and_si256(d0, mask);\
d1 = _mm256_and_si256(d1, mask);\
d2 = _mm256_and_si256(d2, mask);\
\
tmp0 = _mm256_sub_epi32(d0, bound);\
tmp1 = _mm256_sub_epi32(d1, bound);\
tmp2 = _mm256_sub_epi32(d2, bound);\
\
good0 = _mm256_movemask_ps((__m256) tmp0);\
good1 = _mm256_movemask_ps((__m256) tmp1);\
good2 = _mm256_movemask_ps((__m256) tmp2);\
\
tmp0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0]));\
tmp1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good1]));\
tmp2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good2]));\
\
d0 = _mm256_permutevar8x32_epi32(d0, tmp0);\
d1 = _mm256_permutevar8x32_epi32(d1, tmp1);\
d2 = _mm256_permutevar8x32_epi32(d2, tmp2);\
\
_mm256_storeu_si256((__m256i *) &r##n[ctr[n]], d0);\
ctr[n] += _mm_popcnt_u32(good0);\
_mm256_storeu_si256((__m256i *) &r##n[ctr[n]], d1);\
ctr[n] += _mm_popcnt_u32(good1);\
_mm256_storeu_si256((__m256i *) &r##n[ctr[n]], d2);\
ctr[n] += _mm_popcnt_u32(good2);\


void XURQ_AVX2_rej_uniform_avx_s1s2b(int32_t *restrict r0,
                                     int32_t *restrict r1,
                                     int32_t *restrict r2,
                                     int32_t *restrict r3,
                                     int32_t *restrict r4,
                                     int32_t *restrict r5,
                                     int32_t *restrict r6,
                                     int32_t *restrict r7,
                                     const uint8_t buf[8][176], unsigned int *ctr) {

    uint32_t good0, good1, good2, good3;
    __m256i d0, d1, d2, d3;
    __m256i tmp0, tmp1, tmp2, tmp3;
    const __m256i bound = _mm256_set1_epi32(Q);
    const __m256i mask = _mm256_set1_epi32(0x7FFFFF);
    const __m256i idx8 = _mm256_set_epi8(-1, 15, 14, 13, -1, 12, 11, 10,
                                         -1, 9, 8, 7, -1, 6, 5, 4,
                                         -1, 11, 10, 9, -1, 8, 7, 6,
                                         -1, 5, 4, 3, -1, 2, 1, 0);


    SAMPLE_B(0)
    SAMPLE_B(1)
    SAMPLE_B(2)
    SAMPLE_B(3)
    SAMPLE_B(4)
    SAMPLE_B(5)
    SAMPLE_B(6)
    SAMPLE_B(7)
}


unsigned int XURQ_rej_uniform_avx_one_block_with_s12h3(int32_t *restrict r, const uint8_t *restrict buf) {
    unsigned int ctr, pos;
    uint32_t good;
    __m256i d, tmp;
    const __m256i bound = _mm256_set1_epi32(Q);
    const __m256i mask = _mm256_set1_epi32(0x7FFFFF);
    const __m256i idx8 = _mm256_set_epi8(-1, 15, 14, 13, -1, 12, 11, 10,
                                         -1, 9, 8, 7, -1, 6, 5, 4,
                                         -1, 11, 10, 9, -1, 8, 7, 6,
                                         -1, 5, 4, 3, -1, 2, 1, 0);

    ctr = pos = 0;
    for (int i = 0; i < 7; ++i) {
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
    return ctr;
}

unsigned int PQCLEAN_DILITHIUM2_AVX2_rej_eta_avx(int32_t *restrict r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]) {
    unsigned int ctr, pos;
    uint32_t good;
    __m256i f0, f1, f2;
    __m128i g0, g1;
    const __m256i mask = _mm256_set1_epi8(15);
    const __m256i eta = _mm256_set1_epi8(ETA);
    const __m256i bound = mask;
    const __m256i v = _mm256_set1_epi32(-6560);
    const __m256i p = _mm256_set1_epi32(5);

    ctr = pos = 0;
    while (ctr <= N - 8 && pos <= REJ_UNIFORM_ETA_BUFLEN - 16) {
        f0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) &buf[pos]));
        f1 = _mm256_slli_epi16(f0, 4);
        f0 = _mm256_or_si256(f0, f1);
        f0 = _mm256_and_si256(f0, mask);

        f1 = _mm256_sub_epi8(f0, bound);
        f0 = _mm256_sub_epi8(eta, f0);
        good = _mm256_movemask_epi8(f1);

        g0 = _mm256_castsi256_si128(f0);
        g1 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xFF]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        f2 = _mm256_mulhrs_epi16(f1, v);
        f2 = _mm256_mullo_epi16(f2, p);
        f1 = _mm256_add_epi32(f1, f2);
        _mm256_storeu_si256((__m256i *) &r[ctr], f1);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) {
            break;
        }
        g0 = _mm_bsrli_si128(g0, 8);
        g1 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xFF]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        f2 = _mm256_mulhrs_epi16(f1, v);
        f2 = _mm256_mullo_epi16(f2, p);
        f1 = _mm256_add_epi32(f1, f2);
        _mm256_storeu_si256((__m256i *) &r[ctr], f1);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) {
            break;
        }
        g0 = _mm256_extracti128_si256(f0, 1);
        g1 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xFF]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        f2 = _mm256_mulhrs_epi16(f1, v);
        f2 = _mm256_mullo_epi16(f2, p);
        f1 = _mm256_add_epi32(f1, f2);
        _mm256_storeu_si256((__m256i *) &r[ctr], f1);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) {
            break;
        }
        g0 = _mm_bsrli_si128(g0, 8);
        g1 = _mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good]);
        g1 = _mm_shuffle_epi8(g0, g1);
        f1 = _mm256_cvtepi8_epi32(g1);
        f2 = _mm256_mulhrs_epi16(f1, v);
        f2 = _mm256_mullo_epi16(f2, p);
        f1 = _mm256_add_epi32(f1, f2);
        _mm256_storeu_si256((__m256i *) &r[ctr], f1);
        ctr += _mm_popcnt_u32(good);
        pos += 4;
    }

    uint32_t t0, t1;
    while (ctr < N && pos < REJ_UNIFORM_ETA_BUFLEN) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

        if (t0 < 15) {
            t0 = t0 - (205 * t0 >> 10) * 5;
            r[ctr++] = 2 - t0;
        }
        if (t1 < 15 && ctr < N) {
            t1 = t1 - (205 * t1 >> 10) * 5;
            r[ctr++] = 2 - t1;
        }
    }

    return ctr;
}



static uint32_t rej_eta_final(int32_t *restrict r, uint32_t ctr, const uint8_t *buf) {
    int pos = 128;
    uint32_t good;
    __m128i bl, bh;
    __m256i c, tmp;
    __m256i b;
    const __m256i eta = _mm256_set1_epi8(ETA);
    const __m256i bound = _mm256_set1_epi8(0x0f);
    const __m256i num13 = _mm256_set1_epi16(13);
    const __m256i num5 = _mm256_set1_epi16(5);
    const __m256i mask = _mm256_set1_epi8(0xc0);

    while (ctr <= N - 8 && pos <= REJ_UNIFORM_ETA_BUFLEN - 16) {
        b = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) &buf[pos]));
        c = _mm256_slli_epi16(b, 4);
        c = (b | c) & bound;

        // cmp
        b = _mm256_sub_epi8(c, bound);
        good = _mm256_movemask_epi8(b);

        b = _mm256_mullo_epi16(c, num13);
        b &= mask;
        b = _mm256_srli_epi16(b, 6);
        b = _mm256_mullo_epi16(b, num5);
        b = _mm256_sub_epi8(c, b);
        b = _mm256_sub_epi8(eta, b);

        bl = _mm256_castsi256_si128(b);
        bh = _mm256_extracti128_si256(b, 1);

        tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
        c = _mm256_cvtepi8_epi32(bl);
        c = _mm256_permutevar8x32_epi32(c, tmp);
        _mm256_storeu_si256((__m256i *) &r[ctr], c);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) break;

        tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
        c = _mm256_cvtepi8_epi32(_mm_srli_si128(bl, 8));
        c = _mm256_permutevar8x32_epi32(c, tmp);
        _mm256_storeu_si256((__m256i *) &r[ctr], c);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) break;

        tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
        c = _mm256_cvtepi8_epi32(bh);
        c = _mm256_permutevar8x32_epi32(c, tmp);
        _mm256_storeu_si256((__m256i *) &r[ctr], c);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) break;

        tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
        c = _mm256_cvtepi8_epi32(_mm_srli_si128(bh, 8));
        c = _mm256_permutevar8x32_epi32(c, tmp);
        _mm256_storeu_si256((__m256i *) &r[ctr], c);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 4;

    }

    uint32_t t0, t1;
    while (ctr < N && pos < REJ_UNIFORM_ETA_BUFLEN) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

        if (t0 < 15) {
            t0 = t0 - (205 * t0 >> 10) * 5;
            r[ctr++] = 2 - t0;
        }
        if (t1 < 15 && ctr < N) {
            t1 = t1 - (205 * t1 >> 10) * 5;
            r[ctr++] = 2 - t1;
        }
    }

    return ctr;
}

unsigned int XURQ_AVX2_rej_eta_avx(int32_t *restrict r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]) {
    unsigned int ctr, pos;
    uint32_t good;

    __m256i c, tmp;
    __m256i b;

    const __m256i eta = _mm256_set1_epi8(ETA);
    const __m256i bound = _mm256_set1_epi8(0x0f);
    const __m256i num13 = _mm256_set1_epi16(13);
    const __m256i num5 = _mm256_set1_epi16(5);
    const __m256i mask = _mm256_set1_epi8(0xc0);

    __m128i bl, bh;

    ctr = pos = 0;

    for (int i = 0; i < 8; ++i) {
        b = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) &buf[pos]));
        c = _mm256_slli_epi16(b, 4);
        c = (b | c) & bound;

        // cmp
        b = _mm256_sub_epi8(c, bound);
        good = _mm256_movemask_epi8(b);

        b = _mm256_mullo_epi16(c, num13);
        b &= mask;
        b = _mm256_srli_epi16(b, 6);
        b = _mm256_mullo_epi16(b, num5);
        b = _mm256_sub_epi8(c, b);
        b = _mm256_sub_epi8(eta, b);

        bl = _mm256_castsi256_si128(b);
        bh = _mm256_extracti128_si256(b, 1);

        tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
        c = _mm256_cvtepi8_epi32(bl);
        c = _mm256_permutevar8x32_epi32(c, tmp);
        _mm256_storeu_si256((__m256i *) &r[ctr], c);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;

        tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
        c = _mm256_cvtepi8_epi32(_mm_srli_si128(bl, 8));
        c = _mm256_permutevar8x32_epi32(c, tmp);
        _mm256_storeu_si256((__m256i *) &r[ctr], c);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;


        tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
        c = _mm256_cvtepi8_epi32(bh);
        c = _mm256_permutevar8x32_epi32(c, tmp);
        _mm256_storeu_si256((__m256i *) &r[ctr], c);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;


        tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
        c = _mm256_cvtepi8_epi32(_mm_srli_si128(bh, 8));
        c = _mm256_permutevar8x32_epi32(c, tmp);
        _mm256_storeu_si256((__m256i *) &r[ctr], c);
        ctr += _mm_popcnt_u32(good & 0xFF);
        good >>= 8;
        pos += 16;
    }

    ctr = rej_eta_final(r,ctr, buf);

    return ctr;
}

unsigned int XURQ_AVX2_rej_eta_avx_with_s3(int32_t *restrict r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]) {
    unsigned int ctr, pos;
    uint32_t good;

    __m256i c, tmp;
    __m256i b;

    const __m256i eta = _mm256_set1_epi8(ETA);
    const __m256i bound = _mm256_set1_epi8(0x0f);
    const __m256i num13 = _mm256_set1_epi16(13);
    const __m256i num5 = _mm256_set1_epi16(5);
    const __m256i mask = _mm256_set1_epi8(0xc0);

    __m128i bl, bh;

    ctr = pos = 0;

    for (int i = 0; i < 8; ++i) {
        b = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) &buf[pos]));
        c = _mm256_slli_epi16(b, 4);
        c = (b | c) & bound;

        // cmp
        b = _mm256_sub_epi8(c, bound);
        good = _mm256_movemask_epi8(b);

        b = _mm256_mullo_epi16(c, num13);
        b &= mask;
        b = _mm256_srli_epi16(b, 6);
        b = _mm256_mullo_epi16(b, num5);
        b = _mm256_sub_epi8(c, b);
        b = _mm256_sub_epi8(eta, b);

        bl = _mm256_castsi256_si128(b);
        bh = _mm256_extracti128_si256(b, 1);

        c = _mm256_cvtepi8_epi32(bl);
        if ((good & 0xff) == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
            c = _mm256_permutevar8x32_epi32(c, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += _mm_popcnt_u32(good & 0xFF);
        }

        good >>= 8;


        c = _mm256_cvtepi8_epi32(_mm_srli_si128(bl, 8));
        if ((good & 0xff) == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
            c = _mm256_permutevar8x32_epi32(c, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += _mm_popcnt_u32(good & 0xFF);
        }
        good >>= 8;


        c = _mm256_cvtepi8_epi32(bh);
        if ((good & 0xff) == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
            c = _mm256_permutevar8x32_epi32(c, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += _mm_popcnt_u32(good & 0xFF);
        }
        good >>= 8;


        c = _mm256_cvtepi8_epi32(_mm_srli_si128(bh, 8));
        if ((good & 0xff) == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
            c = _mm256_permutevar8x32_epi32(c, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += _mm_popcnt_u32(good & 0xFF);
        }
        good >>= 8;
        pos += 16;
    }

    while (ctr <= N - 8 && pos <= REJ_UNIFORM_ETA_BUFLEN - 16) {
        b = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) &buf[pos]));
        c = _mm256_slli_epi16(b, 4);
        c = (b | c) & bound;

        // cmp
        b = _mm256_sub_epi8(c, bound);
        good = _mm256_movemask_epi8(b);

        b = _mm256_mullo_epi16(c, num13);
        b &= mask;
        b = _mm256_srli_epi16(b, 6);
        b = _mm256_mullo_epi16(b, num5);
        b = _mm256_sub_epi8(c, b);
        b = _mm256_sub_epi8(eta, b);

        bl = _mm256_castsi256_si128(b);
        bh = _mm256_extracti128_si256(b, 1);

        c = _mm256_cvtepi8_epi32(bl);
        if ((good & 0xff) == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
            c = _mm256_permutevar8x32_epi32(c, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += _mm_popcnt_u32(good & 0xFF);
        }

        good >>= 8;
        pos += 4;

        if (ctr > N - 8) break;

        c = _mm256_cvtepi8_epi32(_mm_srli_si128(bl, 8));
        if ((good & 0xff) == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
            c = _mm256_permutevar8x32_epi32(c, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += _mm_popcnt_u32(good & 0xFF);
        }
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) break;

        c = _mm256_cvtepi8_epi32(bh);
        if ((good & 0xff) == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
            c = _mm256_permutevar8x32_epi32(c, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += _mm_popcnt_u32(good & 0xFF);
        }
        good >>= 8;
        pos += 4;

        if (ctr > N - 8) break;

        c = _mm256_cvtepi8_epi32(_mm_srli_si128(bh, 8));
        if ((good & 0xff) == 0xff) {
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += 8;
        } else {
            tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good & 0xff]));
            c = _mm256_permutevar8x32_epi32(c, tmp);
            _mm256_storeu_si256((__m256i *) &r[ctr], c);
            ctr += _mm_popcnt_u32(good & 0xFF);
        }
        good >>= 8;
        pos += 4;

    }

    uint32_t t0, t1;
    while (ctr < N && pos < REJ_UNIFORM_ETA_BUFLEN) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

        if (t0 < 15) {
            t0 = t0 - (205 * t0 >> 10) * 5;
            r[ctr++] = 2 - t0;
        }
        if (t1 < 15 && ctr < N) {
            t1 = t1 - (205 * t1 >> 10) * 5;
            r[ctr++] = 2 - t1;
        }
    }

    return ctr;
}



void XURQ_AVX2_rej_eta_avx_with_s2_vertical(int32_t *restrict r0,
                                            int32_t *restrict r1,
                                            int32_t *restrict r2,
                                            int32_t *restrict r3,
                                            uint32_t *ctr,
                                            const uint8_t *buf0,
                                            const uint8_t *buf1,
                                            const uint8_t *buf2,
                                            const uint8_t *buf3) {
    unsigned int pos;
    uint32_t good0, good1, good2, good3;

    __m128i bl0, bl1, bl2, bl3;
    __m128i bh0, bh1, bh2, bh3;

    __m256i c0, c1, c2, c3, tmp0, tmp1, tmp2, tmp3;
    __m256i b0, b1, b2, b3;

    const __m256i eta = _mm256_set1_epi8(ETA);
    const __m256i bound = _mm256_set1_epi8(0x0f);
    const __m256i num13 = _mm256_set1_epi16(13);
    const __m256i num5 = _mm256_set1_epi16(5);
    const __m256i mask = _mm256_set1_epi8(0xc0);


    ctr[0] = pos = 0;

    for (int i = 0; i < 8; ++i) {
        b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) &buf0[pos]));
        b1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) &buf1[pos]));
        b2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) &buf2[pos]));
        b3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) &buf3[pos]));

        c0 = _mm256_slli_epi16(b0, 4);
        c1 = _mm256_slli_epi16(b1, 4);
        c2 = _mm256_slli_epi16(b2, 4);
        c3 = _mm256_slli_epi16(b3, 4);

        c0 = (b0 | c0) & bound;
        c1 = (b1 | c1) & bound;
        c2 = (b2 | c2) & bound;
        c3 = (b3 | c3) & bound;

        b0 = _mm256_sub_epi8(c0, bound);
        b1 = _mm256_sub_epi8(c1, bound);
        b2 = _mm256_sub_epi8(c2, bound);
        b3 = _mm256_sub_epi8(c3, bound);

        good0 = _mm256_movemask_epi8(b0);
        good1 = _mm256_movemask_epi8(b1);
        good2 = _mm256_movemask_epi8(b2);
        good3 = _mm256_movemask_epi8(b3);

        b0 = _mm256_mullo_epi16(c0, num13);
        b1 = _mm256_mullo_epi16(c1, num13);
        b2 = _mm256_mullo_epi16(c2, num13);
        b3 = _mm256_mullo_epi16(c3, num13);

        b0 &= mask;
        b1 &= mask;
        b2 &= mask;
        b3 &= mask;

        b0 = _mm256_srli_epi16(b0, 6);
        b1 = _mm256_srli_epi16(b1, 6);
        b2 = _mm256_srli_epi16(b2, 6);
        b3 = _mm256_srli_epi16(b3, 6);

        b0 = _mm256_mullo_epi16(b0, num5);
        b1 = _mm256_mullo_epi16(b1, num5);
        b2 = _mm256_mullo_epi16(b2, num5);
        b3 = _mm256_mullo_epi16(b3, num5);

        b0 = _mm256_sub_epi8(c0, b0);
        b1 = _mm256_sub_epi8(c1, b1);
        b2 = _mm256_sub_epi8(c2, b2);
        b3 = _mm256_sub_epi8(c3, b3);

        bl0 = _mm256_castsi256_si128(b0);
        bl1 = _mm256_castsi256_si128(b1);
        bl2 = _mm256_castsi256_si128(b2);
        bl3 = _mm256_castsi256_si128(b3);

        bh0 = _mm256_extracti128_si256(b0, 1);
        bh1 = _mm256_extracti128_si256(b1, 1);
        bh2 = _mm256_extracti128_si256(b2, 1);
        bh3 = _mm256_extracti128_si256(b3, 1);

        tmp0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0 & 0xff]));
        tmp1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good1 & 0xff]));
        tmp2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good2 & 0xff]));
        tmp3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good3 & 0xff]));
        c0 = _mm256_cvtepi8_epi32(bl0);
        c1 = _mm256_cvtepi8_epi32(bl1);
        c2 = _mm256_cvtepi8_epi32(bl2);
        c3 = _mm256_cvtepi8_epi32(bl3);
        c0 = _mm256_permutevar8x32_epi32(c0, tmp0);
        c1 = _mm256_permutevar8x32_epi32(c1, tmp1);
        c2 = _mm256_permutevar8x32_epi32(c2, tmp2);
        c3 = _mm256_permutevar8x32_epi32(c3, tmp3);
        _mm256_storeu_si256((__m256i *) &r0[ctr[0]], c0);
        _mm256_storeu_si256((__m256i *) &r1[ctr[1]], c1);
        _mm256_storeu_si256((__m256i *) &r2[ctr[2]], c2);
        _mm256_storeu_si256((__m256i *) &r3[ctr[3]], c3);
        ctr[0] += _mm_popcnt_u32(good0 & 0xFF);
        ctr[1] += _mm_popcnt_u32(good1 & 0xFF);
        ctr[2] += _mm_popcnt_u32(good2 & 0xFF);
        ctr[3] += _mm_popcnt_u32(good3 & 0xFF);
        good0 >>= 8;
        good1 >>= 8;
        good2 >>= 8;
        good3 >>= 8;

        tmp0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0 & 0xff]));
        tmp1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good1 & 0xff]));
        tmp2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good2 & 0xff]));
        tmp3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good3 & 0xff]));
        c0 = _mm256_cvtepi8_epi32(_mm_srli_si128(bl0, 8));
        c1 = _mm256_cvtepi8_epi32(_mm_srli_si128(bl1, 8));
        c2 = _mm256_cvtepi8_epi32(_mm_srli_si128(bl2, 8));
        c3 = _mm256_cvtepi8_epi32(_mm_srli_si128(bl3, 8));
        c0 = _mm256_permutevar8x32_epi32(c0, tmp0);
        c1 = _mm256_permutevar8x32_epi32(c1, tmp1);
        c2 = _mm256_permutevar8x32_epi32(c2, tmp2);
        c3 = _mm256_permutevar8x32_epi32(c3, tmp3);
        _mm256_storeu_si256((__m256i *) &r0[ctr[0]], c0);
        _mm256_storeu_si256((__m256i *) &r1[ctr[1]], c1);
        _mm256_storeu_si256((__m256i *) &r2[ctr[2]], c2);
        _mm256_storeu_si256((__m256i *) &r3[ctr[3]], c3);
        ctr[0] += _mm_popcnt_u32(good0 & 0xFF);
        ctr[1] += _mm_popcnt_u32(good1 & 0xFF);
        ctr[2] += _mm_popcnt_u32(good2 & 0xFF);
        ctr[3] += _mm_popcnt_u32(good3 & 0xFF);
        good0 >>= 8;
        good1 >>= 8;
        good2 >>= 8;
        good3 >>= 8;

        tmp0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0 & 0xff]));
        tmp1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good1 & 0xff]));
        tmp2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good2 & 0xff]));
        tmp3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good3 & 0xff]));
        c0 = _mm256_cvtepi8_epi32(bh0);
        c1 = _mm256_cvtepi8_epi32(bh1);
        c2 = _mm256_cvtepi8_epi32(bh2);
        c3 = _mm256_cvtepi8_epi32(bh3);
        c0 = _mm256_permutevar8x32_epi32(c0, tmp0);
        c1 = _mm256_permutevar8x32_epi32(c1, tmp1);
        c2 = _mm256_permutevar8x32_epi32(c2, tmp2);
        c3 = _mm256_permutevar8x32_epi32(c3, tmp3);
        _mm256_storeu_si256((__m256i *) &r0[ctr[0]], c0);
        _mm256_storeu_si256((__m256i *) &r1[ctr[1]], c1);
        _mm256_storeu_si256((__m256i *) &r2[ctr[2]], c2);
        _mm256_storeu_si256((__m256i *) &r3[ctr[3]], c3);
        ctr[0] += _mm_popcnt_u32(good0 & 0xFF);
        ctr[1] += _mm_popcnt_u32(good1 & 0xFF);
        ctr[2] += _mm_popcnt_u32(good2 & 0xFF);
        ctr[3] += _mm_popcnt_u32(good3 & 0xFF);
        good0 >>= 8;
        good1 >>= 8;
        good2 >>= 8;
        good3 >>= 8;

        tmp0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good0 & 0xff]));
        tmp1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good1 & 0xff]));
        tmp2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good2 & 0xff]));
        tmp3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &DILITHIUM2_AVX2_idxlut[good3 & 0xff]));
        c0 = _mm256_cvtepi8_epi32(_mm_srli_si128(bh0, 8));
        c1 = _mm256_cvtepi8_epi32(_mm_srli_si128(bh1, 8));
        c2 = _mm256_cvtepi8_epi32(_mm_srli_si128(bh2, 8));
        c3 = _mm256_cvtepi8_epi32(_mm_srli_si128(bh3, 8));
        c0 = _mm256_permutevar8x32_epi32(c0, tmp0);
        c1 = _mm256_permutevar8x32_epi32(c1, tmp0);
        c2 = _mm256_permutevar8x32_epi32(c2, tmp0);
        c3 = _mm256_permutevar8x32_epi32(c3, tmp0);
        _mm256_storeu_si256((__m256i *) &r0[ctr[0]], c0);
        _mm256_storeu_si256((__m256i *) &r1[ctr[1]], c1);
        _mm256_storeu_si256((__m256i *) &r2[ctr[2]], c2);
        _mm256_storeu_si256((__m256i *) &r3[ctr[3]], c3);
        ctr[0] += _mm_popcnt_u32(good0 & 0xFF);
        ctr[1] += _mm_popcnt_u32(good1 & 0xFF);
        ctr[2] += _mm_popcnt_u32(good2 & 0xFF);
        ctr[3] += _mm_popcnt_u32(good3 & 0xFF);


        pos += 16;
    }

    ctr[0] = rej_eta_final(r0, ctr[0], buf0);
    ctr[1] = rej_eta_final(r1, ctr[1], buf1);
    ctr[2] = rej_eta_final(r2, ctr[2], buf2);
    ctr[3] = rej_eta_final(r3, ctr[3], buf3);

}



#define SAMPLE(n) \
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
f8  = _mm256_mulhrs_epi16(f4, v);\
f9  = _mm256_mulhrs_epi16(f5, v);\
f10 = _mm256_mulhrs_epi16(f6, v);\
f11 = _mm256_mulhrs_epi16(f7, v);\
\
f8  = _mm256_mullo_epi16(f8 , p);\
f9  = _mm256_mullo_epi16(f9 , p);\
f10 = _mm256_mullo_epi16(f10, p);\
f11 = _mm256_mullo_epi16(f11, p);\
\
f4 = _mm256_add_epi32(f4, f8 );\
f5 = _mm256_add_epi32(f5, f9 );\
f6 = _mm256_add_epi32(f6, f10);\
f7 = _mm256_add_epi32(f7, f11);\
\
_mm256_storeu_si256((__m256i *) &r[ctr], f4);\
ctr += _mm_popcnt_u32(good##n & 0xFF);\
_mm256_storeu_si256((__m256i *) &r[ctr], f5);\
ctr += _mm_popcnt_u32((good##n >> 8) & 0xFF);\
_mm256_storeu_si256((__m256i *) &r[ctr], f6);\
ctr += _mm_popcnt_u32((good##n >> 16) & 0xFF);\
_mm256_storeu_si256((__m256i *) &r[ctr], f7);\
ctr += _mm_popcnt_u32((good##n >> 24) & 0xFF);\
\



unsigned int XURQ_AVX2_rej_eta_avx_with_s2_horizontal(int32_t *restrict r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]) {
    unsigned int ctr, pos;
    uint32_t good0, good1, good2, good3;
    __m256i f0, f1, f2, f3, f4, f5, f6, f7;
    __m256i f8, f9, f10, f11;
    __m128i g0, g1, g2, g3;
    __m128i d0, d1, d2, d3;
    const __m256i mask = _mm256_set1_epi8(15);
    const __m256i eta = _mm256_set1_epi8(ETA);
    const __m256i bound = mask;
    const __m256i v = _mm256_set1_epi32(-6560);
    const __m256i p = _mm256_set1_epi32(5);

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

    f0 = _mm256_and_si256(f0, mask);
    f1 = _mm256_and_si256(f1, mask);
    f2 = _mm256_and_si256(f2, mask);
    f3 = _mm256_and_si256(f3, mask);

    f4 = _mm256_sub_epi8(f0, bound);
    f5 = _mm256_sub_epi8(f1, bound);
    f6 = _mm256_sub_epi8(f2, bound);
    f7 = _mm256_sub_epi8(f3, bound);

    f0 = _mm256_sub_epi8(eta, f0);
    f1 = _mm256_sub_epi8(eta, f1);
    f2 = _mm256_sub_epi8(eta, f2);
    f3 = _mm256_sub_epi8(eta, f3);

    good0 = _mm256_movemask_epi8(f4);
    good1 = _mm256_movemask_epi8(f5);
    good2 = _mm256_movemask_epi8(f6);
    good3 = _mm256_movemask_epi8(f7);

    SAMPLE(0)
    SAMPLE(1)
    SAMPLE(2)
    SAMPLE(3)

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

    f0 = _mm256_and_si256(f0, mask);
    f1 = _mm256_and_si256(f1, mask);
    f2 = _mm256_and_si256(f2, mask);
    f3 = _mm256_and_si256(f3, mask);

    f4 = _mm256_sub_epi8(f0, bound);
    f5 = _mm256_sub_epi8(f1, bound);
    f6 = _mm256_sub_epi8(f2, bound);
    f7 = _mm256_sub_epi8(f3, bound);

    f0 = _mm256_sub_epi8(eta, f0);
    f1 = _mm256_sub_epi8(eta, f1);
    f2 = _mm256_sub_epi8(eta, f2);
    f3 = _mm256_sub_epi8(eta, f3);

    good0 = _mm256_movemask_epi8(f4);
    good1 = _mm256_movemask_epi8(f5);
    good2 = _mm256_movemask_epi8(f6);
    good3 = _mm256_movemask_epi8(f7);

    SAMPLE(0)
    SAMPLE(1)
    SAMPLE(2)
    SAMPLE(3)

    ctr = rej_eta_final(r,ctr,buf);

    return ctr;
}


#define XU_SAMPLE(x) \
x = _mm256_srli_epi32(x, 6);\
x = x & mask;\
x = _mm256_mul_epu32(x,num5);\
\



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



