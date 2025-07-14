#include "params.h"
#include "rejsample.h"
#include "symmetric.h"
#include "align.h"
#include "poly.h"
#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>

const uint8_t PQCLEAN_DILITHIUM2_AVX2_idxlut[256][8] = {
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

unsigned int PQCLEAN_DILITHIUM2_AVX2_rej_uniform_avx(int32_t *restrict r, const uint8_t buf[REJ_UNIFORM_BUFLEN + 8]) {
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
    while (pos <= REJ_UNIFORM_BUFLEN - 24) {
        d = _mm256_loadu_si256((__m256i *) &buf[pos]);
        d = _mm256_permute4x64_epi64(d, 0x94);
        d = _mm256_shuffle_epi8(d, idx8);
        d = _mm256_and_si256(d, mask);
        pos += 24;

        tmp = _mm256_sub_epi32(d, bound);
        good = _mm256_movemask_ps((__m256) tmp);
        tmp = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) &PQCLEAN_DILITHIUM2_AVX2_idxlut[good]));
        d = _mm256_permutevar8x32_epi32(d, tmp);

        _mm256_storeu_si256((__m256i *) &r[ctr], d);
        ctr += _mm_popcnt_u32(good);

        if (ctr > N - 8) {
            break;
        }
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
        g1 = _mm_loadl_epi64((__m128i *) &PQCLEAN_DILITHIUM2_AVX2_idxlut[good & 0xFF]);
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
        g1 = _mm_loadl_epi64((__m128i *) &PQCLEAN_DILITHIUM2_AVX2_idxlut[good & 0xFF]);
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
        g1 = _mm_loadl_epi64((__m128i *) &PQCLEAN_DILITHIUM2_AVX2_idxlut[good & 0xFF]);
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
        g1 = _mm_loadl_epi64((__m128i *) &PQCLEAN_DILITHIUM2_AVX2_idxlut[good]);
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


ALIGN(64) static const uint8_t idx2[64] = {
        0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80,
        0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80,
        0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80,
        0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80,
};


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
    // int ctr[8] = {0};
    uint16_t g0, g1, g2, g3, g4, g5, g6, g7;
    const __m512i bound = _mm512_set1_epi32(Q);
    const __m512i mask = _mm512_set1_epi32(0x7FFFFF);
    const __m512i index1 = _mm512_setr_epi32(0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11);
    const __m512i index2 = _mm512_load_si512(idx2);
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

        // d0 = _mm512_maskz_permutexvar_epi32(0x7777, index1, d0);
        // d1 = _mm512_maskz_permutexvar_epi32(0x7777, index1, d1);
        // d2 = _mm512_maskz_permutexvar_epi32(0x7777, index1, d2);
        // d3 = _mm512_maskz_permutexvar_epi32(0x7777, index1, d3);
        // d4 = _mm512_maskz_permutexvar_epi32(0x7777, index1, d4);
        // d5 = _mm512_maskz_permutexvar_epi32(0x7777, index1, d5);
        // d6 = _mm512_maskz_permutexvar_epi32(0x7777, index1, d6);
        // d7 = _mm512_maskz_permutexvar_epi32(0x7777, index1, d7);
        //
        // d0 = _mm512_shuffle_epi8(d0, index2);
        // d1 = _mm512_shuffle_epi8(d1, index2);
        // d2 = _mm512_shuffle_epi8(d2, index2);
        // d3 = _mm512_shuffle_epi8(d3, index2);
        // d4 = _mm512_shuffle_epi8(d4, index2);
        // d5 = _mm512_shuffle_epi8(d5, index2);
        // d6 = _mm512_shuffle_epi8(d6, index2);
        // d7 = _mm512_shuffle_epi8(d7, index2);

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


void XURQ_AVX512_rej_eta8x(int32_t *r0,
                           int32_t *r1,
                           int32_t *r2,
                           int32_t *r3,
                           int32_t *r4,
                           int32_t *r5,
                           int32_t *r6,
                           int32_t *r7,
                           uint32_t ctr[8],
                           uint8_t buf[8][136]) {
    uint32_t pos = 0;
    uint64_t g0, g1, g2, g3, g4, g5, g6, g7;
    const __m512i bound = _mm512_set1_epi8(0x0f);
    __m512i c0, c1, c2, c3, c4, c5, c6, c7;
    __m512i b0, b1, b2, b3, b4, b5, b6, b7;

    const __m512i mask = _mm512_set1_epi8(0xc0);
    const __m512i num13 = _mm512_set1_epi16(13);
    const __m512i num5 = _mm512_set1_epi16(5);
    const __m512i eta = _mm512_set1_epi8(2);


    for (int i = 0; i < 4; ++i) {
        b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[0][pos]));
        b1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[1][pos]));
        b2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[2][pos]));
        b3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[3][pos]));
        b4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[4][pos]));
        b5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[5][pos]));
        b6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[6][pos]));
        b7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(&buf[7][pos]));

        c0 = _mm512_slli_epi16(b0, 4);
        c1 = _mm512_slli_epi16(b1, 4);
        c2 = _mm512_slli_epi16(b2, 4);
        c3 = _mm512_slli_epi16(b3, 4);
        c4 = _mm512_slli_epi16(b4, 4);
        c5 = _mm512_slli_epi16(b5, 4);
        c6 = _mm512_slli_epi16(b6, 4);
        c7 = _mm512_slli_epi16(b7, 4);

        c0 = (b0 | c0) & bound;
        c1 = (b1 | c1) & bound;
        c2 = (b2 | c2) & bound;
        c3 = (b3 | c3) & bound;
        c4 = (b4 | c4) & bound;
        c5 = (b5 | c5) & bound;
        c6 = (b6 | c6) & bound;
        c7 = (b7 | c7) & bound;

        g0 = _mm512_cmp_epi8_mask(c0, bound, 1);
        g1 = _mm512_cmp_epi8_mask(c1, bound, 1);
        g2 = _mm512_cmp_epi8_mask(c2, bound, 1);
        g3 = _mm512_cmp_epi8_mask(c3, bound, 1);
        g4 = _mm512_cmp_epi8_mask(c4, bound, 1);
        g5 = _mm512_cmp_epi8_mask(c5, bound, 1);
        g6 = _mm512_cmp_epi8_mask(c6, bound, 1);
        g7 = _mm512_cmp_epi8_mask(c7, bound, 1);

        b0 = _mm512_mullo_epi16(c0, num13);
        b1 = _mm512_mullo_epi16(c1, num13);
        b2 = _mm512_mullo_epi16(c2, num13);
        b3 = _mm512_mullo_epi16(c3, num13);
        b4 = _mm512_mullo_epi16(c4, num13);
        b5 = _mm512_mullo_epi16(c5, num13);
        b6 = _mm512_mullo_epi16(c6, num13);
        b7 = _mm512_mullo_epi16(c7, num13);

        b0 &= mask;
        b1 &= mask;
        b2 &= mask;
        b3 &= mask;
        b4 &= mask;
        b5 &= mask;
        b6 &= mask;
        b7 &= mask;

        b0 = _mm512_srli_epi16(b0, 6);
        b1 = _mm512_srli_epi16(b1, 6);
        b2 = _mm512_srli_epi16(b2, 6);
        b3 = _mm512_srli_epi16(b3, 6);
        b4 = _mm512_srli_epi16(b4, 6);
        b5 = _mm512_srli_epi16(b5, 6);
        b6 = _mm512_srli_epi16(b6, 6);
        b7 = _mm512_srli_epi16(b7, 6);

        b0 = _mm512_mullo_epi16(b0, num5);
        b1 = _mm512_mullo_epi16(b1, num5);
        b2 = _mm512_mullo_epi16(b2, num5);
        b3 = _mm512_mullo_epi16(b3, num5);
        b4 = _mm512_mullo_epi16(b4, num5);
        b5 = _mm512_mullo_epi16(b5, num5);
        b6 = _mm512_mullo_epi16(b6, num5);
        b7 = _mm512_mullo_epi16(b7, num5);

        b0 = _mm512_sub_epi8(c0, b0);
        b1 = _mm512_sub_epi8(c1, b1);
        b2 = _mm512_sub_epi8(c2, b2);
        b3 = _mm512_sub_epi8(c3, b3);
        b4 = _mm512_sub_epi8(c4, b4);
        b5 = _mm512_sub_epi8(c5, b5);
        b6 = _mm512_sub_epi8(c6, b6);
        b7 = _mm512_sub_epi8(c7, b7);

        b0 = _mm512_sub_epi8(eta, b0);
        b1 = _mm512_sub_epi8(eta, b1);
        b2 = _mm512_sub_epi8(eta, b2);
        b3 = _mm512_sub_epi8(eta, b3);
        b4 = _mm512_sub_epi8(eta, b4);
        b5 = _mm512_sub_epi8(eta, b5);
        b6 = _mm512_sub_epi8(eta, b6);
        b7 = _mm512_sub_epi8(eta, b7);

        c0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b0, 0));
        c1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b1, 0));
        c2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b2, 0));
        c3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b3, 0));
        c4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b4, 0));
        c5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b5, 0));
        c6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b6, 0));
        c7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b7, 0));

        _mm512_mask_compressstoreu_epi32(r0 + ctr[0], g0 & 0xffff, c0);
        _mm512_mask_compressstoreu_epi32(r1 + ctr[1], g1 & 0xffff, c1);
        _mm512_mask_compressstoreu_epi32(r2 + ctr[2], g2 & 0xffff, c2);
        _mm512_mask_compressstoreu_epi32(r3 + ctr[3], g3 & 0xffff, c3);
        _mm512_mask_compressstoreu_epi32(r4 + ctr[4], g4 & 0xffff, c4);
        _mm512_mask_compressstoreu_epi32(r5 + ctr[5], g5 & 0xffff, c5);
        _mm512_mask_compressstoreu_epi32(r6 + ctr[6], g6 & 0xffff, c6);
        _mm512_mask_compressstoreu_epi32(r7 + ctr[7], g7 & 0xffff, c7);

        ctr[0] += _mm_popcnt_u64(g0 & 0xffff);
        ctr[1] += _mm_popcnt_u64(g1 & 0xffff);
        ctr[2] += _mm_popcnt_u64(g2 & 0xffff);
        ctr[3] += _mm_popcnt_u64(g3 & 0xffff);
        ctr[4] += _mm_popcnt_u64(g4 & 0xffff);
        ctr[5] += _mm_popcnt_u64(g5 & 0xffff);
        ctr[6] += _mm_popcnt_u64(g6 & 0xffff);
        ctr[7] += _mm_popcnt_u64(g7 & 0xffff);

        c0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b0, 1));
        c1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b1, 1));
        c2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b2, 1));
        c3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b3, 1));
        c4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b4, 1));
        c5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b5, 1));
        c6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b6, 1));
        c7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b7, 1));

        _mm512_mask_compressstoreu_epi32(r0 + ctr[0], (g0 >> 16) & 0xffff, c0);
        _mm512_mask_compressstoreu_epi32(r1 + ctr[1], (g1 >> 16) & 0xffff, c1);
        _mm512_mask_compressstoreu_epi32(r2 + ctr[2], (g2 >> 16) & 0xffff, c2);
        _mm512_mask_compressstoreu_epi32(r3 + ctr[3], (g3 >> 16) & 0xffff, c3);
        _mm512_mask_compressstoreu_epi32(r4 + ctr[4], (g4 >> 16) & 0xffff, c4);
        _mm512_mask_compressstoreu_epi32(r5 + ctr[5], (g5 >> 16) & 0xffff, c5);
        _mm512_mask_compressstoreu_epi32(r6 + ctr[6], (g6 >> 16) & 0xffff, c6);
        _mm512_mask_compressstoreu_epi32(r7 + ctr[7], (g7 >> 16) & 0xffff, c7);

        ctr[0] += _mm_popcnt_u64((g0 >> 16) & 0xffff);
        ctr[1] += _mm_popcnt_u64((g1 >> 16) & 0xffff);
        ctr[2] += _mm_popcnt_u64((g2 >> 16) & 0xffff);
        ctr[3] += _mm_popcnt_u64((g3 >> 16) & 0xffff);
        ctr[4] += _mm_popcnt_u64((g4 >> 16) & 0xffff);
        ctr[5] += _mm_popcnt_u64((g5 >> 16) & 0xffff);
        ctr[6] += _mm_popcnt_u64((g6 >> 16) & 0xffff);
        ctr[7] += _mm_popcnt_u64((g7 >> 16) & 0xffff);

        c0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b0, 2));
        c1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b1, 2));
        c2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b2, 2));
        c3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b3, 2));
        c4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b4, 2));
        c5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b5, 2));
        c6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b6, 2));
        c7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b7, 2));

        _mm512_mask_compressstoreu_epi32(r0 + ctr[0], (g0 >> 32) & 0xffff, c0);
        _mm512_mask_compressstoreu_epi32(r1 + ctr[1], (g1 >> 32) & 0xffff, c1);
        _mm512_mask_compressstoreu_epi32(r2 + ctr[2], (g2 >> 32) & 0xffff, c2);
        _mm512_mask_compressstoreu_epi32(r3 + ctr[3], (g3 >> 32) & 0xffff, c3);
        _mm512_mask_compressstoreu_epi32(r4 + ctr[4], (g4 >> 32) & 0xffff, c4);
        _mm512_mask_compressstoreu_epi32(r5 + ctr[5], (g5 >> 32) & 0xffff, c5);
        _mm512_mask_compressstoreu_epi32(r6 + ctr[6], (g6 >> 32) & 0xffff, c6);
        _mm512_mask_compressstoreu_epi32(r7 + ctr[7], (g7 >> 32) & 0xffff, c7);

        ctr[0] += _mm_popcnt_u64((g0 >> 32) & 0xffff);
        ctr[1] += _mm_popcnt_u64((g1 >> 32) & 0xffff);
        ctr[2] += _mm_popcnt_u64((g2 >> 32) & 0xffff);
        ctr[3] += _mm_popcnt_u64((g3 >> 32) & 0xffff);
        ctr[4] += _mm_popcnt_u64((g4 >> 32) & 0xffff);
        ctr[5] += _mm_popcnt_u64((g5 >> 32) & 0xffff);
        ctr[6] += _mm_popcnt_u64((g6 >> 32) & 0xffff);
        ctr[7] += _mm_popcnt_u64((g7 >> 32) & 0xffff);

        c0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b0, 3));
        c1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b1, 3));
        c2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b2, 3));
        c3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b3, 3));
        c4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b4, 3));
        c5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b5, 3));
        c6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b6, 3));
        c7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(b7, 3));

        _mm512_mask_compressstoreu_epi32(r0 + ctr[0], (g0 >> 48) & 0xffff, c0);
        _mm512_mask_compressstoreu_epi32(r1 + ctr[1], (g1 >> 48) & 0xffff, c1);
        _mm512_mask_compressstoreu_epi32(r2 + ctr[2], (g2 >> 48) & 0xffff, c2);
        _mm512_mask_compressstoreu_epi32(r3 + ctr[3], (g3 >> 48) & 0xffff, c3);
        _mm512_mask_compressstoreu_epi32(r4 + ctr[4], (g4 >> 48) & 0xffff, c4);
        _mm512_mask_compressstoreu_epi32(r5 + ctr[5], (g5 >> 48) & 0xffff, c5);
        _mm512_mask_compressstoreu_epi32(r6 + ctr[6], (g6 >> 48) & 0xffff, c6);
        _mm512_mask_compressstoreu_epi32(r7 + ctr[7], (g7 >> 48) & 0xffff, c7);

        ctr[0] += _mm_popcnt_u64((g0 >> 48) & 0xffff);
        ctr[1] += _mm_popcnt_u64((g1 >> 48) & 0xffff);
        ctr[2] += _mm_popcnt_u64((g2 >> 48) & 0xffff);
        ctr[3] += _mm_popcnt_u64((g3 >> 48) & 0xffff);
        ctr[4] += _mm_popcnt_u64((g4 >> 48) & 0xffff);
        ctr[5] += _mm_popcnt_u64((g5 >> 48) & 0xffff);
        ctr[6] += _mm_popcnt_u64((g6 >> 48) & 0xffff);
        ctr[7] += _mm_popcnt_u64((g7 >> 48) & 0xffff);

        // _mm512_mask_compressstoreu_epi8(r0 + ctr[0], g0, b0);
        // _mm512_mask_compressstoreu_epi8(r1 + ctr[1], g1, b1);
        // _mm512_mask_compressstoreu_epi8(r2 + ctr[2], g2, b2);
        // _mm512_mask_compressstoreu_epi8(r3 + ctr[3], g3, b3);
        // _mm512_mask_compressstoreu_epi8(r4 + ctr[4], g4, b4);
        // _mm512_mask_compressstoreu_epi8(r5 + ctr[5], g5, b5);
        // _mm512_mask_compressstoreu_epi8(r6 + ctr[6], g6, b6);
        // _mm512_mask_compressstoreu_epi8(r7 + ctr[7], g7, b7);

        // ctr[0] += _mm_popcnt_u64(g0);
        // ctr[1] += _mm_popcnt_u64(g1);
        // ctr[2] += _mm_popcnt_u64(g2);
        // ctr[3] += _mm_popcnt_u64(g3);
        // ctr[4] += _mm_popcnt_u64(g4);
        // ctr[5] += _mm_popcnt_u64(g5);
        // ctr[6] += _mm_popcnt_u64(g6);
        // ctr[7] += _mm_popcnt_u64(g7);

        pos += 32;
    }


    uint32_t t0, t1;
    int32_t *rr[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
    for (int i = 0; i < 8; ++i) {
        pos = 128;
        while (ctr[i] < N && pos < 136) {
            t0 = buf[i][pos] & 0x0F;
            t1 = buf[i][pos++] >> 4;

            if (t0 < 15) {
                t0 = t0 - (205 * t0 >> 10) * 5;
                rr[i][ctr[i]++] = 2 - t0;
            }
            if (t1 < 15 && ctr[i] < N) {
                t1 = t1 - (205 * t1 >> 10) * 5;
                rr[i][ctr[i]++] = 2 - t1;
            }
        }
    }

}


// void XURQ_AVX512_polyz_unpack4x(poly *r0, poly *r1, poly *r2, poly *r3,
//                                 const uint8_t buf[4][704]) {
//     __m512i a0, a1, a2, a3;
//     int pos = 0;
//     int ctr = 0;
//     const __m512i mask = _mm512_set1_epi32(0x3FFFF);
//     const __m512i gamma = _mm512_set1_epi32(GAMMA1);
//     const __m512i index3 = _mm512_setr_epi32(0, 1, 2, 2,
//                                              2, 3, 4, 4,
//                                              4, 5, 6, 6,
//                                              6, 7, 8, 8);
//     const __m512i index4 = _mm512_load_si512(idx4);
//     const __m512i index5 = _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6);
//
//
//     for (int i = 0; i < 16; ++i) {
//         a0 = _mm512_loadu_si512(&buf[0][pos]);
//         a1 = _mm512_loadu_si512(&buf[1][pos]);
//         a2 = _mm512_loadu_si512(&buf[2][pos]);
//         a3 = _mm512_loadu_si512(&buf[3][pos]);
//
//         a0 = _mm512_maskz_permutexvar_epi32(0x7777, index3, a0);
//         a1 = _mm512_maskz_permutexvar_epi32(0x7777, index3, a1);
//         a2 = _mm512_maskz_permutexvar_epi32(0x7777, index3, a2);
//         a3 = _mm512_maskz_permutexvar_epi32(0x7777, index3, a3);
//
//
//         a0 = _mm512_shuffle_epi8(a0, index4);
//         a1 = _mm512_shuffle_epi8(a1, index4);
//         a2 = _mm512_shuffle_epi8(a2, index4);
//         a3 = _mm512_shuffle_epi8(a3, index4);
//
//         a0 = _mm512_srlv_epi32(a0, index5);
//         a1 = _mm512_srlv_epi32(a1, index5);
//         a2 = _mm512_srlv_epi32(a2, index5);
//         a3 = _mm512_srlv_epi32(a3, index5);
//
//         a0 &= mask;
//         a1 &= mask;
//         a2 &= mask;
//         a3 &= mask;
//
//         a0 = _mm512_sub_epi32(gamma, a0);
//         a1 = _mm512_sub_epi32(gamma, a1);
//         a2 = _mm512_sub_epi32(gamma, a2);
//         a3 = _mm512_sub_epi32(gamma, a3);
//
//         _mm512_storeu_epi32(r0->coeffs + ctr, a0);
//         _mm512_storeu_epi32(r1->coeffs + ctr, a1);
//         _mm512_storeu_epi32(r2->coeffs + ctr, a2);
//         _mm512_storeu_epi32(r3->coeffs + ctr, a3);
//
//         pos += 36;
//         ctr += 16;
//     }
//
// }



ALIGN(64) static const uint8_t idx4[64] = {
        0, 1, 2, 0x80, 2, 3, 4, 0x80, 4, 5, 6, 0x80, 6, 7, 8, 0x80,
        1, 2, 3, 0x80, 3, 4, 5, 0x80, 5, 6, 7, 0x80, 7, 8, 9, 0x80,
        2, 3, 4, 0x80, 4, 5, 6, 0x80, 6, 7, 8, 0x80, 8, 9, 10, 0x80,
        3, 4, 5, 0x80, 5, 6, 7, 0x80, 7, 8, 9, 0x80, 9, 10, 11, 0x80,
};

ALIGN(64) static const uint8_t idx_gamma[64] = {
        0, 1, 2, 0x80, 2, 3, 4, 0x80, 4, 5, 6, 0x80, 6, 7, 8, 0x80,
        9, 10, 11, 0x80, 11, 12, 13, 0x80, 13, 14, 15, 0x80, 15, 16, 17, 0x80,
        18, 19, 20, 0x80, 20, 21, 22, 0x80, 22, 23, 24, 0x80, 24, 25, 26, 0x80,
        27, 28, 29, 0x80, 29, 30, 31, 0x80, 31, 32, 33, 0x80, 33, 34, 35, 0x80,
};

void XURQ_AVX512_polyz_unpack4x(poly *r0, poly *r1, poly *r2, poly *r3,
                                const uint8_t *buf0,
                                const uint8_t *buf1,
                                const uint8_t *buf2,
                                const uint8_t *buf3) {
    __m512i a0, a1, a2, a3;
    int pos = 0;
    int ctr = 0;
    const __m512i mask = _mm512_set1_epi32(0x3FFFF);
    const __m512i gamma = _mm512_set1_epi32(GAMMA1);
    const __m512i index = _mm512_load_si512(idx_gamma);
    const __m512i index3 = _mm512_setr_epi32(0, 1, 2, 0,
                                             2, 3, 4, 0,
                                             4, 5, 6, 0,
                                             6, 7, 8, 0);
    const __m512i index4 = _mm512_load_si512(idx4);
    const __m512i index5 = _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6);


    for (int i = 0; i < 16; ++i) {
        a0 = _mm512_loadu_si512(buf0 + pos);
        a1 = _mm512_loadu_si512(buf1 + pos);
        a2 = _mm512_loadu_si512(buf2 + pos);
        a3 = _mm512_loadu_si512(buf3 + pos);

        // a0 = _mm512_maskz_permutexvar_epi32(0x7777, index3, a0);
        // a1 = _mm512_maskz_permutexvar_epi32(0x7777, index3, a1);
        // a2 = _mm512_maskz_permutexvar_epi32(0x7777, index3, a2);
        // a3 = _mm512_maskz_permutexvar_epi32(0x7777, index3, a3);
        //
        // a0 = _mm512_shuffle_epi8(a0, index4);
        // a1 = _mm512_shuffle_epi8(a1, index4);
        // a2 = _mm512_shuffle_epi8(a2, index4);
        // a3 = _mm512_shuffle_epi8(a3, index4);

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

        _mm512_storeu_epi32(r0->coeffs + ctr, a0);
        _mm512_storeu_epi32(r1->coeffs + ctr, a1);
        _mm512_storeu_epi32(r2->coeffs + ctr, a2);
        _mm512_storeu_epi32(r3->coeffs + ctr, a3);

        pos += 36;
        ctr += 16;
    }

}