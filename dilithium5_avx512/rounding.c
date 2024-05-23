#include "consts.h"
#include "params.h"
#include "rejsample.h"
#include "rounding.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>



/*************************************************
* Name:        power2round
*
* Description: For finite field elements a, compute a0, a1 such that
*              a mod^+ Q = a1*2^D + a0 with -2^{D-1} < a0 <= 2^{D-1}.
*              Assumes a to be positive standard representative.
*
* Arguments:   - __m256i *a1: output array of length N/8 with high bits
*              - __m256i *a0: output array of length N/8 with low bits a0
*              - const __m256i *a: input array of length N/8
*
**************************************************/

void XURQ_AVX512_power2round_avx(__m512i *a1, __m512i *a0, const __m512i *a) {
    unsigned int i;
    __m512i f, f0, f1;
    const __m512i mask = _mm512_set1_epi32(-(1 << D));
    const __m512i half = _mm512_set1_epi32((1 << (D - 1)) - 1);

    for (i = 0; i < N / 16; ++i) {
        f = _mm512_load_si512(&a[i]);
        f1 = _mm512_add_epi32(f, half);
        f0 = _mm512_and_si512(f1, mask);
        f1 = _mm512_srli_epi32(f1, D);
        f0 = _mm512_sub_epi32(f, f0);
        _mm512_store_si512(&a1[i], f1);
        _mm512_store_si512(&a0[i], f0);
    }
}


/*************************************************
* Name:        decompose
*
* Description: For finite field element a, compute high and low parts a0, a1 such
*              that a mod^+ Q = a1*ALPHA + a0 with -ALPHA/2 < a0 <= ALPHA/2 except
*              if a1 = (Q-1)/ALPHA where we set a1 = 0 and
*              -ALPHA/2 <= a0 = a mod Q - Q < 0. Assumes a to be positive standard
*              representative.
*
* Arguments:   - __m256i *a1: output array of length N/8 with high parts
*              - __m256i *a0: output array of length N/8 with low parts a0
*              - const __m256i *a: input array of length N/8
*
**************************************************/


void XURQ_AVX512_decompose(__m512i *a1, __m512i *a0, const __m512i *a) {
    unsigned int i;
    __m512i f, f0, f1,t;
    const __m512i q = _mm512_set1_epi32(Q);
    const __m512i hq = _mm512_srli_epi32(q, 1);
    const __m512i num1025 = _mm512_set1_epi32(1025);
    const __m512i alpha = _mm512_set1_epi32(2 * GAMMA2);
    const __m512i num127 = _mm512_set1_epi32(127);
    const __m512i num2e21 = _mm512_set1_epi32(1 << 21);
    const __m512i num15 = _mm512_set1_epi32(15);
    const __m512i q12 = _mm512_set1_epi32((Q - 1) / 2);
    uint16_t mask;

    for (i = 0; i < N / 16; i++) {
        f = _mm512_load_si512(&a[i]);
        f1 = _mm512_add_epi32(f, num127);
        f1 = _mm512_srli_epi32(f1, 7);
        f1 = _mm512_mullo_epi32(f1, num1025);
        f1 = _mm512_add_epi32(f1, num2e21);
        f1 = _mm512_srli_epi32(f1, 22);
        f1 &= num15;

        f0 = _mm512_mullo_epi32(f1, alpha);
        f0 = _mm512_sub_epi32(f,f0);
        mask = _mm512_cmpgt_epi32_mask(f0, hq);
        f0 = _mm512_mask_sub_epi32(f0, mask,f0, q);

        _mm512_store_si512(&a1[i], f1);
        _mm512_store_si512(&a0[i], f0);
    }
}



/*************************************************
* Name:        make_hint
*
* Description: Compute indices of polynomial coefficients whose low bits
*              overflow into the high bits.
*
* Arguments:   - uint8_t *hint: hint array
*              - const __m256i *a0: low bits of input elements
*              - const __m256i *a1: high bits of input elements
*
* Returns number of overflowing low bits
**************************************************/


unsigned int make_hint_avx512(uint8_t hint[N], const __m512i *restrict a0, const __m512i *restrict a1) {
    unsigned int i, n = 0;
    __m512i f0, f1, g0;
    __m128i t;
    const __m512i low = _mm512_set1_epi32(-GAMMA2);
    const __m512i high = _mm512_set1_epi32(GAMMA2);
    const __m512i zero = _mm512_setzero_si512();
    const __m128i zero2 = _mm_setzero_si128();
    __mmask16 bad0, bad1;
    for (i = 0; i < N / 16; ++i) {
        f0 = _mm512_load_si512(&a0[i]);
        f1 = _mm512_load_si512(&a1[i]);
        g0 = _mm512_abs_epi32(f0);
        bad0 = _mm512_cmp_epi32_mask(g0, high, 6);
        bad1 = _mm512_cmp_epi32_mask(f0, low,0) & _mm512_cmp_epi32_mask(f1, zero,4);
        bad0 |= bad1;

        t = _mm_mask_set1_epi8(zero2,bad0,1);
        _mm_store_si128((__m128i *) (hint + 16 * i), t);
        n += _mm_popcnt_u32(bad0);
    }

    return n;
}

/*************************************************
* Name:        use_hint
*
* Description: Correct high parts according to hint.
*
* Arguments:   - __m256i *b: output array of length N/8 with corrected high parts
*              - const __m256i *a: input array of length N/8
*              - const __m256i *a: input array of length N/8 with hint bits
*
**************************************************/


void use_hint_avx512(__m512i *b, const __m512i *a, const __m512i *restrict hint) {
    unsigned int i;
    __m512i a0[N / 16];
    __m512i f, g, h, t;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i mask = _mm512_set1_epi32(15);

    XURQ_AVX512_decompose(b, a0, a);
    for (i = 0; i < N / 16; i++) {
        f = _mm512_load_si512(&a0[i]);
        g = _mm512_load_si512(&b[i]);
        h = _mm512_load_si512(&hint[i]);
//        t = _mm512_blendv_epi32(zero, h, f);
        t = _mm512_mask_blend_epi32(_mm512_cmp_epi32_mask(f,zero,4),zero,h);
        t = _mm512_slli_epi32(t, 1);
        h = _mm512_sub_epi32(h, t);
        g = _mm512_add_epi32(g, h);
        g = _mm512_and_si512(g, mask);
        _mm512_store_si512(&b[i], g);
    }
}