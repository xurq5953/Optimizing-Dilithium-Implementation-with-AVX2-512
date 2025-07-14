#include "align.h"
#include "consts.h"
#include "fips202x4.h"
#include "ntt.h"
#include "params.h"
#include "poly.h"
#include "rejsample.h"
#include "rounding.h"
#include "symmetric.h"
#include "fips202x8.h"
#include "align.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define DBENCH_START()
#define DBENCH_STOP(t)

#define _mm256_blendv_epi32(a, b, mask) \
    _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(a), \
                                         _mm256_castsi256_ps(b), \
                                         _mm256_castsi256_ps(mask)))

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_reduce
*
* Description: Inplace reduction of all coefficients of polynomial to
*              representative in [-6283009,6283007]. Assumes input
*              coefficients to be at most 2^31 - 2^22 - 1 in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_reduce(poly *a) {
    unsigned int i;
    __m256i f, g;
    const __m256i q = _mm256_load_si256(&PQCLEAN_DILITHIUM2_AVX2_qdata.vec[_8XQ / 8]);
    const __m256i off = _mm256_set1_epi32(1 << 22);


    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        g = _mm256_add_epi32(f, off);
        g = _mm256_srai_epi32(g, 23);
        g = _mm256_mullo_epi32(g, q);
        f = _mm256_sub_epi32(f, g);
        _mm256_store_si256(&a->vec[i], f);
    }


}

void XURQ_AVX512_poly_reduce(poly *a) {
    unsigned int i;
    __m512i f, g;
    const __m512i q = _mm512_load_si512(&qdata2[Q16X]);
    const __m512i off = _mm512_set1_epi32(1 << 22);


    for (i = 0; i < 16; i++) {
        f = _mm512_load_si512(&a->vec2[i]);
        g = _mm512_add_epi32(f, off);
        g = _mm512_srai_epi32(g, 23);
        g = _mm512_mullo_epi32(g, q);
        f = _mm512_sub_epi32(f, g);
        _mm512_store_si512(&a->vec2[i], f);
    }


}

/*************************************************
* Name:        poly_addq
*
* Description: For all coefficients of in/out polynomial add Q if
*              coefficient is negative.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_caddq(poly *a) {
    unsigned int i;
    __m256i f, g;
    const __m256i q = _mm256_load_si256(&PQCLEAN_DILITHIUM2_AVX2_qdata.vec[_8XQ / 8]);
    const __m256i zero = _mm256_setzero_si256();
    DBENCH_START();

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        g = _mm256_blendv_epi32(zero, q, f);
        f = _mm256_add_epi32(f, g);
        _mm256_store_si256(&a->vec[i], f);
    }

    DBENCH_STOP(*tred);
}

void XURQ_AVX512_poly_caddq(poly *a) {
    unsigned int i;
    __m512i f;
    const __m512i q = _mm512_load_si512(&qdata2[Q16X]);
    const __m512i zero = _mm512_setzero_si512();


    for (i = 0; i < N / 16; i++) {
        f = _mm512_load_si512(a->coeffs + (i * 16));
        f = _mm512_mask_add_epi32(f, _mm512_cmp_epi32_mask(f, zero, 1), f, q);
        _mm512_store_si512(a->coeffs + (i * 16), f);
    }

}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_add
*
* Description: Add polynomials. No modular reduction is performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first summand
*              - const poly *b: pointer to second summand
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_add(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    __m256i f, g;
    DBENCH_START();

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        g = _mm256_load_si256(&b->vec[i]);
        f = _mm256_add_epi32(f, g);
        _mm256_store_si256(&c->vec[i], f);
    }

    DBENCH_STOP(*tadd);
}

void XURQ_AVX512_poly_add(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    __m512i vec0, vec1;
    for (i = 0; i < N; i += 16) {
        vec0 = _mm512_load_si512((__m512i *) &a->coeffs[i]);
        vec1 = _mm512_load_si512((__m512i *) &b->coeffs[i]);
        vec0 = _mm512_add_epi32(vec0, vec1);
        _mm512_store_si512((__m512i *) &c->coeffs[i], vec0);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_sub
*
* Description: Subtract polynomials. No modular reduction is
*              performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial to be
*                               subtraced from first input polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_sub(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    __m256i f, g;
    DBENCH_START();

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        g = _mm256_load_si256(&b->vec[i]);
        f = _mm256_sub_epi32(f, g);
        _mm256_store_si256(&c->vec[i], f);
    }

    DBENCH_STOP(*tadd);
}

void XURQ_AVX512_poly_sub(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    __m512i vec0, vec1;
    for (i = 0; i < N; i += 16) {
        vec0 = _mm512_load_si512((__m512i *) &a->coeffs[i]);
        vec1 = _mm512_load_si512((__m512i *) &b->coeffs[i]);
        vec0 = _mm512_sub_epi32(vec0, vec1);
        _mm512_store_si512((__m512i *) &c->coeffs[i], vec0);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_shiftl
*
* Description: Multiply polynomial by 2^D without modular reduction. Assumes
*              input coefficients to be less than 2^{31-D} in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_shiftl(poly *a) {
    unsigned int i;
    __m256i f;
    DBENCH_START();

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        f = _mm256_slli_epi32(f, D);
        _mm256_store_si256(&a->vec[i], f);
    }

    DBENCH_STOP(*tmul);
}

void XURQ_AVX512_poly_shiftl(poly *a) {
    unsigned int i;
    __m512i f;

    for (i = 0; i < N / 16; i++) {
        f = _mm512_load_si512(a->coeffs + 16 * i);
        f = _mm512_slli_epi32(f, D);
        _mm512_store_si512(a->coeffs + 16 * i, f);
    }

}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_ntt
*
* Description: Inplace forward NTT. Coefficients can grow by up to
*              8*Q in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_ntt(poly *a) {
    DBENCH_START();

    PQCLEAN_DILITHIUM2_AVX2_ntt_avx(a->vec, PQCLEAN_DILITHIUM2_AVX2_qdata.vec);

    DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_invntt_tomont
*
* Description: Inplace inverse NTT and multiplication by 2^{32}.
*              Input coefficients need to be less than Q in absolute
*              value and output coefficients are again bounded by Q.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_invntt_tomont(poly *a) {
    DBENCH_START();

    PQCLEAN_DILITHIUM2_AVX2_invntt_avx(a->vec, PQCLEAN_DILITHIUM2_AVX2_qdata.vec);

    DBENCH_STOP(*tmul);
}

void PQCLEAN_DILITHIUM2_AVX2_poly_nttunpack(poly *a) {
    DBENCH_START();

    PQCLEAN_DILITHIUM2_AVX2_nttunpack_avx(a->vec);

    DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_pointwise_montgomery
*
* Description: Pointwise multiplication of polynomials in NTT domain
*              representation and multiplication of resulting polynomial
*              by 2^{-32}.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
    DBENCH_START();

    PQCLEAN_DILITHIUM2_AVX2_pointwise_avx(c->vec, a->vec, b->vec, PQCLEAN_DILITHIUM2_AVX2_qdata.vec);

    DBENCH_STOP(*tmul);
}

void XURQ_AVX512_poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
    pointwise_avx512(c->coeffs, a->coeffs, b->coeffs);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_power2round
*
* Description: For all coefficients c of the input polynomial,
*              compute c0, c1 such that c mod^+ Q = c1*2^D + c0
*              with -2^{D-1} < c0 <= 2^{D-1}. Assumes coefficients to be
*              positive standard representatives.
*
* Arguments:   - poly *a1: pointer to output polynomial with coefficients c1
*              - poly *a0: pointer to output polynomial with coefficients c0
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_power2round(poly *a1, poly *a0, const poly *a) {
    DBENCH_START();

    PQCLEAN_DILITHIUM2_AVX2_power2round_avx(a1->vec, a0->vec, a->vec);

    DBENCH_STOP(*tround);
}

void XURQ_AVX512_poly_power2round(poly *a1, poly *a0, const poly *a) {
    XURQ_AVX512_power2round_avx(a1->vec2, a0->vec2, a->vec2);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_decompose
*
* Description: For all coefficients c of the input polynomial,
*              compute high and low bits c0, c1 such c mod^+ Q = c1*ALPHA + c0
*              with -ALPHA/2 < c0 <= ALPHA/2 except if c1 = (Q-1)/ALPHA where we
*              set c1 = 0 and -ALPHA/2 <= c0 = c mod Q - Q < 0.
*              Assumes coefficients to be positive standard representatives.
*
* Arguments:   - poly *a1: pointer to output polynomial with coefficients c1
*              - poly *a0: pointer to output polynomial with coefficients c0
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_decompose(poly *a1, poly *a0, const poly *a) {
    DBENCH_START();

    PQCLEAN_DILITHIUM2_AVX2_decompose_avx(a1->vec, a0->vec, a->vec);

    DBENCH_STOP(*tround);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_make_hint
*
* Description: Compute hint array. The coefficients of which are the
*              indices of the coefficients of the input polynomial
*              whose low bits overflow into the high bits.
*
* Arguments:   - uint8_t *h: pointer to output hint array (preallocated of length N)
*              - const poly *a0: pointer to low part of input polynomial
*              - const poly *a1: pointer to high part of input polynomial
*
* Returns number of hints, i.e. length of hint array.
**************************************************/
unsigned int PQCLEAN_DILITHIUM2_AVX2_poly_make_hint(uint8_t hint[N], const poly *a0, const poly *a1) {
    unsigned int r;
    DBENCH_START();

    r = PQCLEAN_DILITHIUM2_AVX2_make_hint_avx(hint, a0->vec, a1->vec);

    DBENCH_STOP(*tround);
    return r;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_use_hint
*
* Description: Use hint polynomial to correct the high bits of a polynomial.
*
* Arguments:   - poly *b: pointer to output polynomial with corrected high bits
*              - const poly *a: pointer to input polynomial
*              - const poly *h: pointer to input hint polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_use_hint(poly *b, const poly *a, const poly *h) {
    DBENCH_START();

    PQCLEAN_DILITHIUM2_AVX2_use_hint_avx(b->vec, a->vec, h->vec);

    DBENCH_STOP(*tround);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_chknorm
*
* Description: Check infinity norm of polynomial against given bound.
*              Assumes input polynomial to be reduced by PQCLEAN_DILITHIUM2_AVX2_poly_reduce().
*
* Arguments:   - const poly *a: pointer to polynomial
*              - int32_t B: norm bound
*
* Returns 0 if norm is strictly smaller than B <= (Q-1)/8 and 1 otherwise.
**************************************************/
int PQCLEAN_DILITHIUM2_AVX2_poly_chknorm(const poly *a, int32_t B) {
    unsigned int i;
    int r;
    __m256i f, t;
    const __m256i bound = _mm256_set1_epi32(B - 1);

    if (B > (Q - 1) / 8) {
        return 1;
    }

    t = _mm256_setzero_si256();
    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        f = _mm256_abs_epi32(f);
        f = _mm256_cmpgt_epi32(f, bound);
        t = _mm256_or_si256(t, f);
    }

    r = 1 - _mm256_testz_si256(t, t);

    return r;
}

int XURQ_AVX512_poly_chknorm(const poly *a, int32_t B) {
    unsigned int i;
    __m512i f;
    const __m512i bound = _mm512_set1_epi32(B - 1);
    uint16_t g;

    for (i = 0; i < N / 16; i++) {
        f = _mm512_load_si512(&a->vec2[i]);
        f = _mm512_abs_epi32(f);
        g |= _mm512_cmpgt_epi32_mask(f, bound);
    }

    return g != 0;
}

/*************************************************
* Name:        rej_uniform
*
* Description: Sample uniformly random coefficients in [0, Q-1] by
*              performing rejection sampling on array of random bytes.
*
* Arguments:   - int32_t *a: pointer to output array (allocated)
*              - unsigned int len: number of coefficients to be sampled
*              - const uint8_t *buf: array of random bytes
*              - unsigned int buflen: length of array of random bytes
*
* Returns number of sampled coefficients. Can be smaller than len if not enough
* random bytes were given.
**************************************************/
static unsigned int rej_uniform(int32_t *a,
                                unsigned int len,
                                const uint8_t *buf,
                                unsigned int buflen) {
    unsigned int ctr, pos;
    uint32_t t;
    DBENCH_START();

    ctr = pos = 0;
    while (ctr < len && pos + 3 <= buflen) {
        t = buf[pos++];
        t |= (uint32_t) buf[pos++] << 8;
        t |= (uint32_t) buf[pos++] << 16;
        t &= 0x7FFFFF;

        if (t < Q) {
            a[ctr++] = t;
        }
    }

    DBENCH_STOP(*tsample);
    return ctr;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_uniform
*
* Description: Sample polynomial with uniformly random coefficients
*              in [0,Q-1] by performing rejection sampling on the
*              output stream of SHAKE256(seed|nonce) or AES256CTR(seed,nonce).
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length SEEDBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_uniform_preinit(poly *a, stream128_state *state) {
    unsigned int ctr;
    /* PQCLEAN_DILITHIUM2_AVX2_rej_uniform_avx reads up to 8 additional bytes */
    ALIGNED_UINT8(REJ_UNIFORM_BUFLEN + 8) buf;

    stream128_squeezeblocks(buf.coeffs, REJ_UNIFORM_NBLOCKS, state);
    ctr = PQCLEAN_DILITHIUM2_AVX2_rej_uniform_avx(a->coeffs, buf.coeffs);

    while (ctr < N) {
        /* length of buf is always divisible by 3; hence, no bytes left */
        stream128_squeezeblocks(buf.coeffs, 1, state);
        ctr += rej_uniform(a->coeffs + ctr, N - ctr, buf.coeffs, STREAM128_BLOCKBYTES);
    }
}

void PQCLEAN_DILITHIUM2_AVX2_poly_uniform(poly *a, const uint8_t seed[SEEDBYTES], uint16_t nonce) {
    stream128_state state;
    stream128_init(&state, seed, nonce);
    PQCLEAN_DILITHIUM2_AVX2_poly_uniform_preinit(a, &state);
    stream128_release(&state);
}

void PQCLEAN_DILITHIUM2_AVX2_poly_uniform_4x(poly *a0,
                                             poly *a1,
                                             poly *a2,
                                             poly *a3,
                                             const uint8_t seed[32],
                                             uint16_t nonce0,
                                             uint16_t nonce1,
                                             uint16_t nonce2,
                                             uint16_t nonce3) {
    unsigned int ctr0, ctr1, ctr2, ctr3;
    ALIGNED_UINT8(REJ_UNIFORM_BUFLEN + 8) buf[4];
    keccakx4_state state;
    __m256i f;

    f = _mm256_loadu_si256((__m256i *) seed);
    _mm256_store_si256(buf[0].vec, f);
    _mm256_store_si256(buf[1].vec, f);
    _mm256_store_si256(buf[2].vec, f);
    _mm256_store_si256(buf[3].vec, f);

    buf[0].coeffs[SEEDBYTES + 0] = nonce0;
    buf[0].coeffs[SEEDBYTES + 1] = nonce0 >> 8;
    buf[1].coeffs[SEEDBYTES + 0] = nonce1;
    buf[1].coeffs[SEEDBYTES + 1] = nonce1 >> 8;
    buf[2].coeffs[SEEDBYTES + 0] = nonce2;
    buf[2].coeffs[SEEDBYTES + 1] = nonce2 >> 8;
    buf[3].coeffs[SEEDBYTES + 0] = nonce3;
    buf[3].coeffs[SEEDBYTES + 1] = nonce3 >> 8;

    PQCLEAN_DILITHIUM2_AVX2_shake128x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs,
                                                   SEEDBYTES + 2);
    PQCLEAN_DILITHIUM2_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs,
                                                     REJ_UNIFORM_NBLOCKS, &state);

    ctr0 = PQCLEAN_DILITHIUM2_AVX2_rej_uniform_avx(a0->coeffs, buf[0].coeffs);
    ctr1 = PQCLEAN_DILITHIUM2_AVX2_rej_uniform_avx(a1->coeffs, buf[1].coeffs);
    ctr2 = PQCLEAN_DILITHIUM2_AVX2_rej_uniform_avx(a2->coeffs, buf[2].coeffs);
    ctr3 = PQCLEAN_DILITHIUM2_AVX2_rej_uniform_avx(a3->coeffs, buf[3].coeffs);

    while (ctr0 < N || ctr1 < N || ctr2 < N || ctr3 < N) {
        PQCLEAN_DILITHIUM2_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1,
                                                         &state);

        ctr0 += rej_uniform(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs, SHAKE128_RATE);
        ctr1 += rej_uniform(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs, SHAKE128_RATE);
        ctr2 += rej_uniform(a2->coeffs + ctr2, N - ctr2, buf[2].coeffs, SHAKE128_RATE);
        ctr3 += rej_uniform(a3->coeffs + ctr3, N - ctr3, buf[3].coeffs, SHAKE128_RATE);
    }
}


void XURQ_AVX512_poly_uniform_8x(poly *restrict a0, poly *restrict a1, poly *restrict a2, poly *restrict a3,
                                 poly *restrict a4, poly *restrict a5, poly *restrict a6, poly *restrict a7,
                                 const uint64_t *restrict seed,
                                 uint16_t nonce0, uint16_t nonce1, uint16_t nonce2, uint16_t nonce3,
                                 uint16_t nonce4, uint16_t nonce5, uint16_t nonce6, uint16_t nonce7) {

    uint32_t ctr[8] = {0};
    keccakx8_state state;
    ALIGN(64) uint8_t buf[8][864];

    state.s[0] = _mm512_set1_epi64(seed[0]);
    state.s[1] = _mm512_set1_epi64(seed[1]);
    state.s[2] = _mm512_set1_epi64(seed[2]);
    state.s[3] = _mm512_set1_epi64(seed[3]);
    state.s[4] = _mm512_set_epi64((0x1f << 16) ^ nonce7, (0x1f << 16) ^ nonce6, (0x1f << 16) ^ nonce5,
                                  (0x1f << 16) ^ nonce4,
                                  (0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2, (0x1f << 16) ^ nonce1,
                                  (0x1f << 16) ^ nonce0);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm512_setzero_si512();

    state.s[20] = _mm512_set1_epi64(0x1ULL << 63);

    XURQ_AVX512_shake128x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 5);

    // XURQ_AVX512_rej_uniform8x(a0->coeffs,
    //                           a1->coeffs,
    //                           a2->coeffs,
    //                           a3->coeffs,
    //                           a4->coeffs,
    //                           a5->coeffs,
    //                           a6->coeffs,
    //                           a7->coeffs,
    //                           ctr, buf);

    ctr[0] = XURQ_rej_uniform_avx512(a0->coeffs, buf[0]);
    ctr[1] = XURQ_rej_uniform_avx512(a1->coeffs, buf[1]);
    ctr[2] = XURQ_rej_uniform_avx512(a2->coeffs, buf[2]);
    ctr[3] = XURQ_rej_uniform_avx512(a3->coeffs, buf[3]);
    ctr[4] = XURQ_rej_uniform_avx512(a4->coeffs, buf[4]);
    ctr[5] = XURQ_rej_uniform_avx512(a5->coeffs, buf[5]);
    ctr[6] = XURQ_rej_uniform_avx512(a6->coeffs, buf[6]);
    ctr[7] = XURQ_rej_uniform_avx512(a7->coeffs, buf[7]);


    while (ctr[0] < N || ctr[1] < N || ctr[2] < N || ctr[3] < N || ctr[4] < N || ctr[5] < N || ctr[6] < N ||
           ctr[7] < N) {
        XURQ_AVX512_shake128x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 1);
        ctr[0] += rej_uniform(a0->coeffs + ctr[0], N - ctr[0], buf[0], SHAKE128_RATE);
        ctr[1] += rej_uniform(a1->coeffs + ctr[1], N - ctr[1], buf[1], SHAKE128_RATE);
        ctr[2] += rej_uniform(a2->coeffs + ctr[2], N - ctr[2], buf[2], SHAKE128_RATE);
        ctr[3] += rej_uniform(a3->coeffs + ctr[3], N - ctr[3], buf[3], SHAKE128_RATE);
        ctr[4] += rej_uniform(a0->coeffs + ctr[4], N - ctr[4], buf[4], SHAKE128_RATE);
        ctr[5] += rej_uniform(a1->coeffs + ctr[5], N - ctr[5], buf[5], SHAKE128_RATE);
        ctr[6] += rej_uniform(a2->coeffs + ctr[6], N - ctr[6], buf[6], SHAKE128_RATE);
        ctr[7] += rej_uniform(a3->coeffs + ctr[7], N - ctr[7], buf[7], SHAKE128_RATE);
    }

}

/*************************************************
* Name:        rej_eta
*
* Description: Sample uniformly random coefficients in [-ETA, ETA] by
*              performing rejection sampling on array of random bytes.
*
* Arguments:   - int32_t *a: pointer to output array (allocated)
*              - unsigned int len: number of coefficients to be sampled
*              - const uint8_t *buf: array of random bytes
*              - unsigned int buflen: length of array of random bytes
*
* Returns number of sampled coefficients. Can be smaller than len if not enough
* random bytes were given.
**************************************************/
static unsigned int rej_eta(int32_t *a,
                            unsigned int len,
                            const uint8_t *buf,
                            unsigned int buflen) {
    unsigned int ctr, pos;
    uint32_t t0, t1;
    DBENCH_START();

    ctr = pos = 0;
    while (ctr < len && pos < buflen) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

        if (t0 < 15) {
            t0 = t0 - (205 * t0 >> 10) * 5;
            a[ctr++] = 2 - t0;
        }
        if (t1 < 15 && ctr < len) {
            t1 = t1 - (205 * t1 >> 10) * 5;
            a[ctr++] = 2 - t1;
        }
    }

    DBENCH_STOP(*tsample);
    return ctr;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_eta
*
* Description: Sample polynomial with uniformly random coefficients
*              in [-ETA,ETA] by performing rejection sampling using the
*              output stream of SHAKE256(seed|nonce)
*              or AES256CTR(seed,nonce).
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length CRHBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_uniform_eta_preinit(poly *a, stream256_state *state) {
    unsigned int ctr;
    ALIGNED_UINT8(REJ_UNIFORM_ETA_BUFLEN) buf;

    stream256_squeezeblocks(buf.coeffs, REJ_UNIFORM_ETA_NBLOCKS, state);
    ctr = PQCLEAN_DILITHIUM2_AVX2_rej_eta_avx(a->coeffs, buf.coeffs);

    while (ctr < N) {
        stream256_squeezeblocks(buf.coeffs, 1, state);
        ctr += rej_eta(a->coeffs + ctr, N - ctr, buf.coeffs, STREAM256_BLOCKBYTES);
    }
}

void PQCLEAN_DILITHIUM2_AVX2_poly_uniform_eta(poly *a, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    stream256_state state;
    stream256_init(&state, seed, nonce);
    PQCLEAN_DILITHIUM2_AVX2_poly_uniform_eta_preinit(a, &state);
    stream256_release(&state);
}

void PQCLEAN_DILITHIUM2_AVX2_poly_uniform_eta_4x(poly *a0,
                                                 poly *a1,
                                                 poly *a2,
                                                 poly *a3,
                                                 const uint8_t seed[64],
                                                 uint16_t nonce0,
                                                 uint16_t nonce1,
                                                 uint16_t nonce2,
                                                 uint16_t nonce3) {
    unsigned int ctr0, ctr1, ctr2, ctr3;
    ALIGNED_UINT8(REJ_UNIFORM_ETA_BUFLEN) buf[4];

    __m256i f;
    keccakx4_state state;

    f = _mm256_loadu_si256((__m256i *) &seed[0]);
    _mm256_store_si256(&buf[0].vec[0], f);
    _mm256_store_si256(&buf[1].vec[0], f);
    _mm256_store_si256(&buf[2].vec[0], f);
    _mm256_store_si256(&buf[3].vec[0], f);

    buf[0].coeffs[32] = nonce0;
    buf[0].coeffs[33] = nonce0 >> 8;
    buf[1].coeffs[32] = nonce1;
    buf[1].coeffs[33] = nonce1 >> 8;
    buf[2].coeffs[32] = nonce2;
    buf[2].coeffs[33] = nonce2 >> 8;
    buf[3].coeffs[32] = nonce3;
    buf[3].coeffs[33] = nonce3 >> 8;

    PQCLEAN_DILITHIUM2_AVX2_shake256x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs,
                                                   34);
    PQCLEAN_DILITHIUM2_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs,
                                                     REJ_UNIFORM_ETA_NBLOCKS, &state);

    ctr0 = PQCLEAN_DILITHIUM2_AVX2_rej_eta_avx(a0->coeffs, buf[0].coeffs);
    ctr1 = PQCLEAN_DILITHIUM2_AVX2_rej_eta_avx(a1->coeffs, buf[1].coeffs);
    ctr2 = PQCLEAN_DILITHIUM2_AVX2_rej_eta_avx(a2->coeffs, buf[2].coeffs);
    ctr3 = PQCLEAN_DILITHIUM2_AVX2_rej_eta_avx(a3->coeffs, buf[3].coeffs);

    while (ctr0 < N || ctr1 < N || ctr2 < N || ctr3 < N) {
        PQCLEAN_DILITHIUM2_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1,
                                                         &state);

        ctr0 += rej_eta(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs, SHAKE256_RATE);
        ctr1 += rej_eta(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs, SHAKE256_RATE);
        ctr2 += rej_eta(a2->coeffs + ctr2, N - ctr2, buf[2].coeffs, SHAKE256_RATE);
        ctr3 += rej_eta(a3->coeffs + ctr3, N - ctr3, buf[3].coeffs, SHAKE256_RATE);
    }
}

void XURQ_AVX512_poly_uniform_eta_8x(
        poly *a0,
        poly *a1,
        poly *a2,
        poly *a3,
        poly *a4,
        poly *a5,
        poly *a6,
        poly *a7,
        const uint64_t seed[4],
        uint16_t nonce0,
        uint16_t nonce1,
        uint16_t nonce2,
        uint16_t nonce3,
        uint16_t nonce4,
        uint16_t nonce5,
        uint16_t nonce6,
        uint16_t nonce7) {

    uint32_t ctr[8] = {0};
    keccakx8_state state;
    ALIGN(64)
    uint8_t buf[8][136];

    state.s[0] = _mm512_set1_epi64(seed[0]);
    state.s[1] = _mm512_set1_epi64(seed[1]);
    state.s[2] = _mm512_set1_epi64(seed[2]);
    state.s[3] = _mm512_set1_epi64(seed[3]);
    state.s[4] = _mm512_set_epi64((0x1f << 16) ^ nonce7, (0x1f << 16) ^ nonce6, (0x1f << 16) ^ nonce5,
                                  (0x1f << 16) ^ nonce4,
                                  (0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2, (0x1f << 16) ^ nonce1,
                                  (0x1f << 16) ^ nonce0);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm512_setzero_si512();

    state.s[16] = _mm512_set1_epi64(0x1ULL << 63);

    XURQ_AVX512_shake256x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 1);

    XURQ_AVX512_rej_eta8x(a0->coeffs,
                          a1->coeffs,
                          a2->coeffs,
                          a3->coeffs,
                          a4->coeffs,
                          a5->coeffs,
                          a6->coeffs,
                          a7->coeffs,
                          ctr, buf);


    while (ctr[0] < N || ctr[1] < N || ctr[2] < N || ctr[3] < N || ctr[4] < N || ctr[5] < N || ctr[6] < N ||
           ctr[7] < N) {
        XURQ_AVX512_shake256x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 1);

        ctr[0] += rej_eta(a0->coeffs + ctr[0], N - ctr[0], buf[0], SHAKE256_RATE);
        ctr[1] += rej_eta(a1->coeffs + ctr[1], N - ctr[1], buf[1], SHAKE256_RATE);
        ctr[2] += rej_eta(a2->coeffs + ctr[2], N - ctr[2], buf[2], SHAKE256_RATE);
        ctr[3] += rej_eta(a3->coeffs + ctr[3], N - ctr[3], buf[3], SHAKE256_RATE);
        ctr[4] += rej_eta(a4->coeffs + ctr[4], N - ctr[4], buf[4], SHAKE256_RATE);
        ctr[5] += rej_eta(a5->coeffs + ctr[5], N - ctr[5], buf[5], SHAKE256_RATE);
        ctr[6] += rej_eta(a6->coeffs + ctr[6], N - ctr[6], buf[6], SHAKE256_RATE);
        ctr[7] += rej_eta(a7->coeffs + ctr[7], N - ctr[7], buf[7], SHAKE256_RATE);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_gamma1
*
* Description: Sample polynomial with uniformly random coefficients
*              in [-(GAMMA1 - 1), GAMMA1] by unpacking output stream
*              of SHAKE256(seed|nonce) or AES256CTR(seed,nonce).
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length CRHBYTES
*              - uint16_t nonce: 16-bit nonce
**************************************************/
#define POLY_UNIFORM_GAMMA1_NBLOCKS ((POLYZ_PACKEDBYTES+STREAM256_BLOCKBYTES-1)/STREAM256_BLOCKBYTES)

void PQCLEAN_DILITHIUM2_AVX2_poly_uniform_gamma1_preinit(poly *a, stream256_state *state) {
    /* PQCLEAN_DILITHIUM2_AVX2_polyz_unpack reads 14 additional bytes */
    ALIGNED_UINT8(POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES + 14) buf;
    stream256_squeezeblocks(buf.coeffs, POLY_UNIFORM_GAMMA1_NBLOCKS, state);
    PQCLEAN_DILITHIUM2_AVX2_polyz_unpack(a, buf.coeffs);
}

void PQCLEAN_DILITHIUM2_AVX2_poly_uniform_gamma1(poly *a, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    stream256_state state;
    stream256_init(&state, seed, nonce);
    PQCLEAN_DILITHIUM2_AVX2_poly_uniform_gamma1_preinit(a, &state);
    stream256_release(&state);
}


void polyz_unpack(poly *r, const uint8_t *a) {
    unsigned int i;

    for (i = 0; i < N / 4; ++i) {
        r->coeffs[4 * i + 0] = a[9 * i + 0];
        r->coeffs[4 * i + 0] |= (uint32_t) a[9 * i + 1] << 8;
        r->coeffs[4 * i + 0] |= (uint32_t) a[9 * i + 2] << 16;
        r->coeffs[4 * i + 0] &= 0x3FFFF;

        r->coeffs[4 * i + 1] = a[9 * i + 2] >> 2;
        r->coeffs[4 * i + 1] |= (uint32_t) a[9 * i + 3] << 6;
        r->coeffs[4 * i + 1] |= (uint32_t) a[9 * i + 4] << 14;
        r->coeffs[4 * i + 1] &= 0x3FFFF;

        r->coeffs[4 * i + 2] = a[9 * i + 4] >> 4;
        r->coeffs[4 * i + 2] |= (uint32_t) a[9 * i + 5] << 4;
        r->coeffs[4 * i + 2] |= (uint32_t) a[9 * i + 6] << 12;
        r->coeffs[4 * i + 2] &= 0x3FFFF;

        r->coeffs[4 * i + 3] = a[9 * i + 6] >> 6;
        r->coeffs[4 * i + 3] |= (uint32_t) a[9 * i + 7] << 2;
        r->coeffs[4 * i + 3] |= (uint32_t) a[9 * i + 8] << 10;
        r->coeffs[4 * i + 3] &= 0x3FFFF;

        r->coeffs[4 * i + 0] = GAMMA1 - r->coeffs[4 * i + 0];
        r->coeffs[4 * i + 1] = GAMMA1 - r->coeffs[4 * i + 1];
        r->coeffs[4 * i + 2] = GAMMA1 - r->coeffs[4 * i + 2];
        r->coeffs[4 * i + 3] = GAMMA1 - r->coeffs[4 * i + 3];
    }

}


void PQCLEAN_DILITHIUM2_AVX2_poly_uniform_gamma1_4x(poly *a0,
                                                    poly *a1,
                                                    poly *a2,
                                                    poly *a3,
                                                    const uint8_t seed[64],
                                                    uint16_t nonce0,
                                                    uint16_t nonce1,
                                                    uint16_t nonce2,
                                                    uint16_t nonce3) {
    ALIGNED_UINT8(POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES + 14) buf[4];
    keccakx4_state state;
    __m256i f;
    __m128i f1;

    f = _mm256_loadu_si256((__m256i *) &seed[0]);
    _mm256_store_si256(&buf[0].vec[0], f);
    _mm256_store_si256(&buf[1].vec[0], f);
    _mm256_store_si256(&buf[2].vec[0], f);
    _mm256_store_si256(&buf[3].vec[0], f);
    f1 = _mm_loadu_si128((__m128i *) &seed[32]);
    _mm_store_si128(&buf[0].vec[1], f1);
    _mm_store_si128(&buf[1].vec[1], f1);
    _mm_store_si128(&buf[2].vec[1], f1);
    _mm_store_si128(&buf[3].vec[1], f1);

    buf[0].coeffs[48] = nonce0;
    buf[0].coeffs[49] = nonce0 >> 8;
    buf[1].coeffs[48] = nonce1;
    buf[1].coeffs[49] = nonce1 >> 8;
    buf[2].coeffs[48] = nonce2;
    buf[2].coeffs[49] = nonce2 >> 8;
    buf[3].coeffs[48] = nonce3;
    buf[3].coeffs[49] = nonce3 >> 8;

    PQCLEAN_DILITHIUM2_AVX2_shake256x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs,
                                                   50);
    PQCLEAN_DILITHIUM2_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs,
                                                     POLY_UNIFORM_GAMMA1_NBLOCKS, &state);

    PQCLEAN_DILITHIUM2_AVX2_polyz_unpack(a0, buf[0].coeffs);
    PQCLEAN_DILITHIUM2_AVX2_polyz_unpack(a1, buf[1].coeffs);
    PQCLEAN_DILITHIUM2_AVX2_polyz_unpack(a2, buf[2].coeffs);
    PQCLEAN_DILITHIUM2_AVX2_polyz_unpack(a3, buf[3].coeffs);
}

void XURQ_AVX512_precompute_gamma1_8x(uint8_t buf[8][704], const uint64_t seed[6],
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
    XURQ_AVX512_shake256x8_squeezeblocks(&state, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 5);

}

void XURQ_AVX512_uniform_gamma1_8x(poly *a0, poly *a1, poly *a2, poly *a3,
                                   uint8_t buf[8][704], int pos) {
    if (pos == 0)
        XURQ_AVX512_polyz_unpack4x(a0, a1, a2, a3, buf[0], buf[1], buf[2], buf[3]);
    else
        XURQ_AVX512_polyz_unpack4x(a0, a1, a2, a3, buf[4], buf[5], buf[6], buf[7]);

}

// void XURQ_AVX512_uniform_gamma1_8x(poly *a0, poly *a1, poly *a2, poly *a3,
//                                    uint8_t buf[8][704], int pos) {
//     if (pos == 0)
//         XURQ_AVX512_polyz_unpack4x(a0, a1, a2, a3, &buf[0]);
//     else
//         XURQ_AVX512_polyz_unpack4x(a0, a1, a2, a3, &buf[4]);
//
// }

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_challenge
*
* Description: Implementation of H. Samples polynomial with TAU nonzero
*              coefficients in {-1,1} using the output stream of
*              SHAKE256(seed).
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const uint8_t mu[]: byte array containing seed of length SEEDBYTES
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_poly_challenge(poly *restrict c, const uint8_t seed[SEEDBYTES]) {
    unsigned int i, b, pos;
    uint64_t signs;
    ALIGNED_UINT8(SHAKE256_RATE) buf;
    shake256incctx state;

    shake256_inc_init(&state);
    shake256_inc_absorb(&state, seed, SEEDBYTES);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(buf.coeffs, SHAKE256_RATE, &state);

    memcpy(&signs, buf.coeffs, 8);
    pos = 8;

    memset(c->vec, 0, sizeof(poly));
    for (i = N - TAU; i < N; ++i) {
        do {
            if (pos >= SHAKE256_RATE) {
                shake256_inc_squeeze(buf.coeffs, SHAKE256_RATE, &state);
                pos = 0;
            }

            b = buf.coeffs[pos++];
        } while (b > i);

        c->coeffs[i] = c->coeffs[b];
        c->coeffs[b] = 1 - 2 * (signs & 1);
        signs >>= 1;
    }
    shake256_inc_ctx_release(&state);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_polyeta_pack
*
* Description: Bit-pack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYETA_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_polyeta_pack(uint8_t r[POLYETA_PACKEDBYTES], const poly *restrict a) {
    unsigned int i;
    uint8_t t[8];
    DBENCH_START();

    for (i = 0; i < N / 8; ++i) {
        t[0] = ETA - a->coeffs[8 * i + 0];
        t[1] = ETA - a->coeffs[8 * i + 1];
        t[2] = ETA - a->coeffs[8 * i + 2];
        t[3] = ETA - a->coeffs[8 * i + 3];
        t[4] = ETA - a->coeffs[8 * i + 4];
        t[5] = ETA - a->coeffs[8 * i + 5];
        t[6] = ETA - a->coeffs[8 * i + 6];
        t[7] = ETA - a->coeffs[8 * i + 7];

        r[3 * i + 0] = (t[0] >> 0) | (t[1] << 3) | (t[2] << 6);
        r[3 * i + 1] = (t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7);
        r[3 * i + 2] = (t[5] >> 1) | (t[6] << 2) | (t[7] << 5);
    }

    DBENCH_STOP(*tpack);
}


ALIGN(64) static const uint8_t idx_eta1[64] = {
        0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26, 32, 33, 34, 40,
        41, 42, 48, 49, 50, 56, 57, 58, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

void XURQ_AVX512_polyeta_pack(uint8_t r[POLYETA_PACKEDBYTES], const poly *restrict a) {
    __m512i b0, b1, b2, b3, d;
    __m128i c0, c1, c2, c3;
    uint64_t mmask = 0x0000000000ffffff;
    ALIGN(64) uint8_t t[256];

    const __m512i eta = _mm512_set1_epi32(ETA);
    const __m512i idx1 = _mm512_load_si512(idx_eta1);

    const __m512i mask0 = _mm512_set1_epi16(0xff);
    const __m512i mask1 = _mm512_set1_epi32(0xffff);
    const __m512i mask2 = _mm512_set1_epi64(0xffffffff);

    for (int i = 0; i < 4; i++) {
        b0 = _mm512_load_si512(a->vec2 + i * 4);
        b1 = _mm512_load_si512(a->vec2 + i * 4 + 1);
        b2 = _mm512_load_si512(a->vec2 + i * 4 + 2);
        b3 = _mm512_load_si512(a->vec2 + i * 4 + 3);

        b0 = _mm512_sub_epi32(eta, b0);
        b1 = _mm512_sub_epi32(eta, b1);
        b2 = _mm512_sub_epi32(eta, b2);
        b3 = _mm512_sub_epi32(eta, b3);

        c0 = _mm512_cvtepi32_epi8(b0);
        c1 = _mm512_cvtepi32_epi8(b1);
        c2 = _mm512_cvtepi32_epi8(b2);
        c3 = _mm512_cvtepi32_epi8(b3);
        //_mm512_cvtsepi32_epi8

        d = _mm512_inserti64x2(d, c0, 0);
        d = _mm512_inserti64x2(d, c1, 1);
        d = _mm512_inserti64x2(d, c2, 2);
        d = _mm512_inserti64x2(d, c3, 3);

        d ^= _mm512_srli_epi16(d, 5);
        d &= mask0;
        d ^= _mm512_srli_epi32(d, 10);
        d &= mask1;
        d ^= _mm512_srli_epi64(d, 20);
        d &= mask2;
        d = _mm512_maskz_permutexvar_epi8(mmask, idx1, d);
        _mm512_mask_storeu_epi32(r + 24 * i, 0x3f, d);

    }

}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_polyeta_unpack
*
* Description: Unpack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_polyeta_unpack(poly *restrict r, const uint8_t a[POLYETA_PACKEDBYTES]) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 8; ++i) {
        r->coeffs[8 * i + 0] = (a[3 * i + 0] >> 0) & 7;
        r->coeffs[8 * i + 1] = (a[3 * i + 0] >> 3) & 7;
        r->coeffs[8 * i + 2] = ((a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2)) & 7;
        r->coeffs[8 * i + 3] = (a[3 * i + 1] >> 1) & 7;
        r->coeffs[8 * i + 4] = (a[3 * i + 1] >> 4) & 7;
        r->coeffs[8 * i + 5] = ((a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1)) & 7;
        r->coeffs[8 * i + 6] = (a[3 * i + 2] >> 2) & 7;
        r->coeffs[8 * i + 7] = (a[3 * i + 2] >> 5) & 7;

        r->coeffs[8 * i + 0] = ETA - r->coeffs[8 * i + 0];
        r->coeffs[8 * i + 1] = ETA - r->coeffs[8 * i + 1];
        r->coeffs[8 * i + 2] = ETA - r->coeffs[8 * i + 2];
        r->coeffs[8 * i + 3] = ETA - r->coeffs[8 * i + 3];
        r->coeffs[8 * i + 4] = ETA - r->coeffs[8 * i + 4];
        r->coeffs[8 * i + 5] = ETA - r->coeffs[8 * i + 5];
        r->coeffs[8 * i + 6] = ETA - r->coeffs[8 * i + 6];
        r->coeffs[8 * i + 7] = ETA - r->coeffs[8 * i + 7];
    }

    DBENCH_STOP(*tpack);
}

ALIGN(64) static const uint8_t idx_eta_up[64] = {
        0, 0xff, 0, 0xff, 0, 1, 1, 0xff, 1, 0xff, 1, 2, 2, 0xff, 2, 0xff,
        3, 0xff, 3, 0xff, 3, 4, 4, 0xff, 4, 0xff, 4, 5, 5, 0xff, 5, 0xff,
        6, 0xff, 6, 0xff, 6, 7, 7, 0xff, 7, 0xff, 7, 8, 8, 0xff, 8, 0xff,
        9, 0xff, 9, 0xff, 9, 10, 10, 0xff, 10, 0xff, 10, 11, 11, 0xff, 11, 0xff,
};

ALIGN(64) static const uint16_t idx_eta_up1[32] = {
        0, 3, 6, 1, 4, 7, 2, 5,
        0, 3, 6, 1, 4, 7, 2, 5,
        0, 3, 6, 1, 4, 7, 2, 5,
        0, 3, 6, 1, 4, 7, 2, 5,
};

void XURQ_AVX512_polyeta_unpack(poly *restrict r, const uint8_t a[POLYETA_PACKEDBYTES]) {
    __m512i b0, b1, b2, b3;
    int pos = 0;
    int ctr = 0;
    const __m512i idx0 = _mm512_load_si512(idx_eta_up);
    const __m512i idx1 = _mm512_load_si512(idx_eta_up1);
    const __m512i mask = _mm512_set1_epi16(7);
    const __m512i eta = _mm512_set1_epi16(2);

    b0 = _mm512_loadu_si512(a + pos);
    b1 = _mm512_loadu_si512(a + pos + 12);
    b2 = _mm512_loadu_si512(a + pos + 24);
    b3 = _mm512_loadu_si512(a + pos + 36);

    b0 = _mm512_permutexvar_epi8(idx0, b0);
    b1 = _mm512_permutexvar_epi8(idx0, b1);
    b2 = _mm512_permutexvar_epi8(idx0, b2);
    b3 = _mm512_permutexvar_epi8(idx0, b3);

    b0 = _mm512_srlv_epi16(b0, idx1);
    b1 = _mm512_srlv_epi16(b1, idx1);
    b2 = _mm512_srlv_epi16(b2, idx1);
    b3 = _mm512_srlv_epi16(b3, idx1);

    b0 &= mask;
    b1 &= mask;
    b2 &= mask;
    b3 &= mask;

    b0 = _mm512_sub_epi16(eta, b0);
    b1 = _mm512_sub_epi16(eta, b1);
    b2 = _mm512_sub_epi16(eta, b2);
    b3 = _mm512_sub_epi16(eta, b3);

    _mm512_storeu_epi32(r->coeffs + ctr, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b0, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 16, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b0, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 32, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b1, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 48, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b1, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 64, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b2, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 80, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b2, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 96, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b3, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 112, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b3, 1)));

    // _mm512_mask_cvtepi16_storeu_epi32(r->coeffs + ctr, 0xffffffff, b0);
    // _mm512_mask_cvtepi16_storeu_epi32(r->coeffs + ctr + 32, 0xffffffff, b1);
    // _mm512_mask_cvtepi16_storeu_epi32(r->coeffs + ctr + 64, 0xffffffff, b2);
    // _mm512_mask_cvtepi16_storeu_epi32(r->coeffs + ctr + 96, 0xffffffff, b3);

    pos += 48;
    ctr += 128;

    b0 = _mm512_loadu_si512(a + pos);
    b1 = _mm512_loadu_si512(a + pos + 12);
    b2 = _mm512_loadu_si512(a + pos + 24);
    b3 = _mm512_loadu_si512(a + pos + 36);

    b0 = _mm512_permutexvar_epi8(idx0, b0);
    b1 = _mm512_permutexvar_epi8(idx0, b1);
    b2 = _mm512_permutexvar_epi8(idx0, b2);
    b3 = _mm512_permutexvar_epi8(idx0, b3);

    b0 = _mm512_srlv_epi16(b0, idx1);
    b1 = _mm512_srlv_epi16(b1, idx1);
    b2 = _mm512_srlv_epi16(b2, idx1);
    b3 = _mm512_srlv_epi16(b3, idx1);

    b0 &= mask;
    b1 &= mask;
    b2 &= mask;
    b3 &= mask;

    b0 = _mm512_sub_epi16(eta, b0);
    b1 = _mm512_sub_epi16(eta, b1);
    b2 = _mm512_sub_epi16(eta, b2);
    b3 = _mm512_sub_epi16(eta, b3);

    _mm512_storeu_epi32(r->coeffs + ctr, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b0, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 16, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b0, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 32, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b1, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 48, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b1, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 64, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b2, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 80, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b2, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 96, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b3, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 112, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b3, 1)));
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_polyt1_pack
*
* Description: Bit-pack polynomial t1 with coefficients fitting in 10 bits.
*              Input coefficients are assumed to be positive standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_polyt1_pack(uint8_t r[POLYT1_PACKEDBYTES], const poly *restrict a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 4; ++i) {
        r[5 * i + 0] = (a->coeffs[4 * i + 0] >> 0);
        r[5 * i + 1] = (a->coeffs[4 * i + 0] >> 8) | (a->coeffs[4 * i + 1] << 2);
        r[5 * i + 2] = (a->coeffs[4 * i + 1] >> 6) | (a->coeffs[4 * i + 2] << 4);
        r[5 * i + 3] = (a->coeffs[4 * i + 2] >> 4) | (a->coeffs[4 * i + 3] << 6);
        r[5 * i + 4] = (a->coeffs[4 * i + 3] >> 2);
    }

    DBENCH_STOP(*tpack);
}

ALIGN(64) static const uint16_t idx_t_p0[32] = {
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

ALIGN(64) static const uint16_t idx_t_p1[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
};

void XURQ_AVX512_polyt1_pack(uint8_t r[POLYT1_PACKEDBYTES], const poly *restrict a) {
    int pos = 0;
    int ctr = 0;
    __m512i f0, f1;
    __m512i p0;

    const __m512i idx0 = _mm512_load_si512(idx_t_p0);
    const __m512i idx1 = _mm512_load_si512(idx_t_p1);
    const __m512i mask0 = _mm512_set1_epi32(0xffff);
    const __m512i mask1 = _mm512_set1_epi64(0xffffffff);


    for (int i = 0; i < 8; ++i) {
        f0 = _mm512_load_si512(a->coeffs + pos);
        f1 = _mm512_load_si512(a->coeffs + pos + 16);

        f1 = _mm512_maskz_permutexvar_epi16(0xffff0000, idx1, f1);
        f0 = _mm512_mask_permutexvar_epi16(f1, 0xffff, idx0, f0);

        p0 = _mm512_srli_epi32(_mm512_andnot_epi32(mask0, f0), 6);
        f0 = _mm512_ternarylogic_epi32(p0, mask0, f0, 0x78);
        p0 = _mm512_srli_epi64(_mm512_andnot_epi64(mask1, f0), 12);
        f0 = _mm512_ternarylogic_epi32(p0, mask1, f0, 0x78);

        _mm512_mask_compressstoreu_epi8(r + ctr, 0x1f1f1f1f1f1f1f1f, f0);

        pos += 32;
        ctr += 40;
    }

}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_polyt1_unpack
*
* Description: Unpack polynomial t1 with 10-bit coefficients.
*              Output coefficients are positive standard representatives.
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_polyt1_unpack(poly *restrict r, const uint8_t a[POLYT1_PACKEDBYTES]) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 4; ++i) {
        r->coeffs[4 * i + 0] = ((a[5 * i + 0] >> 0) | ((uint32_t) a[5 * i + 1] << 8)) & 0x3FF;
        r->coeffs[4 * i + 1] = ((a[5 * i + 1] >> 2) | ((uint32_t) a[5 * i + 2] << 6)) & 0x3FF;
        r->coeffs[4 * i + 2] = ((a[5 * i + 2] >> 4) | ((uint32_t) a[5 * i + 3] << 4)) & 0x3FF;
        r->coeffs[4 * i + 3] = ((a[5 * i + 3] >> 6) | ((uint32_t) a[5 * i + 4] << 2)) & 0x3FF;
    }

    DBENCH_STOP(*tpack);
}


ALIGN(64) static const uint8_t idx_t1_u1[64] = {
        0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9,
        2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11,
        0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9,
        2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11,
};

ALIGN(64) static const uint16_t idx_t1_u2[32] = {
        0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6,
        0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6
};


void XURQ_AVX512_polyt1_unpack(poly *restrict r, const uint8_t a[POLYT1_PACKEDBYTES]) {

    int pos = 0;

    __m512i b;
    const __m512i mask = _mm512_set1_epi16(0x3FF);
    const __m512i index0 = _mm512_setr_epi32(0, 1, 2, 0,
                                             2, 3, 4, 0,
                                             5, 6, 7, 0,
                                             7, 8, 9, 0);
    const __m512i index1 = _mm512_load_si512(idx_t1_u1);
    const __m512i index2 = _mm512_load_si512(idx_t1_u2);

    for (int i = 0; i < 16; i += 2) {
        b = _mm512_loadu_si512(a + pos);
        b = _mm512_maskz_permutexvar_epi32(0x7777, index0, b);
        b = _mm512_shuffle_epi8(b, index1);
        b = _mm512_srlv_epi16(b, index2);
        b &= mask;
        _mm512_store_si512(&r->vec2[i], _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b, 0)));
        _mm512_store_si512(&r->vec2[i + 1], _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b, 1)));
        pos += 40;
    }


}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_polyt0_pack
*
* Description: Bit-pack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT0_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_polyt0_pack(uint8_t r[POLYT0_PACKEDBYTES], const poly *restrict a) {
    unsigned int i;
    uint32_t t[8];
    DBENCH_START();

    for (i = 0; i < N / 8; ++i) {
        t[0] = (1 << (D - 1)) - a->coeffs[8 * i + 0];
        t[1] = (1 << (D - 1)) - a->coeffs[8 * i + 1];
        t[2] = (1 << (D - 1)) - a->coeffs[8 * i + 2];
        t[3] = (1 << (D - 1)) - a->coeffs[8 * i + 3];
        t[4] = (1 << (D - 1)) - a->coeffs[8 * i + 4];
        t[5] = (1 << (D - 1)) - a->coeffs[8 * i + 5];
        t[6] = (1 << (D - 1)) - a->coeffs[8 * i + 6];
        t[7] = (1 << (D - 1)) - a->coeffs[8 * i + 7];

        r[13 * i + 0] = t[0];
        r[13 * i + 1] = t[0] >> 8;
        r[13 * i + 1] |= t[1] << 5;
        r[13 * i + 2] = t[1] >> 3;
        r[13 * i + 3] = t[1] >> 11;
        r[13 * i + 3] |= t[2] << 2;
        r[13 * i + 4] = t[2] >> 6;
        r[13 * i + 4] |= t[3] << 7;
        r[13 * i + 5] = t[3] >> 1;
        r[13 * i + 6] = t[3] >> 9;
        r[13 * i + 6] |= t[4] << 4;
        r[13 * i + 7] = t[4] >> 4;
        r[13 * i + 8] = t[4] >> 12;
        r[13 * i + 8] |= t[5] << 1;
        r[13 * i + 9] = t[5] >> 7;
        r[13 * i + 9] |= t[6] << 6;
        r[13 * i + 10] = t[6] >> 2;
        r[13 * i + 11] = t[6] >> 10;
        r[13 * i + 11] |= t[7] << 3;
        r[13 * i + 12] = t[7] >> 5;
    }

    DBENCH_STOP(*tpack);
}



ALIGN(64) static const uint16_t idx_t0_p3[32] = {
        0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0,
        0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0,
};


void XURQ_AVX512_polyt0_pack(uint8_t r[POLYT0_PACKEDBYTES], const poly *restrict a) {
    int pos = 0;
    int ctr = 0;
    __m512i f0, f1;
    __m512i p0;

    const __m512i n = _mm512_set1_epi32((1 << (D - 1)));
    const __m512i mask0 = _mm512_set1_epi64(0xffffffff);
    const __m512i mask1 = _mm512_set1_epi32(0xffff);
    const __m512i idx0 = _mm512_load_si512(idx_t_p0);
    const __m512i idx1 = _mm512_load_si512(idx_t_p1);
    const __m512i idx3 = _mm512_load_si512(idx_t0_p3);

    for (int i = 0; i < 8; ++i) {
        f0 = _mm512_loadu_si512(a->coeffs + pos);
        f1 = _mm512_loadu_si512(a->coeffs + pos + 16);

        f0 = _mm512_sub_epi32(n, f0);
        f1 = _mm512_sub_epi32(n, f1);

        f1 = _mm512_maskz_permutexvar_epi16(0xffff0000, idx1, f1);
        f0 = _mm512_mask_permutexvar_epi16(f1, 0xffff, idx0, f0);

        // f0 = (f0 & mask1) ^ _mm512_srli_epi32(_mm512_andnot_epi32(mask1,f0),3);
        // f0 = (f0 & mask0) ^ _mm512_srli_epi64(_mm512_andnot_epi64(mask0,f0),6);
        p0 = _mm512_srli_epi32(_mm512_andnot_epi32(mask1, f0), 3);
        f0 = _mm512_ternarylogic_epi32(p0, mask1, f0, 0x78);
        p0 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f0), 6);
        f0 = _mm512_ternarylogic_epi32(p0, mask0, f0, 0x78);

        p0 = _mm512_maskz_permutexvar_epi16(0x08080808, idx3, f0);
        p0 = _mm512_slli_epi16(p0, 4);
        f0 |= p0;
        f0 = _mm512_mask_srli_epi64(f0,0xaa,f0,12);

        _mm512_mask_compressstoreu_epi8(r + ctr, 0x1fff1fff1fff1fff, f0);

        ctr += 52;
        pos += 32;
    }

}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_polyt0_unpack
*
* Description: Unpack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_polyt0_unpack(poly *restrict r, const uint8_t a[POLYT0_PACKEDBYTES]) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 8; ++i) {
        r->coeffs[8 * i + 0] = a[13 * i + 0];
        r->coeffs[8 * i + 0] |= (uint32_t) a[13 * i + 1] << 8;
        r->coeffs[8 * i + 0] &= 0x1FFF;

        r->coeffs[8 * i + 1] = a[13 * i + 1] >> 5;
        r->coeffs[8 * i + 1] |= (uint32_t) a[13 * i + 2] << 3;
        r->coeffs[8 * i + 1] |= (uint32_t) a[13 * i + 3] << 11;
        r->coeffs[8 * i + 1] &= 0x1FFF;

        r->coeffs[8 * i + 2] = a[13 * i + 3] >> 2;
        r->coeffs[8 * i + 2] |= (uint32_t) a[13 * i + 4] << 6;
        r->coeffs[8 * i + 2] &= 0x1FFF;

        r->coeffs[8 * i + 3] = a[13 * i + 4] >> 7;
        r->coeffs[8 * i + 3] |= (uint32_t) a[13 * i + 5] << 1;
        r->coeffs[8 * i + 3] |= (uint32_t) a[13 * i + 6] << 9;
        r->coeffs[8 * i + 3] &= 0x1FFF;

        r->coeffs[8 * i + 4] = a[13 * i + 6] >> 4;
        r->coeffs[8 * i + 4] |= (uint32_t) a[13 * i + 7] << 4;
        r->coeffs[8 * i + 4] |= (uint32_t) a[13 * i + 8] << 12;
        r->coeffs[8 * i + 4] &= 0x1FFF;

        r->coeffs[8 * i + 5] = a[13 * i + 8] >> 1;
        r->coeffs[8 * i + 5] |= (uint32_t) a[13 * i + 9] << 7;
        r->coeffs[8 * i + 5] &= 0x1FFF;

        r->coeffs[8 * i + 6] = a[13 * i + 9] >> 6;
        r->coeffs[8 * i + 6] |= (uint32_t) a[13 * i + 10] << 2;
        r->coeffs[8 * i + 6] |= (uint32_t) a[13 * i + 11] << 10;
        r->coeffs[8 * i + 6] &= 0x1FFF;

        r->coeffs[8 * i + 7] = a[13 * i + 11] >> 3;
        r->coeffs[8 * i + 7] |= (uint32_t) a[13 * i + 12] << 5;
        r->coeffs[8 * i + 7] &= 0x1FFF;

        r->coeffs[8 * i + 0] = (1 << (D - 1)) - r->coeffs[8 * i + 0];
        r->coeffs[8 * i + 1] = (1 << (D - 1)) - r->coeffs[8 * i + 1];
        r->coeffs[8 * i + 2] = (1 << (D - 1)) - r->coeffs[8 * i + 2];
        r->coeffs[8 * i + 3] = (1 << (D - 1)) - r->coeffs[8 * i + 3];
        r->coeffs[8 * i + 4] = (1 << (D - 1)) - r->coeffs[8 * i + 4];
        r->coeffs[8 * i + 5] = (1 << (D - 1)) - r->coeffs[8 * i + 5];
        r->coeffs[8 * i + 6] = (1 << (D - 1)) - r->coeffs[8 * i + 6];
        r->coeffs[8 * i + 7] = (1 << (D - 1)) - r->coeffs[8 * i + 7];
    }

    DBENCH_STOP(*tpack);
}

ALIGN(64) static const uint8_t idx_t0_u[64] = {
        0, 1, 0xff, 0xff,
        1, 2, 3, 0xff,
        3, 4, 0xff, 0xff,
        4, 5, 6, 0xff,
        6, 7, 8, 0xff,
        8, 9, 0xff, 0xff,
        9, 10, 11, 0xff,
        11, 12, 0xff, 0xff,

        13, 14, 0xff, 0xff,
        14, 15, 16, 0xff,
        16, 17, 0xff, 0xff,
        17, 18, 19, 0xff,
        19, 20, 21, 0xff,
        21, 22, 0xff, 0xff,
        22, 23, 24, 0xff,
        24, 25, 0xff, 0xff
};


void XURQ_AVX512_polyt0_unpack(poly *restrict r, const uint8_t a[POLYT0_PACKEDBYTES]) {
    int pos = 0;
    int ctr = 0;
    __m512i f0, f1, f2, f3;
    const __m512i idx0 = _mm512_load_si512(idx_t0_u);
    const __m512i idx1 = _mm512_setr_epi32(0, 5, 2, 7,
                                           4, 1, 6, 3,
                                           0, 5, 2, 7,
                                           4, 1, 6, 3);
    const __m512i mask = _mm512_set1_epi32(0x1FFF);
    const __m512i n = _mm512_set1_epi32((1 << (D - 1)));


    for (int i = 0; i < 4; ++i) {
        f0 = _mm512_load_si512(a + pos);
        f1 = _mm512_load_si512(a + pos + 26);
        f2 = _mm512_load_si512(a + pos + 52);
        f3 = _mm512_load_si512(a + pos + 78);

        f0 = _mm512_permutexvar_epi8(idx0, f0);
        f1 = _mm512_permutexvar_epi8(idx0, f1);
        f2 = _mm512_permutexvar_epi8(idx0, f2);
        f3 = _mm512_permutexvar_epi8(idx0, f3);

        f0 = _mm512_srlv_epi32(f0, idx1);
        f1 = _mm512_srlv_epi32(f1, idx1);
        f2 = _mm512_srlv_epi32(f2, idx1);
        f3 = _mm512_srlv_epi32(f3, idx1);

        f0 &= mask;
        f1 &= mask;
        f2 &= mask;
        f3 &= mask;

        f0 = _mm512_sub_epi32(n, f0);
        f1 = _mm512_sub_epi32(n, f1);
        f2 = _mm512_sub_epi32(n, f2);
        f3 = _mm512_sub_epi32(n, f3);

        _mm512_storeu_epi32(r->coeffs + ctr, f0);
        _mm512_storeu_epi32(r->coeffs + ctr + 16, f1);
        _mm512_storeu_epi32(r->coeffs + ctr + 32, f2);
        _mm512_storeu_epi32(r->coeffs + ctr + 48, f3);

        pos += 104;
        ctr += 64;
    }

}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_polyz_pack
*
* Description: Bit-pack polynomial with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYZ_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_polyz_pack(uint8_t r[POLYZ_PACKEDBYTES], const poly *restrict a) {
    unsigned int i;
    uint32_t t[4];
    DBENCH_START();

    for (i = 0; i < N / 4; ++i) {
        t[0] = GAMMA1 - a->coeffs[4 * i + 0];
        t[1] = GAMMA1 - a->coeffs[4 * i + 1];
        t[2] = GAMMA1 - a->coeffs[4 * i + 2];
        t[3] = GAMMA1 - a->coeffs[4 * i + 3];

        r[9 * i + 0] = t[0];
        r[9 * i + 1] = t[0] >> 8;
        r[9 * i + 2] = t[0] >> 16;
        r[9 * i + 2] |= t[1] << 2;
        r[9 * i + 3] = t[1] >> 6;
        r[9 * i + 4] = t[1] >> 14;
        r[9 * i + 4] |= t[2] << 4;
        r[9 * i + 5] = t[2] >> 4;
        r[9 * i + 6] = t[2] >> 12;
        r[9 * i + 6] |= t[3] << 6;
        r[9 * i + 7] = t[3] >> 2;
        r[9 * i + 8] = t[3] >> 10;
    }

    DBENCH_STOP(*tpack);
}


ALIGN(64) static const uint32_t idx_z_p[16] = {
        0, 2, 0, 0,
        0, 6, 0, 0,
        0, 10, 0, 0,
        0, 14, 0, 0
};

void XURQ_AVX512_polyz_pack(uint8_t r[POLYZ_PACKEDBYTES], const poly *restrict a) {
    int pos = 0;
    int ctr = 0;
    __m512i f0, f1, f2, f3;
    __m512i p0, p1, p2, p3;

    const __m512i mask0 = _mm512_set1_epi64(0xffffffff);
    const __m512i gamma = _mm512_set1_epi32(GAMMA1);
    const __m512i idx0 = _mm512_load_si512(idx_z_p);

    for (int i = 0; i < 4; ++i) {
        f0 = _mm512_loadu_si512(a->coeffs + pos);
        f1 = _mm512_loadu_si512(a->coeffs + pos + 16);
        f2 = _mm512_loadu_si512(a->coeffs + pos + 32);
        f3 = _mm512_loadu_si512(a->coeffs + pos + 48);

        f0 = _mm512_sub_epi32(gamma, f0);
        f1 = _mm512_sub_epi32(gamma, f1);
        f2 = _mm512_sub_epi32(gamma, f2);
        f3 = _mm512_sub_epi32(gamma, f3);

        p0 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f0), 14);
        p1 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f1), 14);
        p2 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f2), 14);
        p3 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f3), 14);

        f0 = _mm512_ternarylogic_epi32(p0, f0, mask0, 0x78);
        f1 = _mm512_ternarylogic_epi32(p1, f1, mask0, 0x78);
        f2 = _mm512_ternarylogic_epi32(p2, f2, mask0, 0x78);
        f3 = _mm512_ternarylogic_epi32(p3, f3, mask0, 0x78);

        p0 = _mm512_maskz_permutexvar_epi32(0x2222, idx0, f0);
        p1 = _mm512_maskz_permutexvar_epi32(0x2222, idx0, f1);
        p2 = _mm512_maskz_permutexvar_epi32(0x2222, idx0, f2);
        p3 = _mm512_maskz_permutexvar_epi32(0x2222, idx0, f3);

        p0 = _mm512_slli_epi32(p0, 4);
        p1 = _mm512_slli_epi32(p1, 4);
        p2 = _mm512_slli_epi32(p2, 4);
        p3 = _mm512_slli_epi32(p3, 4);

        f0 |= p0;
        f1 |= p1;
        f2 |= p2;
        f3 |= p3;

        f0 = _mm512_mask_srli_epi64(f0, 0xaa, f0, 28);
        f1 = _mm512_mask_srli_epi64(f1, 0xaa, f1, 28);
        f2 = _mm512_mask_srli_epi64(f2, 0xaa, f2, 28);
        f3 = _mm512_mask_srli_epi64(f3, 0xaa, f3, 28);

        _mm512_mask_compressstoreu_epi8(r + ctr, 0x01ff01ff01ff01ff, f0);
        _mm512_mask_compressstoreu_epi8(r + ctr + 36, 0x01ff01ff01ff01ff, f1);
        _mm512_mask_compressstoreu_epi8(r + ctr + 72, 0x01ff01ff01ff01ff, f2);
        _mm512_mask_compressstoreu_epi8(r + ctr + 108, 0x01ff01ff01ff01ff, f3);

        pos += 64;
        ctr += 144;

    }

}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_polyz_unpack
*
* Description: Unpack polynomial z with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_polyz_unpack(poly *restrict r, const uint8_t *a) {
    unsigned int i;
    __m256i f;
    const __m256i shufbidx = _mm256_set_epi8(-1, 9, 8, 7, -1, 7, 6, 5, -1, 5, 4, 3, -1, 3, 2, 1,
                                             -1, 8, 7, 6, -1, 6, 5, 4, -1, 4, 3, 2, -1, 2, 1, 0);
    const __m256i srlvdidx = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
    const __m256i mask = _mm256_set1_epi32(0x3FFFF);
    const __m256i gamma1 = _mm256_set1_epi32(GAMMA1);
    DBENCH_START();

    for (i = 0; i < N / 8; i++) {
        f = _mm256_loadu_si256((__m256i *) &a[18 * i]);
        f = _mm256_permute4x64_epi64(f, 0x94);
        f = _mm256_shuffle_epi8(f, shufbidx);
        f = _mm256_srlv_epi32(f, srlvdidx);
        f = _mm256_and_si256(f, mask);
        f = _mm256_sub_epi32(gamma1, f);
        _mm256_store_si256(&r->vec[i], f);
    }

    DBENCH_STOP(*tpack);
}


/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_polyw1_pack
*
* Description: Bit-pack polynomial w1 with coefficients in [0,15] or [0,43].
*              Input coefficients are assumed to be positive standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYW1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM2_AVX2_polyw1_pack(uint8_t *r, const poly *restrict a) {
    unsigned int i;
    __m256i f0, f1, f2, f3;
    const __m256i shift1 = _mm256_set1_epi16((64 << 8) + 1);
    const __m256i shift2 = _mm256_set1_epi32((4096 << 16) + 1);
    const __m256i shufdidx1 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    const __m256i shufdidx2 = _mm256_set_epi32(-1, -1, 6, 5, 4, 2, 1, 0);
    const __m256i shufbidx = _mm256_set_epi8(-1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0,
                                             -1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);
    DBENCH_START();

    for (i = 0; i < N / 32; i++) {
        f0 = _mm256_load_si256(&a->vec[4 * i + 0]);
        f1 = _mm256_load_si256(&a->vec[4 * i + 1]);
        f2 = _mm256_load_si256(&a->vec[4 * i + 2]);
        f3 = _mm256_load_si256(&a->vec[4 * i + 3]);
        f0 = _mm256_packus_epi32(f0, f1);
        f1 = _mm256_packus_epi32(f2, f3);
        f0 = _mm256_packus_epi16(f0, f1);
        f0 = _mm256_maddubs_epi16(f0, shift1);
        f0 = _mm256_madd_epi16(f0, shift2);
        f0 = _mm256_permutevar8x32_epi32(f0, shufdidx1);
        f0 = _mm256_shuffle_epi8(f0, shufbidx);
        f0 = _mm256_permutevar8x32_epi32(f0, shufdidx2);
        _mm256_storeu_si256((__m256i *) &r[24 * i], f0);
    }

    DBENCH_STOP(*tpack);
}

ALIGN(64) static const uint8_t idx_w1_p0[64] = {
        0, 1, 2,
        4, 5, 6,
        8, 9, 10,
        12, 13, 14,
        16, 17, 18,
        20, 21, 22,
        24, 25, 26,
        28, 29, 30,
        32, 33, 34,
        36, 37, 38,
        40, 41, 42,
        44, 45, 46,
        48, 49, 50,
        52, 53, 54,
        56, 57, 58,
        60, 61, 62,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

ALIGN(64) static const uint8_t idx_w1_p[64] = {
        0, 1, 2,
        16, 17, 18,
        32, 33, 34,
        48, 49, 50,

        4, 5, 6,
        20, 21, 22,
        36, 37, 38,
        52, 53, 54,

        8, 9, 10,
        24, 25, 26,
        40, 41, 42,
        56, 57, 58,

        12, 13, 14,
        28, 29, 30,
        44, 45, 46,
        60, 61, 62,

        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

void XURQ_AVX512_polyw1_pack(uint8_t *r, const poly *restrict a) {
    __m512i f0, f1, f2, f3;
    const __m512i shift1 = _mm512_set1_epi16((64 << 8) + 1);
    const __m512i shift2 = _mm512_set1_epi32((4096 << 16) + 1);
    const __m512i idx0 = _mm512_load_si512(idx_w1_p);

    for (int i = 0; i < 4; i++) {
        f0 = _mm512_load_si512(&a->vec2[4 * i + 0]);
        f1 = _mm512_load_si512(&a->vec2[4 * i + 1]);
        f2 = _mm512_load_si512(&a->vec2[4 * i + 2]);
        f3 = _mm512_load_si512(&a->vec2[4 * i + 3]);
        f0 = _mm512_packus_epi32(f0, f1);
        f1 = _mm512_packus_epi32(f2, f3);
        f0 = _mm512_packus_epi16(f0, f1);
        f0 = _mm512_maddubs_epi16(f0, shift1);
        f0 = _mm512_madd_epi16(f0, shift2);
        f0 = _mm512_permutexvar_epi8(idx0, f0);
        _mm512_storeu_si512(&r[48 * i], f0);
    }

}


