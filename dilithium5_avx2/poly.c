#include "align.h"
#include "consts.h"
#include "fips202x4.h"
#include "ntt.h"
#include "params.h"
#include "poly.h"
#include "rejsample.h"
#include "rounding.h"
#include "symmetric.h"
#include "polyvec.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "ntt/ntt.h"

#define DBENCH_START()
#define DBENCH_STOP(t)

#define _mm256_blendv_epi32(a,b,mask) \
    _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(a), \
                                         _mm256_castsi256_ps(b), \
                                         _mm256_castsi256_ps(mask)))

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_reduce
*
* Description: Inplace reduction of all coefficients of polynomial to
*              representative in [-6283009,6283007]. Assumes input
*              coefficients to be at most 2^31 - 2^22 - 1 in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_reduce(poly *a) {
    unsigned int i;
    __m256i f, g;
    const __m256i q = _mm256_load_si256(&PQCLEAN_DILITHIUM5_AVX2_qdata.vec[_8XQ / 8]);
    const __m256i off = _mm256_set1_epi32(1 << 22);
    DBENCH_START();

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        g = _mm256_add_epi32(f, off);
        g = _mm256_srai_epi32(g, 23);
        g = _mm256_mullo_epi32(g, q);
        f = _mm256_sub_epi32(f, g);
        _mm256_store_si256(&a->vec[i], f);
    }

    DBENCH_STOP(*tred);
}

/*************************************************
* Name:        poly_addq
*
* Description: For all coefficients of in/out polynomial add Q if
*              coefficient is negative.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_caddq(poly *a) {
    unsigned int i;
    __m256i f, g;
    const __m256i q = _mm256_load_si256(&PQCLEAN_DILITHIUM5_AVX2_qdata.vec[_8XQ / 8]);
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

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_add
*
* Description: Add polynomials. No modular reduction is performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first summand
*              - const poly *b: pointer to second summand
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_add(poly *c, const poly *a, const poly *b)  {
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

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_sub
*
* Description: Subtract polynomials. No modular reduction is
*              performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial to be
*                               subtraced from first input polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_sub(poly *c, const poly *a, const poly *b) {
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

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_shiftl
*
* Description: Multiply polynomial by 2^D without modular reduction. Assumes
*              input coefficients to be less than 2^{31-D} in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_shiftl(poly *a) {
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

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_ntt
*
* Description: Inplace forward NTT. Coefficients can grow by up to
*              8*Q in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_ntt(poly *a) {
    DBENCH_START();

    PQCLEAN_DILITHIUM5_AVX2_ntt_avx(a->vec, PQCLEAN_DILITHIUM5_AVX2_qdata.vec);

    DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_invntt_tomont
*
* Description: Inplace inverse NTT and multiplication by 2^{32}.
*              Input coefficients need to be less than Q in absolute
*              value and output coefficients are again bounded by Q.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_invntt_tomont(poly *a) {
    DBENCH_START();

    PQCLEAN_DILITHIUM5_AVX2_invntt_avx(a->vec, PQCLEAN_DILITHIUM5_AVX2_qdata.vec);

    DBENCH_STOP(*tmul);
}

void PQCLEAN_DILITHIUM5_AVX2_poly_nttunpack(poly *a) {
    DBENCH_START();

    PQCLEAN_DILITHIUM5_AVX2_nttunpack_avx(a->vec);

    DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_pointwise_montgomery
*
* Description: Pointwise multiplication of polynomials in NTT domain
*              representation and multiplication of resulting polynomial
*              by 2^{-32}.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
    DBENCH_START();

    PQCLEAN_DILITHIUM5_AVX2_pointwise_avx(c->vec, a->vec, b->vec, PQCLEAN_DILITHIUM5_AVX2_qdata.vec);

    DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_power2round
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
void PQCLEAN_DILITHIUM5_AVX2_poly_power2round(poly *a1, poly *a0, const poly *a) {
    DBENCH_START();

    PQCLEAN_DILITHIUM5_AVX2_power2round_avx(a1->vec, a0->vec, a->vec);

    DBENCH_STOP(*tround);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_decompose
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
void PQCLEAN_DILITHIUM5_AVX2_poly_decompose(poly *a1, poly *a0, const poly *a) {
    DBENCH_START();

    PQCLEAN_DILITHIUM5_AVX2_decompose_avx(a1->vec, a0->vec, a->vec);

    DBENCH_STOP(*tround);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_make_hint
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
unsigned int PQCLEAN_DILITHIUM5_AVX2_poly_make_hint(uint8_t hint[N], const poly *a0, const poly *a1) {
    unsigned int r;
    DBENCH_START();

    r = PQCLEAN_DILITHIUM5_AVX2_make_hint_avx(hint, a0->vec, a1->vec);

    DBENCH_STOP(*tround);
    return r;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_use_hint
*
* Description: Use hint polynomial to correct the high bits of a polynomial.
*
* Arguments:   - poly *b: pointer to output polynomial with corrected high bits
*              - const poly *a: pointer to input polynomial
*              - const poly *h: pointer to input hint polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_use_hint(poly *b, const poly *a, const poly *h) {
    DBENCH_START();

    PQCLEAN_DILITHIUM5_AVX2_use_hint_avx(b->vec, a->vec, h->vec);

    DBENCH_STOP(*tround);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_chknorm
*
* Description: Check infinity norm of polynomial against given bound.
*              Assumes input polynomial to be reduced by PQCLEAN_DILITHIUM5_AVX2_poly_reduce().
*
* Arguments:   - const poly *a: pointer to polynomial
*              - int32_t B: norm bound
*
* Returns 0 if norm is strictly smaller than B <= (Q-1)/8 and 1 otherwise.
**************************************************/
int PQCLEAN_DILITHIUM5_AVX2_poly_chknorm(const poly *a, int32_t B) {
    unsigned int i;
    int r;
    __m256i f, t;
    const __m256i bound = _mm256_set1_epi32(B - 1);
    DBENCH_START();

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
    DBENCH_STOP(*tsample);
    return r;
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
        t  = buf[pos++];
        t |= (uint32_t)buf[pos++] << 8;
        t |= (uint32_t)buf[pos++] << 16;
        t &= 0x7FFFFF;

        if (t < Q) {
            a[ctr++] = t;
        }
    }

    DBENCH_STOP(*tsample);
    return ctr;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_uniform
*
* Description: Sample polynomial with uniformly random coefficients
*              in [0,Q-1] by performing rejection sampling on the
*              output stream of SHAKE256(seed|nonce) or AES256CTR(seed,nonce).
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length SEEDBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_uniform_preinit(poly *a, stream128_state *state) {
    unsigned int ctr;
    /* PQCLEAN_DILITHIUM5_AVX2_rej_uniform_avx reads up to 8 additional bytes */
    ALIGNED_UINT8(REJ_UNIFORM_BUFLEN + 8) buf;

    stream128_squeezeblocks(buf.coeffs, REJ_UNIFORM_NBLOCKS, state);
    ctr = PQCLEAN_DILITHIUM5_AVX2_rej_uniform_avx(a->coeffs, buf.coeffs);

    while (ctr < N) {
        /* length of buf is always divisible by 3; hence, no bytes left */
        stream128_squeezeblocks(buf.coeffs, 1, state);
        ctr += rej_uniform(a->coeffs + ctr, N - ctr, buf.coeffs, STREAM128_BLOCKBYTES);
    }
}

void PQCLEAN_DILITHIUM5_AVX2_poly_uniform(poly *a, const uint8_t seed[SEEDBYTES], uint16_t nonce) {
    stream128_state state;
    stream128_init(&state, seed, nonce);
    PQCLEAN_DILITHIUM5_AVX2_poly_uniform_preinit(a, &state);
    stream128_release(&state);
}

void PQCLEAN_DILITHIUM5_AVX2_poly_uniform_4x(poly *a0,
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

    f = _mm256_loadu_si256((__m256i *)seed);
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

    PQCLEAN_DILITHIUM5_AVX2_shake128x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, SEEDBYTES + 2);
    PQCLEAN_DILITHIUM5_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, REJ_UNIFORM_NBLOCKS, &state);

    ctr0 = PQCLEAN_DILITHIUM5_AVX2_rej_uniform_avx(a0->coeffs, buf[0].coeffs);
    ctr1 = PQCLEAN_DILITHIUM5_AVX2_rej_uniform_avx(a1->coeffs, buf[1].coeffs);
    ctr2 = PQCLEAN_DILITHIUM5_AVX2_rej_uniform_avx(a2->coeffs, buf[2].coeffs);
    ctr3 = PQCLEAN_DILITHIUM5_AVX2_rej_uniform_avx(a3->coeffs, buf[3].coeffs);

    while (ctr0 < N || ctr1 < N || ctr2 < N || ctr3 < N) {
        PQCLEAN_DILITHIUM5_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

        ctr0 += rej_uniform(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs, SHAKE128_RATE);
        ctr1 += rej_uniform(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs, SHAKE128_RATE);
        ctr2 += rej_uniform(a2->coeffs + ctr2, N - ctr2, buf[2].coeffs, SHAKE128_RATE);
        ctr3 += rej_uniform(a3->coeffs + ctr3, N - ctr3, buf[3].coeffs, SHAKE128_RATE);
    }
}



void XURQ_AVX2_poly_uniform_4x_with_s1s3(poly *a0,
                                         poly *a1,
                                         poly *a2,
                                         poly *a3,
                                         const uint8_t seed[32],
                                         uint16_t nonce0,
                                         uint16_t nonce1,
                                         uint16_t nonce2,
                                         uint16_t nonce3) {
    unsigned int ctr[4] = {0};
    ALIGNED_UINT8(176) buf[4];
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

    for (int i = 0; i < 4; ++i) {
        XURQ_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

        ctr[0] = XURQ_AVX2_rej_uniform_avx_s1s3(a0->coeffs, buf[0].coeffs, ctr[0]);
        ctr[1] = XURQ_AVX2_rej_uniform_avx_s1s3(a1->coeffs, buf[1].coeffs, ctr[1]);
        ctr[2] = XURQ_AVX2_rej_uniform_avx_s1s3(a2->coeffs, buf[2].coeffs, ctr[2]);
        ctr[3] = XURQ_AVX2_rej_uniform_avx_s1s3(a3->coeffs, buf[3].coeffs, ctr[3]);
    }

    XURQ_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

    ctr[0] = XURQ_AVX2_rej_uniform_avx_s1s3_final(a0->coeffs, buf[0].coeffs, ctr[0]);
    ctr[1] = XURQ_AVX2_rej_uniform_avx_s1s3_final(a1->coeffs, buf[1].coeffs, ctr[1]);
    ctr[2] = XURQ_AVX2_rej_uniform_avx_s1s3_final(a2->coeffs, buf[2].coeffs, ctr[2]);
    ctr[3] = XURQ_AVX2_rej_uniform_avx_s1s3_final(a3->coeffs, buf[3].coeffs, ctr[3]);

    while (ctr[0] < N || ctr[1] < N || ctr[2] < N || ctr[3] < N) {
        XURQ_AVX2_shake128x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

        ctr[0] += rej_uniform(a0->coeffs + ctr[0], N - ctr[0], buf[0].coeffs, SHAKE128_RATE);
        ctr[1] += rej_uniform(a1->coeffs + ctr[1], N - ctr[1], buf[1].coeffs, SHAKE128_RATE);
        ctr[2] += rej_uniform(a2->coeffs + ctr[2], N - ctr[2], buf[2].coeffs, SHAKE128_RATE);
        ctr[3] += rej_uniform(a3->coeffs + ctr[3], N - ctr[3], buf[3].coeffs, SHAKE128_RATE);
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
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_uniform_eta
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
void PQCLEAN_DILITHIUM5_AVX2_poly_uniform_eta_preinit(poly *a, stream256_state *state) {
    unsigned int ctr;
    ALIGNED_UINT8(REJ_UNIFORM_ETA_BUFLEN) buf;

    stream256_squeezeblocks(buf.coeffs, REJ_UNIFORM_ETA_NBLOCKS, state);
    ctr = PQCLEAN_DILITHIUM5_AVX2_rej_eta_avx(a->coeffs, buf.coeffs);

    while (ctr < N) {
        stream256_squeezeblocks(buf.coeffs, 1, state);
        ctr += rej_eta(a->coeffs + ctr, N - ctr, buf.coeffs, STREAM256_BLOCKBYTES);
    }
}

void PQCLEAN_DILITHIUM5_AVX2_poly_uniform_eta(poly *a, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    stream256_state state;
    stream256_init(&state, seed, nonce);
    PQCLEAN_DILITHIUM5_AVX2_poly_uniform_eta_preinit(a, &state);
    stream256_release(&state);
}

void PQCLEAN_DILITHIUM5_AVX2_poly_uniform_eta_4x(poly *a0,
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

    f = _mm256_loadu_si256((__m256i *)&seed[0]);
    _mm256_store_si256(&buf[0].vec[0], f);
    _mm256_store_si256(&buf[1].vec[0], f);
    _mm256_store_si256(&buf[2].vec[0], f);
    _mm256_store_si256(&buf[3].vec[0], f);
    f = _mm256_loadu_si256((__m256i *)&seed[32]);
    _mm256_store_si256(&buf[0].vec[1], f);
    _mm256_store_si256(&buf[1].vec[1], f);
    _mm256_store_si256(&buf[2].vec[1], f);
    _mm256_store_si256(&buf[3].vec[1], f);

    buf[0].coeffs[64] = nonce0;
    buf[0].coeffs[65] = nonce0 >> 8;
    buf[1].coeffs[64] = nonce1;
    buf[1].coeffs[65] = nonce1 >> 8;
    buf[2].coeffs[64] = nonce2;
    buf[2].coeffs[65] = nonce2 >> 8;
    buf[3].coeffs[64] = nonce3;
    buf[3].coeffs[65] = nonce3 >> 8;

    PQCLEAN_DILITHIUM5_AVX2_shake256x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 66);
    PQCLEAN_DILITHIUM5_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, REJ_UNIFORM_ETA_NBLOCKS, &state);

    ctr0 = PQCLEAN_DILITHIUM5_AVX2_rej_eta_avx(a0->coeffs, buf[0].coeffs);
    ctr1 = PQCLEAN_DILITHIUM5_AVX2_rej_eta_avx(a1->coeffs, buf[1].coeffs);
    ctr2 = PQCLEAN_DILITHIUM5_AVX2_rej_eta_avx(a2->coeffs, buf[2].coeffs);
    ctr3 = PQCLEAN_DILITHIUM5_AVX2_rej_eta_avx(a3->coeffs, buf[3].coeffs);

    while (ctr0 < N || ctr1 < N || ctr2 < N || ctr3 < N) {
        PQCLEAN_DILITHIUM5_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

        ctr0 += rej_eta(a0->coeffs + ctr0, N - ctr0, buf[0].coeffs, SHAKE256_RATE);
        ctr1 += rej_eta(a1->coeffs + ctr1, N - ctr1, buf[1].coeffs, SHAKE256_RATE);
        ctr2 += rej_eta(a2->coeffs + ctr2, N - ctr2, buf[2].coeffs, SHAKE256_RATE);
        ctr3 += rej_eta(a3->coeffs + ctr3, N - ctr3, buf[3].coeffs, SHAKE256_RATE);
    }
}


void XURQ_AVX2_poly_uniform_eta_4x(poly *a0,
                                   poly *a1,
                                   poly *a2,
                                   poly *a3,
                                   const uint8_t seed[64],
                                   uint16_t nonce0,
                                   uint16_t nonce1,
                                   uint16_t nonce2,
                                   uint16_t nonce3) {
    unsigned int ctr[4] = {0};
    ALIGNED_UINT8(REJ_UNIFORM_ETA_BUFLEN) buf[4];
    uint64_t *seed64 = (uint64_t *) seed;

    keccakx4_state state;

    state.s[0] = _mm256_set1_epi64x(seed64[0]);
    state.s[1] = _mm256_set1_epi64x(seed64[1]);
    state.s[2] = _mm256_set1_epi64x(seed64[2]);
    state.s[3] = _mm256_set1_epi64x(seed64[3]);
    state.s[4] = _mm256_set_epi64x((0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2,
                                   (0x1f << 16) ^ nonce1, (0x1f << 16) ^ nonce0);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm256_setzero_si256();

    state.s[16] = _mm256_set1_epi64x(0x1ULL << 63);

    XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

    ctr[0] = XURQ_AVX2_rej_eta_avx_with_s2_horizontal(a0->coeffs, buf[0].coeffs);
    ctr[1] = XURQ_AVX2_rej_eta_avx_with_s2_horizontal(a1->coeffs, buf[1].coeffs);
    ctr[2] = XURQ_AVX2_rej_eta_avx_with_s2_horizontal(a2->coeffs, buf[2].coeffs);
    ctr[3] = XURQ_AVX2_rej_eta_avx_with_s2_horizontal(a3->coeffs, buf[3].coeffs);

    while (ctr[0] < N || ctr[1] < N || ctr[2] < N || ctr[3] < N) {
        XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1,
                                           &state);

        ctr[0] += rej_eta(a0->coeffs + ctr[0], N - ctr[0], buf[0].coeffs, SHAKE256_RATE);
        ctr[1] += rej_eta(a1->coeffs + ctr[1], N - ctr[1], buf[1].coeffs, SHAKE256_RATE);
        ctr[2] += rej_eta(a2->coeffs + ctr[2], N - ctr[2], buf[2].coeffs, SHAKE256_RATE);
        ctr[3] += rej_eta(a3->coeffs + ctr[3], N - ctr[3], buf[3].coeffs, SHAKE256_RATE);
    }
}


static void pack_eta(uint8_t *r, const uint8_t *pipe) {
    __m256i b0, b1, b2, b3, b4, b5, b6, b7;
    int ptr = 0;

    const __m256i mask0 = _mm256_set1_epi16(0xff);
    const __m256i mask1 = _mm256_set1_epi32(0xffff);
    const __m256i mask2 = _mm256_set1_epi64x(0xffffffff);
    const __m256i mask3 = _mm256_set_epi64x(0, 0xffffffffffffffffULL,0,0xffffffffffffffffULL);

    b0  = _mm256_load_si256((__m256i *) &pipe[0]);
    b1  = _mm256_load_si256((__m256i *) &pipe[32]);
    b2  = _mm256_load_si256((__m256i *) &pipe[64]);
    b3  = _mm256_load_si256((__m256i *) &pipe[96]);

    b0 ^= _mm256_srli_epi16(b0, 5);
    b1 ^= _mm256_srli_epi16(b1, 5);
    b2 ^= _mm256_srli_epi16(b2, 5);
    b3 ^= _mm256_srli_epi16(b3, 5);

    b0 &= mask0;
    b1 &= mask0;
    b2 &= mask0;
    b3 &= mask0;

    b0 ^= _mm256_srli_epi32(b0, 10);
    b1 ^= _mm256_srli_epi32(b1, 10);
    b2 ^= _mm256_srli_epi32(b2, 10);
    b3 ^= _mm256_srli_epi32(b3, 10);

    b0 &= mask1;
    b1 &= mask1;
    b2 &= mask1;
    b3 &= mask1;

    b0 ^= _mm256_srli_epi64(b0, 20);
    b1 ^= _mm256_srli_epi64(b1, 20);
    b2 ^= _mm256_srli_epi64(b2, 20);
    b3 ^= _mm256_srli_epi64(b3, 20);

    b0 &= mask2;
    b1 &= mask2;
    b2 &= mask2;
    b3 &= mask2;

    b0 ^= _mm256_bsrli_epi128(b0, 5);
    b1 ^= _mm256_bsrli_epi128(b1, 5);
    b2 ^= _mm256_bsrli_epi128(b2, 5);
    b3 ^= _mm256_bsrli_epi128(b3, 5);

    b0 &= mask3;
    b1 &= mask3;
    b2 &= mask3;
    b3 &= mask3;

    b0 ^= _mm256_bsrli_epi128(_mm256_permute4x64_epi64(b0,0x59), 2);
    b1 ^= _mm256_bsrli_epi128(_mm256_permute4x64_epi64(b1,0x59), 2);
    b2 ^= _mm256_bsrli_epi128(_mm256_permute4x64_epi64(b2,0x59), 2);
    b3 ^= _mm256_bsrli_epi128(_mm256_permute4x64_epi64(b3,0x59), 2);


    _mm256_storeu_si256((__m256i *)&r[ptr],b0);
    ptr += 12;
    _mm256_storeu_si256((__m256i *)&r[ptr],b1);
    ptr += 12;
    _mm256_storeu_si256((__m256i *)&r[ptr],b2);
    ptr += 12;
    _mm256_storeu_si256((__m256i *)&r[ptr],b3);
    ptr += 12;


    b0  = _mm256_load_si256((__m256i *) &pipe[128]);
    b1  = _mm256_load_si256((__m256i *) &pipe[160]);
    b2  = _mm256_load_si256((__m256i *) &pipe[192]);
    b3  = _mm256_load_si256((__m256i *) &pipe[224]);

    b0 ^= _mm256_srli_epi16(b0, 5);
    b1 ^= _mm256_srli_epi16(b1, 5);
    b2 ^= _mm256_srli_epi16(b2, 5);
    b3 ^= _mm256_srli_epi16(b3, 5);

    b0 &= mask0;
    b1 &= mask0;
    b2 &= mask0;
    b3 &= mask0;

    b0 ^= _mm256_srli_epi32(b0, 10);
    b1 ^= _mm256_srli_epi32(b1, 10);
    b2 ^= _mm256_srli_epi32(b2, 10);
    b3 ^= _mm256_srli_epi32(b3, 10);

    b0 &= mask1;
    b1 &= mask1;
    b2 &= mask1;
    b3 &= mask1;

    b0 ^= _mm256_srli_epi64(b0, 20);
    b1 ^= _mm256_srli_epi64(b1, 20);
    b2 ^= _mm256_srli_epi64(b2, 20);
    b3 ^= _mm256_srli_epi64(b3, 20);

    b0 &= mask2;
    b1 &= mask2;
    b2 &= mask2;
    b3 &= mask2;

    b0 ^= _mm256_bsrli_epi128(b0, 5);
    b1 ^= _mm256_bsrli_epi128(b1, 5);
    b2 ^= _mm256_bsrli_epi128(b2, 5);
    b3 ^= _mm256_bsrli_epi128(b3, 5);

    b0 &= mask3;
    b1 &= mask3;
    b2 &= mask3;
    b3 &= mask3;

    b0 ^= _mm256_bsrli_epi128(_mm256_permute4x64_epi64(b0,0x59), 2);
    b1 ^= _mm256_bsrli_epi128(_mm256_permute4x64_epi64(b1,0x59), 2);
    b2 ^= _mm256_bsrli_epi128(_mm256_permute4x64_epi64(b2,0x59), 2);
    b3 ^= _mm256_bsrli_epi128(_mm256_permute4x64_epi64(b3,0x59), 2);

    _mm256_storeu_si256((__m256i *)&r[ptr],b0);
    ptr += 12;
    _mm256_storeu_si256((__m256i *)&r[ptr],b1);
    ptr += 12;
    _mm256_storeu_si256((__m256i *)&r[ptr],b2);
    ptr += 12;
    _mm256_storeu_si256((__m256i *)&r[ptr],b3);
    //最后这里会溢出 但sk保留了足够的空间

}


static uint32_t rej_eta_with_pipe(int32_t *a,
                                  uint32_t ctr,
                                  uint8_t *pipe,
                                  const uint8_t *buf) {
    int32_t t0, t1;
    int pos = 0;
    while (ctr < N && pos < REJ_UNIFORM_ETA_BUFLEN) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

        if (t0 < 15) {
            t0 = t0 - ((13 * t0) >> 6) * 5;
            pipe[ctr] = t0;
            a[ctr++] = ETA - t0;
        }
        if (t1 < 15 && ctr < N) {
            t1 = t1 - ((13 * t1) >> 6) * 5;
            pipe[ctr] = t1;
            a[ctr++] = ETA - t1;
        }
    }

    return ctr;
}

void ExpandS_with_pack(polyvecl *s1,
                       polyveck *s2,
                       uint8_t *r,
                       const uint64_t seed[4]){
    unsigned int ctr[4] = {0};
    ALIGNED_UINT8(REJ_UNIFORM_ETA_BUFLEN) buf[4];
    ALIGNED_UINT8(276) pipe[4]; //20 bytes redundancy

    keccakx4_state state;

    // sample and pack s1

    state.s[0] = _mm256_set1_epi64x(seed[0]);
    state.s[1] = _mm256_set1_epi64x(seed[1]);
    state.s[2] = _mm256_set1_epi64x(seed[2]);
    state.s[3] = _mm256_set1_epi64x(seed[3]);
    state.s[4] = _mm256_set_epi64x((0x1f << 16) ^ 3, (0x1f << 16) ^ 2,
                                   (0x1f << 16) ^ 1, (0x1f << 16) ^ 0);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm256_setzero_si256();

    state.s[16] = _mm256_set1_epi64x(0x1ULL << 63);

    XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

    ctr[0] = XURQ_AVX2_rej_eta_avx_with_pack(s1->vec[0].coeffs, pipe[0].coeffs,buf[0].coeffs);
    ctr[1] = XURQ_AVX2_rej_eta_avx_with_pack(s1->vec[1].coeffs, pipe[1].coeffs,buf[1].coeffs);
    ctr[2] = XURQ_AVX2_rej_eta_avx_with_pack(s1->vec[2].coeffs, pipe[2].coeffs,buf[2].coeffs);
    ctr[3] = XURQ_AVX2_rej_eta_avx_with_pack(s1->vec[3].coeffs, pipe[3].coeffs,buf[3].coeffs);

    while (ctr[0] < N || ctr[1] < N || ctr[2] < N || ctr[3] < N) {
        XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

        ctr[0] = rej_eta_with_pipe(s1->vec[0].coeffs,ctr[0], pipe[0].coeffs,buf[0].coeffs);
        ctr[1] = rej_eta_with_pipe(s1->vec[1].coeffs,ctr[1], pipe[1].coeffs,buf[1].coeffs);
        ctr[2] = rej_eta_with_pipe(s1->vec[2].coeffs,ctr[2], pipe[2].coeffs,buf[2].coeffs);
        ctr[3] = rej_eta_with_pipe(s1->vec[3].coeffs,ctr[3], pipe[3].coeffs,buf[3].coeffs);
    }

    for ( int i = 0; i < 4; i++) {
        pack_eta(r + i * POLYETA_PACKEDBYTES, pipe[i].coeffs);
    }

    state.s[0] = _mm256_set1_epi64x(seed[0]);
    state.s[1] = _mm256_set1_epi64x(seed[1]);
    state.s[2] = _mm256_set1_epi64x(seed[2]);
    state.s[3] = _mm256_set1_epi64x(seed[3]);
    state.s[4] = _mm256_set_epi64x((0x1f << 16) ^ 7, (0x1f << 16) ^ 6,
                                   (0x1f << 16) ^ 5, (0x1f << 16) ^ 4);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm256_setzero_si256();

    state.s[16] = _mm256_set1_epi64x(0x1ULL << 63);

    XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

    ctr[0] = XURQ_AVX2_rej_eta_avx_with_pack(s1->vec[4].coeffs, pipe[0].coeffs,buf[0].coeffs);
    ctr[1] = XURQ_AVX2_rej_eta_avx_with_pack(s1->vec[5].coeffs, pipe[1].coeffs,buf[1].coeffs);
    ctr[2] = XURQ_AVX2_rej_eta_avx_with_pack(s1->vec[6].coeffs, pipe[2].coeffs,buf[2].coeffs);
//    ctr[3] = XURQ_AVX2_rej_eta_avx_with_pack(s1->vec[3].coeffs, pipe[3].coeffs,buf[3].coeffs);

    while (ctr[0] < N || ctr[1] < N || ctr[2] < N || ctr[3] < N) {
        XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

        ctr[0] = rej_eta_with_pipe(s1->vec[4].coeffs,ctr[0], pipe[0].coeffs,buf[0].coeffs);
        ctr[1] = rej_eta_with_pipe(s1->vec[5].coeffs,ctr[1], pipe[1].coeffs,buf[1].coeffs);
        ctr[2] = rej_eta_with_pipe(s1->vec[6].coeffs,ctr[2], pipe[2].coeffs,buf[2].coeffs);
//        ctr[3] = rej_eta_with_pipe(s1->vec[3].coeffs,ctr[3], pipe[3].coeffs,buf[3].coeffs);
    }

    for ( int i = 4; i < L; i++) {
        pack_eta(r + i * POLYETA_PACKEDBYTES, pipe[i-4].coeffs);
    }

    // sample and pack s2

    state.s[0] = _mm256_set1_epi64x(seed[0]);
    state.s[1] = _mm256_set1_epi64x(seed[1]);
    state.s[2] = _mm256_set1_epi64x(seed[2]);
    state.s[3] = _mm256_set1_epi64x(seed[3]);
    state.s[4] = _mm256_set_epi64x((0x1f << 16) ^ 10, (0x1f << 16) ^ 9,
                                   (0x1f << 16) ^ 8, (0x1f << 16) ^ 7);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm256_setzero_si256();

    state.s[16] = _mm256_set1_epi64x(0x1ULL << 63);

    XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

    ctr[0] = XURQ_AVX2_rej_eta_avx_with_pack(s2->vec[0].coeffs, pipe[0].coeffs,buf[0].coeffs);
    ctr[1] = XURQ_AVX2_rej_eta_avx_with_pack(s2->vec[1].coeffs, pipe[1].coeffs,buf[1].coeffs);
    ctr[2] = XURQ_AVX2_rej_eta_avx_with_pack(s2->vec[2].coeffs, pipe[2].coeffs,buf[2].coeffs);
    ctr[3] = XURQ_AVX2_rej_eta_avx_with_pack(s2->vec[3].coeffs, pipe[3].coeffs,buf[3].coeffs);

    while (ctr[0] < N || ctr[1] < N || ctr[2] < N || ctr[3] < N) {
        XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

        ctr[0] += rej_eta_with_pipe(s2->vec[0].coeffs, ctr[0],pipe[0].coeffs,buf[0].coeffs);
        ctr[1] += rej_eta_with_pipe(s2->vec[1].coeffs, ctr[1],pipe[1].coeffs,buf[1].coeffs);
        ctr[2] += rej_eta_with_pipe(s2->vec[2].coeffs, ctr[2],pipe[2].coeffs,buf[2].coeffs);
        ctr[3] += rej_eta_with_pipe(s2->vec[3].coeffs, ctr[3],pipe[3].coeffs,buf[3].coeffs);
    }


    for (int i = 0; i < 4; i++) {
        pack_eta(r + (L + i) * POLYETA_PACKEDBYTES, pipe[i].coeffs);
    }


    state.s[0] = _mm256_set1_epi64x(seed[0]);
    state.s[1] = _mm256_set1_epi64x(seed[1]);
    state.s[2] = _mm256_set1_epi64x(seed[2]);
    state.s[3] = _mm256_set1_epi64x(seed[3]);
    state.s[4] = _mm256_set_epi64x((0x1f << 16) ^ 14, (0x1f << 16) ^ 13,
                                   (0x1f << 16) ^ 12, (0x1f << 16) ^ 11);

    for (int j = 5; j < 25; ++j)
        state.s[j] = _mm256_setzero_si256();

    state.s[16] = _mm256_set1_epi64x(0x1ULL << 63);

    XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

    ctr[0] = XURQ_AVX2_rej_eta_avx_with_pack(s2->vec[4].coeffs, pipe[0].coeffs,buf[0].coeffs);
    ctr[1] = XURQ_AVX2_rej_eta_avx_with_pack(s2->vec[5].coeffs, pipe[1].coeffs,buf[1].coeffs);
    ctr[2] = XURQ_AVX2_rej_eta_avx_with_pack(s2->vec[6].coeffs, pipe[2].coeffs,buf[2].coeffs);
    ctr[3] = XURQ_AVX2_rej_eta_avx_with_pack(s2->vec[7].coeffs, pipe[3].coeffs,buf[3].coeffs);

    while (ctr[0] < N || ctr[1] < N || ctr[2] < N || ctr[3] < N) {
        XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 1, &state);

        ctr[0] += rej_eta_with_pipe(s2->vec[4].coeffs, ctr[0],pipe[0].coeffs,buf[0].coeffs);
        ctr[1] += rej_eta_with_pipe(s2->vec[5].coeffs, ctr[1],pipe[1].coeffs,buf[1].coeffs);
        ctr[2] += rej_eta_with_pipe(s2->vec[6].coeffs, ctr[2],pipe[2].coeffs,buf[2].coeffs);
        ctr[3] += rej_eta_with_pipe(s2->vec[7].coeffs, ctr[3],pipe[3].coeffs,buf[3].coeffs);
    }


    for (int i = 4; i < K; i++) {
        pack_eta(r + (L + i) * POLYETA_PACKEDBYTES, pipe[i-4].coeffs);
    }

}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_poly_uniform_gamma1
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
void PQCLEAN_DILITHIUM5_AVX2_poly_uniform_gamma1_preinit(poly *a, stream256_state *state) {
    /* PQCLEAN_DILITHIUM5_AVX2_polyz_unpack reads 14 additional bytes */
    ALIGNED_UINT8(POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES + 14) buf;
    stream256_squeezeblocks(buf.coeffs, POLY_UNIFORM_GAMMA1_NBLOCKS, state);
    PQCLEAN_DILITHIUM5_AVX2_polyz_unpack(a, buf.coeffs);
}

void PQCLEAN_DILITHIUM5_AVX2_poly_uniform_gamma1(poly *a, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    stream256_state state;
    stream256_init(&state, seed, nonce);
    PQCLEAN_DILITHIUM5_AVX2_poly_uniform_gamma1_preinit(a, &state);
    stream256_release(&state);
}

void PQCLEAN_DILITHIUM5_AVX2_poly_uniform_gamma1_4x(poly *a0,
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

    f = _mm256_loadu_si256((__m256i *)&seed[0]);
    _mm256_store_si256(&buf[0].vec[0], f);
    _mm256_store_si256(&buf[1].vec[0], f);
    _mm256_store_si256(&buf[2].vec[0], f);
    _mm256_store_si256(&buf[3].vec[0], f);
    f = _mm256_loadu_si256((__m256i *)&seed[32]);
    _mm256_store_si256(&buf[0].vec[1], f);
    _mm256_store_si256(&buf[1].vec[1], f);
    _mm256_store_si256(&buf[2].vec[1], f);
    _mm256_store_si256(&buf[3].vec[1], f);

    buf[0].coeffs[64] = nonce0;
    buf[0].coeffs[65] = nonce0 >> 8;
    buf[1].coeffs[64] = nonce1;
    buf[1].coeffs[65] = nonce1 >> 8;
    buf[2].coeffs[64] = nonce2;
    buf[2].coeffs[65] = nonce2 >> 8;
    buf[3].coeffs[64] = nonce3;
    buf[3].coeffs[65] = nonce3 >> 8;

    PQCLEAN_DILITHIUM5_AVX2_shake256x4_absorb_once(&state, buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, 66);
    PQCLEAN_DILITHIUM5_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs, POLY_UNIFORM_GAMMA1_NBLOCKS, &state);

    PQCLEAN_DILITHIUM5_AVX2_polyz_unpack(a0, buf[0].coeffs);
    PQCLEAN_DILITHIUM5_AVX2_polyz_unpack(a1, buf[1].coeffs);
    PQCLEAN_DILITHIUM5_AVX2_polyz_unpack(a2, buf[2].coeffs);
    PQCLEAN_DILITHIUM5_AVX2_polyz_unpack(a3, buf[3].coeffs);
}

void XURQ_AVX2_polyz_unpack(poly *restrict r, const uint8_t *a) {
    unsigned int i;
    __m256i f0, f1, f2, f3;
    const __m256i shufbidx = _mm256_set_epi8(-1, 11, 10, 9, -1, 9, 8, 7, -1, 6, 5, 4, -1, 4, 3, 2,
                                             -1, 9, 8, 7, -1, 7, 6, 5, -1, 4, 3, 2, -1, 2, 1, 0);
    const __m256i srlvdidx = _mm256_set1_epi64x((uint64_t)4 << 32);
    const __m256i mask = _mm256_set1_epi32(0xFFFFF);
    const __m256i gamma1 = _mm256_set1_epi32(GAMMA1);

    for (i = 0; i < 8; i++) {
        f0 = _mm256_loadu_si256((__m256i *) &a[80 * i]);
        f1 = _mm256_loadu_si256((__m256i *) &a[80 * i + 20]);
        f2 = _mm256_loadu_si256((__m256i *) &a[80 * i + 40]);
        f3 = _mm256_loadu_si256((__m256i *) &a[80 * i + 60]);

        f0 = _mm256_permute4x64_epi64(f0, 0x94);
        f1 = _mm256_permute4x64_epi64(f1, 0x94);
        f2 = _mm256_permute4x64_epi64(f2, 0x94);
        f3 = _mm256_permute4x64_epi64(f3, 0x94);

        f0 = _mm256_shuffle_epi8(f0, shufbidx);
        f1 = _mm256_shuffle_epi8(f1, shufbidx);
        f2 = _mm256_shuffle_epi8(f2, shufbidx);
        f3 = _mm256_shuffle_epi8(f3, shufbidx);

        f0 = _mm256_srlv_epi32(f0, srlvdidx);
        f1 = _mm256_srlv_epi32(f1, srlvdidx);
        f2 = _mm256_srlv_epi32(f2, srlvdidx);
        f3 = _mm256_srlv_epi32(f3, srlvdidx);

        f0 = _mm256_and_si256(f0, mask);
        f1 = _mm256_and_si256(f1, mask);
        f2 = _mm256_and_si256(f2, mask);
        f3 = _mm256_and_si256(f3, mask);

        f0 = _mm256_sub_epi32(gamma1, f0);
        f1 = _mm256_sub_epi32(gamma1, f1);
        f2 = _mm256_sub_epi32(gamma1, f2);
        f3 = _mm256_sub_epi32(gamma1, f3);

        _mm256_store_si256(&r->vec[4 * i], f0);
        _mm256_store_si256(&r->vec[4 * i + 1], f1);
        _mm256_store_si256(&r->vec[4 * i + 2], f2);
        _mm256_store_si256(&r->vec[4 * i + 3], f3);
    }

}

void XURQ_AVX2_poly_uniform_gamma1_4x(poly *a0,
                                      poly *a1,
                                      poly *a2,
                                      poly *a3,
                                      const uint8_t seed[64],
                                      uint16_t nonce0,
                                      uint16_t nonce1,
                                      uint16_t nonce2,
                                      uint16_t nonce3) {
    ALIGNED_UINT8(712) buf[4];
    keccakx4_state state;

    state.s[0] = _mm256_set1_epi64x(seed[0]);
    state.s[1] = _mm256_set1_epi64x(seed[1]);
    state.s[2] = _mm256_set1_epi64x(seed[2]);
    state.s[3] = _mm256_set1_epi64x(seed[3]);
    state.s[4] = _mm256_set1_epi64x(seed[4]);
    state.s[5] = _mm256_set1_epi64x(seed[5]);
    state.s[6] = _mm256_set_epi64x((0x1f << 16) ^ nonce3, (0x1f << 16) ^ nonce2, (0x1f << 16) ^ nonce1,
                                   (0x1f << 16) ^ nonce0);

    for (int j = 7; j < 25; ++j)
        state.s[j] = _mm256_setzero_si256();

    state.s[16] = _mm256_set1_epi64x(0x1ULL << 63);

    XURQ_AVX2_shake256x4_squeezeblocks(buf[0].coeffs, buf[1].coeffs, buf[2].coeffs, buf[3].coeffs,
                                       5, &state);

    XURQ_AVX2_polyz_unpack(a0, buf[0].coeffs);
    XURQ_AVX2_polyz_unpack(a1, buf[1].coeffs);
    XURQ_AVX2_polyz_unpack(a2, buf[2].coeffs);
    XURQ_AVX2_polyz_unpack(a3, buf[3].coeffs);
}


/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_challenge
*
* Description: Implementation of H. Samples polynomial with TAU nonzero
*              coefficients in {-1,1} using the output stream of
*              SHAKE256(seed).
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const uint8_t mu[]: byte array containing seed of length SEEDBYTES
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_poly_challenge(poly *restrict c, const uint8_t seed[SEEDBYTES]) {
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
* Name:        PQCLEAN_DILITHIUM5_AVX2_polyeta_pack
*
* Description: Bit-pack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYETA_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_polyeta_pack(uint8_t r[POLYETA_PACKEDBYTES], const poly *restrict a) {
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

        r[3 * i + 0]  = (t[0] >> 0) | (t[1] << 3) | (t[2] << 6);
        r[3 * i + 1]  = (t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7);
        r[3 * i + 2]  = (t[5] >> 1) | (t[6] << 2) | (t[7] << 5);
    }

    DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_polyeta_unpack
*
* Description: Unpack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_polyeta_unpack(poly *restrict r, const uint8_t a[POLYETA_PACKEDBYTES]) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 8; ++i) {
        r->coeffs[8 * i + 0] =  (a[3 * i + 0] >> 0) & 7;
        r->coeffs[8 * i + 1] =  (a[3 * i + 0] >> 3) & 7;
        r->coeffs[8 * i + 2] = ((a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2)) & 7;
        r->coeffs[8 * i + 3] =  (a[3 * i + 1] >> 1) & 7;
        r->coeffs[8 * i + 4] =  (a[3 * i + 1] >> 4) & 7;
        r->coeffs[8 * i + 5] = ((a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1)) & 7;
        r->coeffs[8 * i + 6] =  (a[3 * i + 2] >> 2) & 7;
        r->coeffs[8 * i + 7] =  (a[3 * i + 2] >> 5) & 7;

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

void polyeta_unpack_avx2(poly *restrict r, const uint8_t a[POLYETA_PACKEDBYTES]) {
    __m256i f0,f1,f2,f3,f4,f5,f6;

    const __m256i mask0 = _mm256_set1_epi32(0x7);
    const __m256i idx0 = _mm256_setr_epi32(0,3,6,9,12,15,18,21);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i eta = _mm256_set1_epi32(ETA);

    for (int i = 0; i < 6; ++i) {
        f0 = _mm256_loadu_si256((__m256i *) (a + 15 * i));

        f1 = _mm256_permutevar8x32_epi32(f0,zero);
        f1 = _mm256_srlv_epi32(f1,idx0);
        f1 = f1 & mask0;
        f1 = _mm256_sub_epi32(eta,f1);
        _mm256_store_si256(&r->vec[i * 5],f1);

        f2 = _mm256_srli_si256(f0,3);
        f2 = _mm256_permutevar8x32_epi32(f2,zero);
        f2 = _mm256_srlv_epi32(f2,idx0);
        f2 = f2 & mask0;
        f2 = _mm256_sub_epi32(eta,f2);
        _mm256_store_si256(&r->vec[i * 5 + 1],f2);

        f3 = _mm256_srli_si256(f0,6);
        f3 = _mm256_permutevar8x32_epi32(f3,zero);
        f3 = _mm256_srlv_epi32(f3,idx0);
        f3 = f3 & mask0;
        f3 = _mm256_sub_epi32(eta,f3);
        _mm256_store_si256(&r->vec[i * 5 + 2],f3);

        f4 = _mm256_srli_si256(f0,9);
        f4 = _mm256_permutevar8x32_epi32(f4,zero);
        f4 = _mm256_srlv_epi32(f4,idx0);
        f4 = f4 & mask0;
        f4 = _mm256_sub_epi32(eta,f4);
        _mm256_store_si256(&r->vec[i * 5 + 3],f4);

        f5 = _mm256_srli_si256(f0,12);
        f5 = _mm256_permutevar8x32_epi32(f5,zero);
        f5 = _mm256_srlv_epi32(f5,idx0);
        f5 = f5 & mask0;
        f5 = _mm256_sub_epi32(eta,f5);
        _mm256_store_si256(&r->vec[i * 5 + 4],f5);
    }

    f0 = _mm256_loadu_si256((__m256i *) (a + 90));

    f1 = _mm256_permutevar8x32_epi32(f0,zero);
    f1 = _mm256_srlv_epi32(f1,idx0);
    f1 = f1 & mask0;
    f1 = _mm256_sub_epi32(eta,f1);
    _mm256_store_si256(&r->vec[30],f1);

    f2 = _mm256_srli_si256(f0,3);
    f2 = _mm256_permutevar8x32_epi32(f2,zero);
    f2 = _mm256_srlv_epi32(f2,idx0);
    f2 = f2 & mask0;
    f2 = _mm256_sub_epi32(eta,f2);
    _mm256_store_si256(&r->vec[31],f2);

}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_polyt1_pack
*
* Description: Bit-pack polynomial t1 with coefficients fitting in 10 bits.
*              Input coefficients are assumed to be positive standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_polyt1_pack(uint8_t r[POLYT1_PACKEDBYTES], const poly *restrict a) {
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

void polyt1_pack_avx2(uint8_t r[320], const poly *restrict a) {
    __m256i f0,f1,f2,f3;
    __m256i t0,t1,t2,t3;
    __m128i g0,g1;
    const __m256i mask0 = _mm256_set1_epi64x(0xffffffff00000000UL);
    const __m256i idx0 = _mm256_setr_epi64x(0,20,0,20);

    for (int i = 0; i < 32; ++i) {
        //concatenate32(r,10)
        f0 = _mm256_load_si256(&a->vec[i]);
        t0 = _mm256_srli_epi64(f0, 22);
        f0 = t0 ^ f0;
        f0 = _mm256_andnot_si256(mask0,f0);

        //concatenate64(r, 20)
        t0 = _mm256_srli_si256(f0, 4);
        t0 = _mm256_srli_epi64(t0, 12);
        f0 = f0 ^ t0;
        f0 = _mm256_srlv_epi64(f0,idx0);

        g0 = _mm256_castsi256_si128(f0);
        g1 = _mm256_extractf128_si256(f0,1);
        _mm_storeu_si128((__m128i_u *) (r + i * 10), g0);
        _mm_storeu_si128((__m128i_u *) (r + i * 10 + 5), g1);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_polyt1_unpack
*
* Description: Unpack polynomial t1 with 10-bit coefficients.
*              Output coefficients are positive standard representatives.
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_polyt1_unpack(poly *restrict r, const uint8_t a[POLYT1_PACKEDBYTES]) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 4; ++i) {
        r->coeffs[4 * i + 0] = ((a[5 * i + 0] >> 0) | ((uint32_t)a[5 * i + 1] << 8)) & 0x3FF;
        r->coeffs[4 * i + 1] = ((a[5 * i + 1] >> 2) | ((uint32_t)a[5 * i + 2] << 6)) & 0x3FF;
        r->coeffs[4 * i + 2] = ((a[5 * i + 2] >> 4) | ((uint32_t)a[5 * i + 3] << 4)) & 0x3FF;
        r->coeffs[4 * i + 3] = ((a[5 * i + 3] >> 6) | ((uint32_t)a[5 * i + 4] << 2)) & 0x3FF;
    }

    DBENCH_STOP(*tpack);
}

void polyt1_unpack_avx2(poly *restrict r, const uint8_t a[320]) {
    __m256i b, b0;
    __m256i mask0 = _mm256_set_epi32(4, 4, 3, 2, 3, 2, 1, 0);
    __m256i mask1 = _mm256_set_epi8(11, 10, 10, 9, 9, 8, 8, 7, 6, 5, 5, 4, 4, 3, 3, 2,9, 8, 8, 7, 7, 6, 6, 5, 4, 3, 3, 2, 2, 1, 1, 0);
    __m256i mask2 = _mm256_set1_epi32(0x3ff);
    __m256i index = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);

    for (int i = 0; i < 16; ++i) {
        b = _mm256_loadu_si256((__m256i *) (a + 20 * i));
        b = _mm256_permutevar8x32_epi32(b, mask0);
        b = _mm256_shuffle_epi8(b, mask1);
        b0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b));
        b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b, 1));
        b0 = _mm256_srlv_epi32(b0, index);
        b = _mm256_srlv_epi32(b, index);
        b0 &= mask2;
        b &= mask2;
        _mm256_store_si256(&r->vec[i * 2], b0);
        _mm256_store_si256(&r->vec[i * 2 + 1], b);
    }

}
/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_polyt0_pack
*
* Description: Bit-pack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT0_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_polyt0_pack(uint8_t r[POLYT0_PACKEDBYTES], const poly *restrict a) {
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

        r[13 * i + 0]  =  t[0];
        r[13 * i + 1]  =  t[0] >>  8;
        r[13 * i + 1] |=  t[1] <<  5;
        r[13 * i + 2]  =  t[1] >>  3;
        r[13 * i + 3]  =  t[1] >> 11;
        r[13 * i + 3] |=  t[2] <<  2;
        r[13 * i + 4]  =  t[2] >>  6;
        r[13 * i + 4] |=  t[3] <<  7;
        r[13 * i + 5]  =  t[3] >>  1;
        r[13 * i + 6]  =  t[3] >>  9;
        r[13 * i + 6] |=  t[4] <<  4;
        r[13 * i + 7]  =  t[4] >>  4;
        r[13 * i + 8]  =  t[4] >> 12;
        r[13 * i + 8] |=  t[5] <<  1;
        r[13 * i + 9]  =  t[5] >>  7;
        r[13 * i + 9] |=  t[6] <<  6;
        r[13 * i + 10]  =  t[6] >>  2;
        r[13 * i + 11]  =  t[6] >> 10;
        r[13 * i + 11] |=  t[7] <<  3;
        r[13 * i + 12]  =  t[7] >>  5;
    }

    DBENCH_STOP(*tpack);
}

void polyt0_pack_avx2(uint8_t r[416], const poly *restrict a) {
    __m256i f0,f1,f2,f3;
    __m256i t0,t1,t2,t3;
    __m128i g0,g1;
    const __m256i d = _mm256_set1_epi32(1 << (D - 1));
    const __m256i mask0 = _mm256_set1_epi64x(0xffffffff00000000UL);
    const __m256i idx0 = _mm256_setr_epi64x(0,12,0,12);

    for (int i = 0; i < 32; ++i) {
        //concatenate32(r,13)
        f0 = _mm256_load_si256(&a->vec[i]);
        f0 = _mm256_sub_epi32(d, f0);
        t0 = _mm256_srli_epi64(f0, 19);
        f0 = t0 ^ f0;
        f0 = _mm256_andnot_si256(mask0, f0);

        //concatenate64(r, 26)
        t0 = _mm256_srli_si256(f0, 4);
        f0 = f0 ^ t0;
        t0 = f0 & mask0;
        t0 = _mm256_srli_epi64(t0, 6);
        f0 = _mm256_andnot_si256(mask0, f0);
        f0 = f0 ^ t0;
        f0 = _mm256_permute4x64_epi64(f0,0xd8);

        //concatenate64(r,52)
        t0 = _mm256_srli_si256(f0,4);
        t0 = t0 & mask0;
        t0 = _mm256_slli_epi32(t0, 20);
        f0 = f0 ^ t0;
        f0 = _mm256_srlv_epi64(f0,idx0);

        g0 = _mm256_castsi256_si128(f0);
        _mm_storeu_si128((__m128i_u *) (r + i * 13), g0);
    }

}
/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_polyt0_unpack
*
* Description: Unpack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_polyt0_unpack(poly *restrict r, const uint8_t a[POLYT0_PACKEDBYTES]) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 8; ++i) {
        r->coeffs[8 * i + 0]  = a[13 * i + 0];
        r->coeffs[8 * i + 0] |= (uint32_t)a[13 * i + 1] << 8;
        r->coeffs[8 * i + 0] &= 0x1FFF;

        r->coeffs[8 * i + 1]  = a[13 * i + 1] >> 5;
        r->coeffs[8 * i + 1] |= (uint32_t)a[13 * i + 2] << 3;
        r->coeffs[8 * i + 1] |= (uint32_t)a[13 * i + 3] << 11;
        r->coeffs[8 * i + 1] &= 0x1FFF;

        r->coeffs[8 * i + 2]  = a[13 * i + 3] >> 2;
        r->coeffs[8 * i + 2] |= (uint32_t)a[13 * i + 4] << 6;
        r->coeffs[8 * i + 2] &= 0x1FFF;

        r->coeffs[8 * i + 3]  = a[13 * i + 4] >> 7;
        r->coeffs[8 * i + 3] |= (uint32_t)a[13 * i + 5] << 1;
        r->coeffs[8 * i + 3] |= (uint32_t)a[13 * i + 6] << 9;
        r->coeffs[8 * i + 3] &= 0x1FFF;

        r->coeffs[8 * i + 4]  = a[13 * i + 6] >> 4;
        r->coeffs[8 * i + 4] |= (uint32_t)a[13 * i + 7] << 4;
        r->coeffs[8 * i + 4] |= (uint32_t)a[13 * i + 8] << 12;
        r->coeffs[8 * i + 4] &= 0x1FFF;

        r->coeffs[8 * i + 5]  = a[13 * i + 8] >> 1;
        r->coeffs[8 * i + 5] |= (uint32_t)a[13 * i + 9] << 7;
        r->coeffs[8 * i + 5] &= 0x1FFF;

        r->coeffs[8 * i + 6]  = a[13 * i + 9] >> 6;
        r->coeffs[8 * i + 6] |= (uint32_t)a[13 * i + 10] << 2;
        r->coeffs[8 * i + 6] |= (uint32_t)a[13 * i + 11] << 10;
        r->coeffs[8 * i + 6] &= 0x1FFF;

        r->coeffs[8 * i + 7]  = a[13 * i + 11] >> 3;
        r->coeffs[8 * i + 7] |= (uint32_t)a[13 * i + 12] << 5;
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

void polyt0_unpack_avx2(poly *restrict r, const uint8_t a[POLYT0_PACKEDBYTES]) {
    __m256i f0,f1,f2,f3,f4,f5,f6;

    const __m256i mask0 = _mm256_set1_epi32(0x1fff);
    const __m256i idx0 = _mm256_setr_epi8(0,1,2,3,0,1,2,3,2,3,4,5,4,5,6,7,
                                          6,7,8,9,8,9,8,9,8,9,10,11,10,11,12,13);
    const __m256i idx1 = _mm256_setr_epi32(0,13,10,7,4,1,14,11);
    const __m256i d = _mm256_set1_epi32(1 << (D - 1));

    for (int i = 0; i < N / 8; ++i) {
        f0 = _mm256_loadu_si256((__m256i *) (a + 13 * i));

        f1 = _mm256_permute4x64_epi64(f0, 0x44);
        f1 = _mm256_shuffle_epi8(f1,idx0);
        f1 = _mm256_srlv_epi32(f1,idx1);
        f1 = f1 & mask0;
        f1 = _mm256_sub_epi32(d, f1);

        _mm256_store_si256(&r->vec[i],f1);
    }

}
/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_polyz_pack
*
* Description: Bit-pack polynomial with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYZ_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_polyz_pack(uint8_t r[POLYZ_PACKEDBYTES], const poly *restrict a) {
    unsigned int i;
    uint32_t t[4];
    DBENCH_START();

    for (i = 0; i < N / 2; ++i) {
        t[0] = GAMMA1 - a->coeffs[2 * i + 0];
        t[1] = GAMMA1 - a->coeffs[2 * i + 1];

        r[5 * i + 0]  = t[0];
        r[5 * i + 1]  = t[0] >> 8;
        r[5 * i + 2]  = t[0] >> 16;
        r[5 * i + 2] |= t[1] << 4;
        r[5 * i + 3]  = t[1] >> 4;
        r[5 * i + 4]  = t[1] >> 12;
    }

    DBENCH_STOP(*tpack);
}

void polyz_pack_avx2(uint8_t r[POLYZ_PACKEDBYTES], const poly *restrict a) {
    __m256i f0, f1, f2, f3;
    __m256i p0, p1, p2, p3;

    const __m256i mask0 = _mm256_set1_epi64x(0xffffffff);
    const __m256i gamma = _mm256_set1_epi32(GAMMA1);
    const __m256i index = _mm256_setr_epi8(0,1,2,3,4,8,9,10,
                                           11,12,-1,-1,-1,-1,-1,-1,
                                           0,1,2,3,4,8,9,10,
                                           11,12,-1,-1,-1,-1,-1,-1);


    for (int i = 0; i < N / 8; ++i) {
        f0 = _mm256_load_si256(&a->vec[i]);

        f0 = _mm256_sub_epi32(gamma, f0);

        p0 = _mm256_andnot_si256(mask0, f0);
        f0 = (f0 & mask0) | _mm256_srli_epi64(p0, 12);

        f0 = _mm256_shuffle_epi8(f0, index);

        _mm_storeu_si128(r + 20 * i, _mm256_castsi256_si128(f0));
        _mm_storeu_si128(r + 20 * i + 10, _mm256_extracti128_si256(f0, 1));
    }


}

/*************************************************
* Name:        PQCLEAN_DILITHIUM5_AVX2_polyz_unpack
*
* Description: Unpack polynomial z with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_polyz_unpack(poly *restrict r, const uint8_t *a) {
    unsigned int i;
    __m256i f;
    const __m256i shufbidx = _mm256_set_epi8(-1, 11, 10, 9, -1, 9, 8, 7, -1, 6, 5, 4, -1, 4, 3, 2,
                             -1, 9, 8, 7, -1, 7, 6, 5, -1, 4, 3, 2, -1, 2, 1, 0);
    const __m256i srlvdidx = _mm256_set1_epi64x((uint64_t)4 << 32);
    const __m256i mask = _mm256_set1_epi32(0xFFFFF);
    const __m256i gamma1 = _mm256_set1_epi32(GAMMA1);
    DBENCH_START();

    for (i = 0; i < N / 8; i++) {
        f = _mm256_loadu_si256((__m256i *)&a[20 * i]);
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
* Name:        PQCLEAN_DILITHIUM5_AVX2_polyw1_pack
*
* Description: Bit-pack polynomial w1 with coefficients in [0,15] or [0,43].
*              Input coefficients are assumed to be positive standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYW1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void PQCLEAN_DILITHIUM5_AVX2_polyw1_pack(uint8_t *r, const poly *restrict a) {
    unsigned int i;
    __m256i f0, f1, f2, f3, f4, f5, f6, f7;
    const __m256i shift = _mm256_set1_epi16((16 << 8) + 1);
    const __m256i shufbidx = _mm256_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
                             15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    DBENCH_START();

    for (i = 0; i < N / 64; ++i) {
        f0 = _mm256_load_si256(&a->vec[8 * i + 0]);
        f1 = _mm256_load_si256(&a->vec[8 * i + 1]);
        f2 = _mm256_load_si256(&a->vec[8 * i + 2]);
        f3 = _mm256_load_si256(&a->vec[8 * i + 3]);
        f4 = _mm256_load_si256(&a->vec[8 * i + 4]);
        f5 = _mm256_load_si256(&a->vec[8 * i + 5]);
        f6 = _mm256_load_si256(&a->vec[8 * i + 6]);
        f7 = _mm256_load_si256(&a->vec[8 * i + 7]);
        f0 = _mm256_packus_epi32(f0, f1);
        f1 = _mm256_packus_epi32(f2, f3);
        f2 = _mm256_packus_epi32(f4, f5);
        f3 = _mm256_packus_epi32(f6, f7);
        f0 = _mm256_packus_epi16(f0, f1);
        f1 = _mm256_packus_epi16(f2, f3);
        f0 = _mm256_maddubs_epi16(f0, shift);
        f1 = _mm256_maddubs_epi16(f1, shift);
        f0 = _mm256_packus_epi16(f0, f1);
        f0 = _mm256_permute4x64_epi64(f0, 0xD8);
        f0 = _mm256_shuffle_epi8(f0, shufbidx);
        _mm256_storeu_si256((__m256i *)&r[32 * i], f0);
    }

    DBENCH_STOP(*tpack);
}

void poly_ntt_bo_avx2(poly *a) {

    XRQ_ntt_avx2_bo(a->coeffs);

}

void poly_ntt_so_avx2(poly *a) {

    XRQ_ntt_avx2_so(a->coeffs);

}

void poly_intt_bo_avx2(poly *a) {

    XRQ_intt_avx2_bo(a->coeffs);

}

void poly_intt_so_avx2(poly *a) {

    XRQ_intt_avx2_so(a->coeffs);

}