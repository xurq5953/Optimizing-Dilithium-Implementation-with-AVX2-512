#include "align.h"
#include "consts.h"
#include "fips202x4.h"
#include "ntt.h"
#include "params.h"
#include "poly.h"
#include "rejsample.h"
#include "rounding.h"
#include "polyvec.h"
#include "packing.h"
#include <immintrin.h>
#include <x86intrin.h>
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
* Name:        poly_reduce
*
* Description: Inplace reduction of all coefficients of polynomial to
*              representative in [-6283009,6283007]. Assumes input
*              coefficients to be at most 2^31 - 2^22 - 1 in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_reduce(poly *a) {
    unsigned int i;
    __m256i f, g;
    const __m256i q = _mm256_load_si256(&DILITHIUM2_AVX2_qdata.vec[_8XQ / 8]);
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

/*************************************************
* Name:        poly_addq
*
* Description: For all coefficients of in/out polynomial add Q if
*              coefficient is negative.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_caddq(poly *a) {
    unsigned int i;
    __m256i f, g;
    const __m256i q = _mm256_load_si256(&DILITHIUM2_AVX2_qdata.vec[_8XQ / 8]);
    const __m256i zero = _mm256_setzero_si256();

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        g = _mm256_blendv_epi32(zero, q, f);
        f = _mm256_add_epi32(f, g);
        _mm256_store_si256(&a->vec[i], f);
    }

}

/*************************************************
* Name:        poly_add
*
* Description: Add polynomials. No modular reduction is performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first summand
*              - const poly *b: pointer to second summand
**************************************************/
void poly_add(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    __m256i f, g;

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        g = _mm256_load_si256(&b->vec[i]);
        f = _mm256_add_epi32(f, g);
        _mm256_store_si256(&c->vec[i], f);
    }

}

/*************************************************
* Name:        poly_sub
*
* Description: Subtract polynomials. No modular reduction is
*              performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial to be
*                               subtraced from first input polynomial
**************************************************/
void poly_sub(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    __m256i f, g;

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        g = _mm256_load_si256(&b->vec[i]);
        f = _mm256_sub_epi32(f, g);
        _mm256_store_si256(&c->vec[i], f);
    }

}

/*************************************************
* Name:        poly_shiftl_avx2
*
* Description: Multiply polynomial by 2^D without modular reduction. Assumes
*              input coefficients to be less than 2^{31-D} in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_shiftl_avx2(poly *a) {
    unsigned int i;
    __m256i f;

    for (i = 0; i < N / 8; i++) {
        f = _mm256_load_si256(&a->vec[i]);
        f = _mm256_slli_epi32(f, D);
        _mm256_store_si256(&a->vec[i], f);
    }

}

/*************************************************
* Name:        poly_ntt
*
* Description: Inplace forward NTT. Coefficients can grow by up to
*              8*Q in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/


void poly_ntt_bo_avx2(poly *a) {

    XRQ_ntt_avx2_bo(a->coeffs);

}

void poly_ntt_so_avx2(poly *a) {

    XRQ_ntt_avx2_so(a->coeffs);

}


/*************************************************
* Name:        intt
*
* Description: Inplace inverse NTT and multiplication by 2^{32}.
*              Input coefficients need to be less than Q in absolute
*              value and output coefficients are again bounded by Q.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/

void poly_intt_bo_avx2(poly *a) {

    XRQ_intt_avx2_bo(a->coeffs);

}

void poly_intt_so_avx2(poly *a) {

    XRQ_intt_avx2_so(a->coeffs);

}

void XRQ_AVX2_poly_nttunpack(poly *a) {

    shuffle(a->coeffs);

}

/*************************************************
* Name:        poly_pointwise_montgomery
*
* Description: Pointwise multiplication of polynomials in NTT domain
*              representation and multiplication of resulting polynomial
*              by 2^{-32}.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial
**************************************************/
void poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {

    PQCLEAN_DILITHIUM2_AVX2_pointwise_avx(c->vec, a->vec, b->vec, DILITHIUM2_AVX2_qdata.vec);

}

/*************************************************
* Name:        poly_power2round
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
void poly_power2round(poly *a1, poly *a0, const poly *a) {

    power2round_avx2(a1->vec, a0->vec, a->vec);

}

/*************************************************
* Name:        poly_decompose_avx2
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
void poly_decompose_avx2(poly *a1, poly *a0, const poly *a) {
    decompose_avx2(a1->vec, a0->vec, a->vec);

}

/*************************************************
* Name:        poly_make_hint_avx2
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
unsigned int poly_make_hint_avx2(uint8_t hint[256], const poly *a0, const poly *a1) {
    unsigned int r;

    r = make_hint_avx2(hint, a0->vec, a1->vec);

    return r;
}

/*************************************************
* Name:        poly_use_hint_avx2
*
* Description: Use hint polynomial to correct the high bits of a polynomial.
*
* Arguments:   - poly *b: pointer to output polynomial with corrected high bits
*              - const poly *a: pointer to input polynomial
*              - const poly *h: pointer to input hint polynomial
**************************************************/
void poly_use_hint_avx2(poly *b, const poly *a, const poly *h) {

    use_hint_avx2(b->vec, a->vec, h->vec);

}

/*************************************************
* Name:        poly_chknorm_avx2
*
* Description: Check infinity norm of polynomial against given bound.
*              Assumes input polynomial to be reduced by poly_reduce().
*
* Arguments:   - const poly *a: pointer to polynomial
*              - int32_t B: norm bound
*
* Returns 0 if norm is strictly smaller than B <= (Q-1)/8 and 1 otherwise.
**************************************************/
int poly_chknorm_avx2(const poly *a, int32_t B) {
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

    return ctr;
}

/*************************************************
* Name:       poly_uniform
*
* Description: Sample polynomial with uniformly random coefficients
*              in [0,Q-1] by performing rejection sampling on the
*              output stream of SHAKE256(seed|nonce) or AES256CTR(seed,nonce).
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length SEEDBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/


void poly_uniform_4x_op13(poly *a0,
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
* Name:        poly_uniform_eta
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

static void pack_eta_avx2(uint8_t *r, const uint8_t *pipe) {
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
                       const uint64_t seed[4]) {
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

    for ( int i = 0; i < L; i++) {
        pack_eta_avx2(r + i * POLYETA_PACKEDBYTES, pipe[i].coeffs);
    }

    // sample and pack s2

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


    for (int i = 0; i < K; i++) {
        pack_eta_avx2(r + (L + i) * POLYETA_PACKEDBYTES, pipe[i].coeffs);
    }
}






/*************************************************
* Name:        challenge
*
* Description: Implementation of H. Samples polynomial with TAU nonzero
*              coefficients in {-1,1} using the output stream of
*              SHAKE256(seed).
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const uint8_t mu[]: byte array containing seed of length SEEDBYTES
**************************************************/
void poly_challenge(poly *restrict c, const uint8_t seed[32]) {
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








