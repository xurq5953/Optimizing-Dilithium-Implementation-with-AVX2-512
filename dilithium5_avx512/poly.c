#include "align.h"
#include "consts.h"
#include "ntt.h"
#include "params.h"
#include "poly.h"
#include "rejsample.h"
#include "rounding.h"
#include "symmetric.h"
#include "fips202x8.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>




void poly_reduce_avx512(poly *a) {
    unsigned int i;
    __m512i f, g;
    const __m512i q = _mm512_load_si512(&qdata2[Q16X]);
    const __m512i off = _mm512_set1_epi32(1 << 22);

    for (i = 0; i < N / 16; i++) {
        f = _mm512_load_si512(&a->vec2[i]);
        g = _mm512_add_epi32(f, off);
        g = _mm512_srai_epi32(g, 23);
        g = _mm512_mullo_epi32(g, q);
        f = _mm512_sub_epi32(f, g);
        _mm512_store_si512(&a->vec2[i], f);
    }

}


void poly_caddq_avx512(poly *a) {
    unsigned int i;
    __m512i f0, f1, f2, f3;
    const __m512i q = _mm512_set1_epi32(Q);
    const __m512i zero = _mm512_setzero_si512();


    for (i = 0; i < 4; i++) {
        f0 = _mm512_load_si512(a->vec2 + 4 * i);
        f1 = _mm512_load_si512(a->vec2 + 4 * i + 1);
        f2 = _mm512_load_si512(a->vec2 + 4 * i + 2);
        f3 = _mm512_load_si512(a->vec2 + 4 * i + 3);
        f0 = _mm512_mask_add_epi32(f0, _mm512_cmp_epi32_mask(f0, zero, 1), f0, q);
        f1 = _mm512_mask_add_epi32(f1, _mm512_cmp_epi32_mask(f1, zero, 1), f1, q);
        f2 = _mm512_mask_add_epi32(f2, _mm512_cmp_epi32_mask(f2, zero, 1), f2, q);
        f3 = _mm512_mask_add_epi32(f3, _mm512_cmp_epi32_mask(f3, zero, 1), f3, q);
        _mm512_store_si512(a->vec2 + 4 * i, f0);
        _mm512_store_si512(a->vec2 + 4 * i + 1, f1);
        _mm512_store_si512(a->vec2 + 4 * i + 2, f2);
        _mm512_store_si512(a->vec2 + 4 * i + 3, f3);
    }

}



void poly_add_avx512(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    __m512i vec0, vec1;
    for (i = 0; i < N; i += 16) {
        vec0 = _mm512_load_si512((__m512i *) &a->coeffs[i]);
        vec1 = _mm512_load_si512((__m512i *) &b->coeffs[i]);
        vec0 = _mm512_add_epi32(vec0, vec1);
        _mm512_store_si512((__m512i *) &c->coeffs[i], vec0);
    }
}


void poly_sub_avx512(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    __m512i vec0, vec1, vec2, vec3;
    for (i = 0; i < N / 32; i++) {
        vec0 = _mm512_load_si512((__m512i *) &a->vec2[2 * i]);
        vec1 = _mm512_load_si512((__m512i *) &b->vec2[2 * i]);
        vec2 = _mm512_load_si512((__m512i *) &a->vec2[2 * i + 1]);
        vec3 = _mm512_load_si512((__m512i *) &b->vec2[2 * i + 1]);
        vec0 = _mm512_sub_epi32(vec0, vec1);
        vec2 = _mm512_sub_epi32(vec2, vec3);
        _mm512_store_si512((__m512i *) &c->vec2[2 * i], vec0);
        _mm512_store_si512((__m512i *) &c->vec2[2 * i + 1], vec2);
    }
}


void poly_shiftl_avx512(poly *a) {
    unsigned int i;
    __m512i f;

    for (i = 0; i < N / 16; i++) {
        f = _mm512_load_si512(a->coeffs + 16 * i);
        f = _mm512_slli_epi32(f, D);
        _mm512_store_si512(a->coeffs + 16 * i, f);
    }

}



void XURQ_AVX512_poly_ntt(poly *a) {

    ntt_so_avx512(a->coeffs);

}


void poly_intt_bo_avx512(poly *a) {

    intt_bo_avx512(a->coeffs);

}



void poly_pointwise_montgomery_avx512(poly *c, const poly *a, const poly *b) {
    pointwise_avx512(c->coeffs, a->coeffs, b->coeffs);
}



void poly_power2round_avx512(poly *a1, poly *a0, const poly *a) {
    XURQ_AVX512_power2round_avx(a1->vec2, a0->vec2, a->vec2);
}





unsigned int poly_make_hint_avx512(uint8_t hint[N], const poly *a0, const poly *a1) {
    unsigned int r;

    r = make_hint_avx512(hint, a0->vec2, a1->vec2);

    return r;
}


void poly_use_hint_avx512(poly *b, const poly *a, const poly *h) {

    use_hint_avx512(b->vec2, a->vec2, h->vec2);

}


int poly_chknorm_avx512(const poly *a, int32_t B) {
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

    return ctr;
}



/*************************************************
* Name:       poly_uniform_gamma1
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


ALIGN(64) static const uint16_t idx_t_p0[32] = {
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

ALIGN(64) static const uint16_t idx_t_p1[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
};

void polyt1_pack_avx512(uint8_t r[320], const poly *restrict a) {
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


void polyt1_unpack_avx512(poly *restrict r, const uint8_t a[320]) {

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

ALIGN(64) static const uint16_t idx_t0_p3[32] = {
        0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0,
        0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0,
};

void polyt0_pack_avx512(uint8_t r[416], const poly *restrict a) {
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

        p0 = _mm512_srli_epi32(_mm512_andnot_epi32(mask1, f0), 3);
        f0 = _mm512_ternarylogic_epi32(p0, mask1, f0, 0x78);
        p0 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f0), 6);
        f0 = _mm512_ternarylogic_epi32(p0, mask0, f0, 0x78);

        p0 = _mm512_maskz_permutexvar_epi16(0x08080808, idx3, f0);
        p0 = _mm512_slli_epi16(p0, 4);
        f0 |= p0;
        f0 = _mm512_mask_srli_epi64(f0, 0xaa, f0, 12);

        _mm512_mask_compressstoreu_epi8(r + ctr, 0x1fff1fff1fff1fff, f0);

        ctr += 52;
        pos += 32;
    }

}


void polyz_pack_avx512(uint8_t r[640], const poly *restrict a) {
    int pos = 0;
    int ctr = 0;
    __m512i f0, f1, f2, f3;
    __m512i p0, p1, p2, p3;

    const __m512i mask0 = _mm512_set1_epi64(0xffffffff);
    const __m512i gamma = _mm512_set1_epi32(GAMMA1);

    for (int i = 0; i < 4; ++i) {
        f0 = _mm512_loadu_si512(a->coeffs + pos);
        f1 = _mm512_loadu_si512(a->coeffs + pos + 16);
        f2 = _mm512_loadu_si512(a->coeffs + pos + 32);
        f3 = _mm512_loadu_si512(a->coeffs + pos + 48);

        f0 = _mm512_sub_epi32(gamma, f0);
        f1 = _mm512_sub_epi32(gamma, f1);
        f2 = _mm512_sub_epi32(gamma, f2);
        f3 = _mm512_sub_epi32(gamma, f3);

        p0 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f0), 12);
        p1 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f1), 12);
        p2 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f2), 12);
        p3 = _mm512_srli_epi64(_mm512_andnot_epi64(mask0, f3), 12);

        f0 = _mm512_ternarylogic_epi32(p0, f0, mask0, 0x78);
        f1 = _mm512_ternarylogic_epi32(p1, f1, mask0, 0x78);
        f2 = _mm512_ternarylogic_epi32(p2, f2, mask0, 0x78);
        f3 = _mm512_ternarylogic_epi32(p3, f3, mask0, 0x78);

        _mm512_mask_compressstoreu_epi8(r + ctr, 0x1f1f1f1f1f1f1f1f, f0);
        _mm512_mask_compressstoreu_epi8(r + ctr + 40, 0x1f1f1f1f1f1f1f1f, f1);
        _mm512_mask_compressstoreu_epi8(r + ctr + 80, 0x1f1f1f1f1f1f1f1f, f2);
        _mm512_mask_compressstoreu_epi8(r + ctr + 120, 0x1f1f1f1f1f1f1f1f, f3);

        pos += 64;
        ctr += 160;

    }

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


ALIGN(64) static const uint8_t idx_w1_p[64] = {
        0, 1, 16, 17, 32, 33, 48, 49,
        2, 3, 18, 19, 34, 35, 50, 51,
        4, 5, 20, 21, 36, 37, 52, 53,
        6, 7, 22, 23, 38, 39, 54, 55,
        8, 9, 24, 25, 40, 41, 56, 57,
        10, 11, 26, 27, 42, 43, 58, 59,
        12, 13, 28, 29, 44, 45, 60, 61,
        14, 15, 30, 31, 46, 47, 62, 63
};

void polyw1_pack_avx512(uint8_t *r, const poly *restrict a) {
    __m512i f0, f1, f2, f3, f4, f5, f6, f7;
    const __m512i shift1 = _mm512_set1_epi16((16 << 8) + 1);
    const __m512i idx0 = _mm512_load_si512(idx_w1_p);

    for (int i = 0; i < 2; i++) {
        f0 = _mm512_load_si512(&a->vec2[8 * i + 0]);
        f1 = _mm512_load_si512(&a->vec2[8 * i + 1]);
        f2 = _mm512_load_si512(&a->vec2[8 * i + 2]);
        f3 = _mm512_load_si512(&a->vec2[8 * i + 3]);
        f4 = _mm512_load_si512(&a->vec2[8 * i + 4]);
        f5 = _mm512_load_si512(&a->vec2[8 * i + 5]);
        f6 = _mm512_load_si512(&a->vec2[8 * i + 6]);
        f7 = _mm512_load_si512(&a->vec2[8 * i + 7]);
        f0 = _mm512_packus_epi32(f0, f1);
        f1 = _mm512_packus_epi32(f2, f3);
        f2 = _mm512_packus_epi32(f4, f5);
        f3 = _mm512_packus_epi32(f6, f7);
        f0 = _mm512_packus_epi16(f0, f1);
        f1 = _mm512_packus_epi16(f2, f3);
        f0 = _mm512_maddubs_epi16(f0, shift1);
        f1 = _mm512_maddubs_epi16(f1, shift1);
        f0 = _mm512_packus_epi16(f0, f1);

        f0 = _mm512_permutexvar_epi8(idx0, f0);
        _mm512_storeu_si512(r + 64 * i, f0);
    }

}
