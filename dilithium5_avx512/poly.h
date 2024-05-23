#ifndef DILITHIUM5_AVX2_POLY_H
#define DILITHIUM5_AVX2_POLY_H
#include "align.h"
#include "params.h"
#include "symmetric.h"
#include <stdint.h>

typedef ALIGNED_INT32(N) poly;


void poly_challenge(poly *c, const uint8_t seed[32]);

void XURQ_AVX512_poly_uniform_8x(poly *restrict a0, poly *restrict a1, poly *restrict a2, poly *restrict a3,
                                 poly *restrict a4, poly *restrict a5, poly *restrict a6, poly *restrict a7,
                                 const uint64_t *restrict seed,
                                 uint16_t nonce0, uint16_t nonce1, uint16_t nonce2, uint16_t nonce3,
                                 uint16_t nonce4, uint16_t nonce5, uint16_t nonce6, uint16_t nonce7);

void poly_intt_bo_avx512(poly *a);

void XURQ_AVX512_poly_ntt(poly *a);

void poly_pointwise_montgomery_avx512(poly *c, const poly *a, const poly *b);

void poly_add_avx512(poly *c, const poly *a, const poly *b);

void poly_caddq_avx512(poly *a);

void poly_sub_avx512(poly *c, const poly *a, const poly *b);

void poly_reduce_avx512(poly *a);

int poly_chknorm_avx512(const poly *a, int32_t B);

void poly_power2round_avx512(poly *a1, poly *a0, const poly *a);

void polyt0_pack_avx512(uint8_t r[416], const poly *restrict a);

void polyt1_pack_avx512(uint8_t r[320], const poly *restrict a);

void polyz_pack_avx512(uint8_t r[640], const poly *restrict a);

void polyw1_pack_avx512(uint8_t *r, const poly *restrict a);

void polyz_unpack_avx512(poly *r, const uint8_t *buf);

void polyt1_unpack_avx512(poly *restrict r, const uint8_t a[320]);

void poly_shiftl_avx512(poly *a);

unsigned int poly_make_hint_avx512(uint8_t hint[N], const poly *a0, const poly *a1);

void poly_use_hint_avx512(poly *b, const poly *a, const poly *h);
#endif
