#ifndef DILITHIUM2_AVX2_POLY_H
#define DILITHIUM2_AVX2_POLY_H

#include "align.h"
#include "params.h"
#include <stdint.h>

typedef ALIGNED_INT32(N) poly;

void poly_reduce_avx2(poly *a);

void poly_caddq_avx2(poly *a);

void poly_add_avx2(poly *c, const poly *a, const poly *b);

void poly_sub_avx2(poly *c, const poly *a, const poly *b);

void poly_shiftl_avx2(poly *a);

void poly_pointwise_montgomery(poly *c, const poly *a, const poly *b);

void poly_power2round_avx2(poly *a1, poly *a0, const poly *a);

void poly_decompose_avx2(poly *a1, poly *a0, const poly *a);

unsigned int poly_make_hint_avx2(uint8_t hint[256], const poly *a0, const poly *a1);

void poly_use_hint_avx2(poly *b, const poly *a, const poly *h);

int poly_chknorm_avx2(const poly *a, int32_t B);

void poly_challenge(poly *c, const uint8_t seed[32]);

void poly_uniform_4x_op13(poly *a0,
                          poly *a1,
                          poly *a2,
                          poly *a3,
                          const uint8_t seed[32],
                          uint16_t nonce0,
                          uint16_t nonce1,
                          uint16_t nonce2,
                          uint16_t nonce3);


void ExpandMask(poly *a0,
                poly *a1,
                poly *a2,
                poly *a3,
                const uint8_t seed[64],
                uint16_t nonce0,
                uint16_t nonce1,
                uint16_t nonce2,
                uint16_t nonce3);




void poly_ntt_bo_avx2(poly *a);

void poly_ntt_so_avx2(poly *a);

void poly_intt_bo_avx2(poly *a);

void poly_intt_so_avx2(poly *a);



#endif
