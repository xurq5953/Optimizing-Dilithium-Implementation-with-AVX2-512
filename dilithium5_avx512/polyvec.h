#ifndef DILITHIUM5_AVX2_POLYVEC_H
#define DILITHIUM5_AVX2_POLYVEC_H

#include "params.h"
#include "poly.h"
#include <stdint.h>

/* Vectors of polynomials of length L */
typedef struct {
    poly vec[L];
} polyvecl;


void polyvecl_matrix_pointwise_mont(poly *w, const polyvecl *u, const polyvecl *v);


/* Vectors of polynomials of length K */
typedef struct {
    poly vec[K];
} polyveck;


void D5_matrix_expand_row(polyvecl **row, polyvecl buf[2], const uint8_t rho[32], unsigned int i);

void ExpandA_so(polyvecl mat[8], const uint8_t rho[32]);

void polyvecl_ntt_so_avx512(polyvecl *v);

void polyveck_intt_bo(polyveck *v);

void XURQ_AVX512_polyveck_ntt_so(polyveck *v);

void polyvecl_ntt_bo(polyvecl *v);

void XURQ_AVX512_polyveck_ntt(polyveck *v);

void polyveck_caddq_avx512(polyveck *v);

void XURQ_AVX512_polyveck_decompose(polyveck *v1, polyveck *v0, const polyveck *v);

void polyveck_pack_w1_avx512(uint8_t r[1024], const polyveck *w1);

void polyveck_ntt_so_avx512(polyveck *v);

void polyveck_intt_so_avx512(polyveck *v);

void ExpandA_bo(polyvecl mat[8], const uint8_t rho[32]);

void polyvec_matrix_pointwise_mont_avx512(polyveck *t, const polyvecl mat[K], const polyvecl *v);

#endif
