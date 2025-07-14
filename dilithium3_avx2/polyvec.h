#ifndef DILITHIUM3_AVX2_POLYVEC_H
#define DILITHIUM3_AVX2_POLYVEC_H
#include "params.h"
#include "poly.h"
#include <stdint.h>

/* Vectors of polynomials of length L */
typedef struct {
    poly vec[L];
} polyvecl;


void polyvecl_pointwise_acc_montgomery(poly *w,
                                       const polyvecl *u,
                                       const polyvecl *v);

/* Vectors of polynomials of length K */
typedef struct {
    poly vec[K];
} polyveck;


void polyveck_caddq_avx2(polyveck *v);

void polyveck_decompose_avx2(polyveck *v1, polyveck *v0, const polyveck *v);

void polyveck_pack_w1_avx2(uint8_t r[768], const polyveck *w1);


void polyvec_matrix_pointwise_montgomery(polyveck *t, const polyvecl mat[6], const polyvecl *v);

void polyvecl_ntt_bo_avx2(polyvecl *v);

void polyvecl_ntt_so_avx2(polyvecl *v);

void polyveck_ntt_so_avx2(polyveck *v);

void polyveck_intt_so_avx2(polyveck *v);

void ExpandA_shuffled(polyvecl mat[6], const uint8_t rho[32]);

void XURQ_AVX2_polyvec_matrix_expand_row0(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void XURQ_AVX2_polyvec_matrix_expand_row1(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void XURQ_AVX2_polyvec_matrix_expand_row2(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void XURQ_AVX2_polyvec_matrix_expand_row3(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void XURQ_AVX2_polyvec_matrix_expand_row4(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void XURQ_AVX2_polyvec_matrix_expand_row5(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);

void ExpandA_row(polyvecl **row, polyvecl buf[2], const uint8_t rho[32], unsigned int i);

        void ExpandS_with_pack(polyvecl *s1,
                       polyveck *s2,
                       uint8_t *r,
                       const uint64_t seed[4]);
#endif
