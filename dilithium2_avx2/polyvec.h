#ifndef DILITHIUM2_AVX2_POLYVEC_H
#define DILITHIUM2_AVX2_POLYVEC_H
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

void polyvecl_unpack_z(polyvecl *z, const uint8_t *sig);

/* Vectors of polynomials of length K */
typedef struct {
    poly vec[K];
} polyveck;


void polyveck_caddq(polyveck *v);

void polyveck_decompose(polyveck *v1, polyveck *v0, const polyveck *v);

void polyveck_pack_w1(uint8_t r[768], const polyveck *w1);

void polyvec_matrix_pointwise_montgomery(polyveck *t, const polyvecl mat[4], const polyvecl *v);

void ExpandA_shuffled(polyvecl mat[4], const uint8_t rho[32]);

void polyvecl_ntt_bo(polyvecl *v);

void polyvecl_ntt_so(polyvecl *v);

void polyveck_ntt_so(polyveck *v);

void polyveck_intt_so_avx2(polyveck *v);

void ExpandS_with_pack(polyvecl *s1,
                       polyveck *s2,
                       uint8_t *r,
                       const uint64_t seed[4]);
#endif
