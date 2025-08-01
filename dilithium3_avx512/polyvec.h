#ifndef PQCLEAN_DILITHIUM3_AVX2_POLYVEC_H
#define PQCLEAN_DILITHIUM3_AVX2_POLYVEC_H
#include "params.h"
#include "poly.h"
#include <stdint.h>

/* Vectors of polynomials of length L */
typedef struct {
    poly vec[L];
} polyvecl;

void PQCLEAN_DILITHIUM3_AVX2_polyvecl_uniform_eta(polyvecl *v, const uint8_t seed[CRHBYTES], uint16_t nonce);

void PQCLEAN_DILITHIUM3_AVX2_polyvecl_uniform_gamma1(polyvecl *v, const uint8_t seed[CRHBYTES], uint16_t nonce);

void PQCLEAN_DILITHIUM3_AVX2_polyvecl_reduce(polyvecl *v);

void PQCLEAN_DILITHIUM3_AVX2_polyvecl_add(polyvecl *w, const polyvecl *u, const polyvecl *v);

void PQCLEAN_DILITHIUM3_AVX2_polyvecl_ntt(polyvecl *v);
void PQCLEAN_DILITHIUM3_AVX2_polyvecl_invntt_tomont(polyvecl *v);
void PQCLEAN_DILITHIUM3_AVX2_polyvecl_pointwise_poly_montgomery(polyvecl *r, const poly *a, const polyvecl *v);
void PQCLEAN_DILITHIUM3_AVX2_polyvecl_pointwise_acc_montgomery(poly *w,
        const polyvecl *u,
        const polyvecl *v);

int PQCLEAN_DILITHIUM3_AVX2_polyvecl_chknorm(const polyvecl *v, int32_t B);

/* Vectors of polynomials of length K */
typedef struct {
    poly vec[K];
} polyveck;

void PQCLEAN_DILITHIUM3_AVX2_polyveck_uniform_eta(polyveck *v, const uint8_t seed[CRHBYTES], uint16_t nonce);

void PQCLEAN_DILITHIUM3_AVX2_polyveck_reduce(polyveck *v);
void PQCLEAN_DILITHIUM3_AVX2_polyveck_caddq(polyveck *v);

void PQCLEAN_DILITHIUM3_AVX2_polyveck_add(polyveck *w, const polyveck *u, const polyveck *v);
void PQCLEAN_DILITHIUM3_AVX2_polyveck_sub(polyveck *w, const polyveck *u, const polyveck *v);
void PQCLEAN_DILITHIUM3_AVX2_polyveck_shiftl(polyveck *v);

void PQCLEAN_DILITHIUM3_AVX2_polyveck_ntt(polyveck *v);
void PQCLEAN_DILITHIUM3_AVX2_polyveck_invntt_tomont(polyveck *v);
void PQCLEAN_DILITHIUM3_AVX2_polyveck_pointwise_poly_montgomery(polyveck *r, const poly *a, const polyveck *v);

int PQCLEAN_DILITHIUM3_AVX2_polyveck_chknorm(const polyveck *v, int32_t B);

void PQCLEAN_DILITHIUM3_AVX2_polyveck_power2round(polyveck *v1, polyveck *v0, const polyveck *v);
void PQCLEAN_DILITHIUM3_AVX2_polyveck_decompose(polyveck *v1, polyveck *v0, const polyveck *v);
unsigned int PQCLEAN_DILITHIUM3_AVX2_polyveck_make_hint(uint8_t *hint, const polyveck *v0, const polyveck *v1);
void PQCLEAN_DILITHIUM3_AVX2_polyveck_use_hint(polyveck *w, const polyveck *u, const polyveck *h);

void PQCLEAN_DILITHIUM3_AVX2_polyveck_pack_w1(uint8_t r[K * POLYW1_PACKEDBYTES], const polyveck *w1);

void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand(polyvecl mat[K], const uint8_t rho[SEEDBYTES]);

void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row0(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row1(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row2(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row3(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row4(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row5(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row6(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);
void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row7(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]);

void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_pointwise_montgomery(polyveck *t, const polyvecl mat[K], const polyvecl *v);

void XURQ_polyvecl_ntt_bo(polyvecl *v);

void XURQ_AVX512_polyvecl_ntt_so(polyvecl *v);

void XURQ_AVX512_polyveck_ntt_so(polyveck *v);

void XURQ_AVX512_polyveck_invntt_so(polyveck *v);

void ExpandA_shuffled(polyvecl mat[6], const uint8_t rho[32]);

void XURQ_AVX512_polyvec_matrix_expand_row0(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]);
void XURQ_AVX512_polyvec_matrix_expand_row1(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]);
void XURQ_AVX512_polyvec_matrix_expand_row2(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]);
void XURQ_AVX512_polyvec_matrix_expand_row3(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]);
void XURQ_AVX512_polyvec_matrix_expand_row4(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]);
void XURQ_AVX512_polyvec_matrix_expand_row5(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]);

void ExpandS_with_pack(polyvecl *s1,
                       polyveck *s2,
                       uint8_t *r,
                       const uint64_t seed[4]);

void polyvecl_ntt_bo(polyvecl *v);

void ExpandA(polyvecl mat[6], const uint8_t rho[32]);

void XURQ_AVX512_polyveck_caddq(polyveck *v);

void XURQ_AVX512_polyveck_pack_w1(uint8_t r[K * POLYW1_PACKEDBYTES], const polyveck *w1);
#endif
