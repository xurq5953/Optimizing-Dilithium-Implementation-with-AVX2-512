#include "consts.h"
#include "ntt.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include <stdint.h>

#define UNUSED(x) (void)x

/*************************************************
* Name:        expand_mat
*
* Description: Implementation of ExpandA_shuffled. Generates matrix A with uniformly
*              random coefficients a_{i,j} by performing rejection
*              sampling on the output stream of SHAKE128(rho|j|i)
*              or AES256CTR(rho,j|i).
*
* Arguments:   - polyvecl mat[K]: output matrix
*              - const uint8_t rho[]: byte array containing seed rho
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand(polyvecl mat[K], const uint8_t rho[SEEDBYTES]) {
    polyvecl tmp;
    PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row0(&mat[0], &mat[1], rho);
    PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row1(&mat[1], &mat[2], rho);
    PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row2(&mat[2], &mat[3], rho);
    PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row3(&mat[3], NULL, rho);
    PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row4(&mat[4], &mat[5], rho);
    PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row5(&mat[5], &tmp, rho);
}

void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row0(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_4x(&rowa->vec[0], &rowa->vec[1], &rowa->vec[2], &rowa->vec[3], rho, 0, 1, 2, 3);
    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_4x(&rowa->vec[4], &rowb->vec[0], &rowb->vec[1], &rowb->vec[2], rho, 4, 256, 257, 258);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[0]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[1]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[2]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[3]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[4]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowb->vec[0]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowb->vec[1]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowb->vec[2]);
}

void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row1(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_4x(&rowa->vec[3], &rowa->vec[4], &rowb->vec[0], &rowb->vec[1], rho, 259, 260, 512, 513);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[3]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[4]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowb->vec[0]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowb->vec[1]);
}

void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row2(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_4x(&rowa->vec[2], &rowa->vec[3], &rowa->vec[4], &rowb->vec[0], rho, 514, 515, 516, 768);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[2]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[3]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[4]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowb->vec[0]);
}

void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row3(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    UNUSED(rowb);
    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_4x(&rowa->vec[1], &rowa->vec[2], &rowa->vec[3], &rowa->vec[4], rho, 769, 770, 771, 772);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[1]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[2]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[3]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[4]);
}

void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row4(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_4x(&rowa->vec[0], &rowa->vec[1], &rowa->vec[2], &rowa->vec[3], rho, 1024, 1025, 1026, 1027);
    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_4x(&rowa->vec[4], &rowb->vec[0], &rowb->vec[1], &rowb->vec[2], rho, 1028, 1280, 1281, 1282);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[0]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[1]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[2]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[3]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[4]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowb->vec[0]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowb->vec[1]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowb->vec[2]);
}

void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_expand_row5(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_4x(&rowa->vec[3], &rowa->vec[4], &rowb->vec[0], &rowb->vec[1], rho, 1283, 1284, 1536, 1537);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[3]);
    PQCLEAN_DILITHIUM3_AVX2_poly_nttunpack(&rowa->vec[4]);
}



void XURQ_AVX512_polyvec_matrix_expand_row0(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {

    XURQ_AVX512_poly_uniform_8x(&rowa->vec[0],
                                &rowa->vec[1],
                                &rowa->vec[2],
                                &rowa->vec[3],
                                &rowa->vec[4],
                                &rowb->vec[0],
                                &rowb->vec[1],
                                &rowb->vec[2],
                                rho,
                                0, 1, 2, 3,
                                4, 256, 257, 258);
}

void XURQ_AVX512_polyvec_matrix_expand_row1(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {

    poly t;
    XURQ_AVX512_poly_uniform_8x(&rowa->vec[3], &rowa->vec[4], &rowb->vec[0], &rowb->vec[1],
                                &rowb->vec[2], &rowb->vec[3], &rowb->vec[4], &t,
                                rho,
                                259, 260, 512, 513,
                                514, 515, 516, 0);

}

void XURQ_AVX512_polyvec_matrix_expand_row2(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {
}

void XURQ_AVX512_polyvec_matrix_expand_row3(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {
    XURQ_AVX512_poly_uniform_8x(&rowa->vec[0], &rowa->vec[1], &rowa->vec[2], &rowa->vec[3],
                                &rowa->vec[4],&rowb->vec[0],&rowb->vec[1],&rowb->vec[2],
                                rho,
                                768, 769, 770, 771,
                                772,1024, 1025, 1026);

}

void XURQ_AVX512_polyvec_matrix_expand_row4(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {

    poly t;
    XURQ_AVX512_poly_uniform_8x(&rowa->vec[3], &rowa->vec[4], &rowb->vec[0], &rowb->vec[1],
                                &rowb->vec[2], &rowb->vec[3], &rowb->vec[4], &t,
                                rho,
                                1027, 1028, 1280, 1281,
                                1282, 1283, 1284, 0);
}

void XURQ_AVX512_polyvec_matrix_expand_row5(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {
}

static void XURQ_AVX512_polyvec_matrix_expand_row0_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {
    XURQ_AVX512_poly_uniform_8x(&rowa->vec[0],
                                &rowa->vec[1],
                                &rowa->vec[2],
                                &rowa->vec[3],
                                &rowa->vec[4],
                                &rowb->vec[0],
                                &rowb->vec[1],
                                &rowb->vec[2],
                                rho,
                                0, 1, 2, 3,
                                4, 256, 257, 258);
    shuffle(rowa->vec[0].coeffs);
    shuffle(rowa->vec[1].coeffs);
    shuffle(rowa->vec[2].coeffs);
    shuffle(rowa->vec[3].coeffs);
    shuffle(rowa->vec[4].coeffs);
    shuffle(rowb->vec[0].coeffs);
    shuffle(rowb->vec[1].coeffs);
    shuffle(rowb->vec[2].coeffs);
}

static void XURQ_AVX512_polyvec_matrix_expand_row1_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {
    poly t;
    XURQ_AVX512_poly_uniform_8x(&rowa->vec[3],
                                &rowa->vec[4],
                                &rowb->vec[0],
                                &rowb->vec[1],
                                &rowb->vec[2],
                                &rowb->vec[3],
                                &rowb->vec[4],
                                &t,
                                rho,
                                259, 260, 512, 513,
                                514, 515, 516, 0);
    shuffle(rowa->vec[3].coeffs);
    shuffle(rowa->vec[4].coeffs);
    shuffle(rowb->vec[0].coeffs);
    shuffle(rowb->vec[1].coeffs);
    shuffle(rowb->vec[2].coeffs);
    shuffle(rowb->vec[3].coeffs);
    shuffle(rowb->vec[4].coeffs);
}

static void XURQ_AVX512_polyvec_matrix_expand_row2_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {

}

static void XURQ_AVX512_polyvec_matrix_expand_row3_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {
    XURQ_AVX512_poly_uniform_8x(&rowa->vec[0],
                                &rowa->vec[1],
                                &rowa->vec[2],
                                &rowa->vec[3],
                                &rowa->vec[4],
                                &rowb->vec[0],
                                &rowb->vec[1],
                                &rowb->vec[2],
                                rho,
                                768, 769, 770, 771,
                                772,1024, 1025, 1026);
    shuffle(rowa->vec[0].coeffs);
    shuffle(rowa->vec[1].coeffs);
    shuffle(rowa->vec[2].coeffs);
    shuffle(rowa->vec[3].coeffs);
    shuffle(rowa->vec[4].coeffs);
    shuffle(rowb->vec[0].coeffs);
    shuffle(rowb->vec[1].coeffs);
    shuffle(rowb->vec[2].coeffs);
}

static void XURQ_AVX512_polyvec_matrix_expand_row4_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {
    poly t;
    XURQ_AVX512_poly_uniform_8x(&rowa->vec[3],
                                &rowa->vec[4],
                                &rowb->vec[0],
                                &rowb->vec[1],
                                &rowb->vec[2],
                                &rowb->vec[3],
                                &rowb->vec[4],
                                &t,
                                rho,
                                1027, 1028, 1280, 1281,
                                1282, 1283, 1284, 0);
    shuffle(rowa->vec[3].coeffs);
    shuffle(rowa->vec[4].coeffs);
    shuffle(rowb->vec[0].coeffs);
    shuffle(rowb->vec[1].coeffs);
    shuffle(rowb->vec[2].coeffs);
    shuffle(rowb->vec[3].coeffs);
    shuffle(rowb->vec[4].coeffs);
}

static void XURQ_AVX512_polyvec_matrix_expand_row5_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[32]) {

}

void ExpandA_shuffled(polyvecl mat[6], const uint8_t rho[32]) {
    XURQ_AVX512_polyvec_matrix_expand_row0_shuffled(&mat[0], &mat[1], rho);
    XURQ_AVX512_polyvec_matrix_expand_row1_shuffled(&mat[1], &mat[2], rho);
//    XURQ_AVX512_polyvec_matrix_expand_row2_shuffled(&mat[2], &mat[3], rho);
    XURQ_AVX512_polyvec_matrix_expand_row3_shuffled(&mat[3], &mat[4], rho);
    XURQ_AVX512_polyvec_matrix_expand_row4_shuffled(&mat[4], &mat[5], rho);
//    XURQ_AVX512_polyvec_matrix_expand_row5_shuffled(&mat[5], &tmp, rho);
}

void ExpandA(polyvecl mat[6], const uint8_t rho[32]) {
    XURQ_AVX512_polyvec_matrix_expand_row0(&mat[0], &mat[1], rho);
    XURQ_AVX512_polyvec_matrix_expand_row1(&mat[1], &mat[2], rho);
//    XURQ_AVX512_polyvec_matrix_expand_row2(&mat[2], &mat[3], rho);
    XURQ_AVX512_polyvec_matrix_expand_row3(&mat[3], &mat[4], rho);
    XURQ_AVX512_polyvec_matrix_expand_row4(&mat[4], &mat[5], rho);
//    XURQ_AVX512_polyvec_matrix_expand_row5(&mat[5], &tmp, rho);
}

void PQCLEAN_DILITHIUM3_AVX2_polyvec_matrix_pointwise_montgomery(polyveck *t, const polyvecl mat[K], const polyvecl *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_polyvecl_pointwise_acc_montgomery(&t->vec[i], &mat[i], v);
    }
}

/**************************************************************/
/************ Vectors of polynomials of length L **************/
/**************************************************************/

void PQCLEAN_DILITHIUM3_AVX2_polyvecl_uniform_eta(polyvecl *v, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_uniform_eta(&v->vec[i], seed, nonce++);
    }
}

void PQCLEAN_DILITHIUM3_AVX2_polyvecl_uniform_gamma1(polyvecl *v, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_uniform_gamma1(&v->vec[i], seed, L * nonce + i);
    }
}

void PQCLEAN_DILITHIUM3_AVX2_polyvecl_reduce(polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_reduce(&v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyvecl_add
*
* Description: Add vectors of polynomials of length L.
*              No modular reduction is performed.
*
* Arguments:   - polyvecl *w: pointer to output vector
*              - const polyvecl *u: pointer to first summand
*              - const polyvecl *v: pointer to second summand
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyvecl_add(polyvecl *w, const polyvecl *u, const polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_add(&w->vec[i], &u->vec[i], &v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyvecl_ntt
*
* Description: Forward NTT of all polynomials in vector of length L. Output
*              coefficients can be up to 16*Q larger than input coefficients.
*
* Arguments:   - polyvecl *v: pointer to input/output vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyvecl_ntt(polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_ntt(&v->vec[i]);
    }
}

void XURQ_polyvecl_ntt_bo(polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        XURQ_AVX512_poly_ntt_bo(&v->vec[i]);
    }
}

void polyvecl_ntt_bo(polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        ntt_bo_avx512(v->vec[i].coeffs);
    }
}


void XURQ_AVX512_polyvecl_ntt_so(polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        XURQ_AVX512_poly_ntt_so(&v->vec[i]);
    }
}

void PQCLEAN_DILITHIUM3_AVX2_polyvecl_invntt_tomont(polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_invntt_tomont(&v->vec[i]);
    }
}

void PQCLEAN_DILITHIUM3_AVX2_polyvecl_pointwise_poly_montgomery(polyvecl *r, const poly *a, const polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_pointwise_montgomery(&r->vec[i], a, &v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyvecl_pointwise_acc_montgomery
*
* Description: Pointwise multiply vectors of polynomials of length L, multiply
*              resulting vector by 2^{-32} and add (accumulate) polynomials
*              in it. Input/output vectors are in NTT domain representation.
*
* Arguments:   - poly *w: output polynomial
*              - const polyvecl *u: pointer to first input vector
*              - const polyvecl *v: pointer to second input vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyvecl_pointwise_acc_montgomery(poly *w, const polyvecl *u, const polyvecl *v) {
    PQCLEAN_DILITHIUM3_AVX2_pointwise_acc_avx(w->vec, u->vec->vec, v->vec->vec, PQCLEAN_DILITHIUM3_AVX2_qdata.vec);
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyvecl_chknorm
*
* Description: Check infinity norm of polynomials in vector of length L.
*              Assumes input polyvecl to be reduced by PQCLEAN_DILITHIUM3_AVX2_polyvecl_reduce().
*
* Arguments:   - const polyvecl *v: pointer to vector
*              - int32_t B: norm bound
*
* Returns 0 if norm of all polynomials is strictly smaller than B <= (Q-1)/8
* and 1 otherwise.
**************************************************/
int PQCLEAN_DILITHIUM3_AVX2_polyvecl_chknorm(const polyvecl *v, int32_t bound)  {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        if (PQCLEAN_DILITHIUM3_AVX2_poly_chknorm(&v->vec[i], bound)) {
            return 1;
        }
    }

    return 0;
}

/**************************************************************/
/************ Vectors of polynomials of length K **************/
/**************************************************************/

void PQCLEAN_DILITHIUM3_AVX2_polyveck_uniform_eta(polyveck *v, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_uniform_eta(&v->vec[i], seed, nonce++);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_reduce
*
* Description: Reduce coefficients of polynomials in vector of length K
*              to representatives in [-6283009,6283007].
*
* Arguments:   - polyveck *v: pointer to input/output vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyveck_reduce(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_reduce(&v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_caddq
*
* Description: For all coefficients of polynomials in vector of length K
*              add Q if coefficient is negative.
*
* Arguments:   - polyveck *v: pointer to input/output vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyveck_caddq(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_caddq(&v->vec[i]);
    }
}

void XURQ_AVX512_polyveck_caddq(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        XURQ_AVX512_poly_caddq(&v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_add
*
* Description: Add vectors of polynomials of length K.
*              No modular reduction is performed.
*
* Arguments:   - polyveck *w: pointer to output vector
*              - const polyveck *u: pointer to first summand
*              - const polyveck *v: pointer to second summand
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyveck_add(polyveck *w, const polyveck *u, const polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_add(&w->vec[i], &u->vec[i], &v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_sub
*
* Description: Subtract vectors of polynomials of length K.
*              No modular reduction is performed.
*
* Arguments:   - polyveck *w: pointer to output vector
*              - const polyveck *u: pointer to first input vector
*              - const polyveck *v: pointer to second input vector to be
*                                   subtracted from first input vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyveck_sub(polyveck *w, const polyveck *u, const polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_sub(&w->vec[i], &u->vec[i], &v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_shiftl
*
* Description: Multiply vector of polynomials of Length K by 2^D without modular
*              reduction. Assumes input coefficients to be less than 2^{31-D}.
*
* Arguments:   - polyveck *v: pointer to input/output vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyveck_shiftl(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_shiftl(&v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_ntt
*
* Description: Forward NTT of all polynomials in vector of length K. Output
*              coefficients can be up to 16*Q larger than input coefficients.
*
* Arguments:   - polyveck *v: pointer to input/output vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyveck_ntt(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_ntt(&v->vec[i]);
    }
}

void XURQ_AVX512_polyveck_ntt_so(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        XURQ_AVX512_poly_ntt_so(&v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_invntt_tomont
*
* Description: Inverse NTT and multiplication by 2^{32} of polynomials
*              in vector of length K. Input coefficients need to be less
*              than 2*Q.
*
* Arguments:   - polyveck *v: pointer to input/output vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyveck_invntt_tomont(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_invntt_tomont(&v->vec[i]);
    }
}

void XURQ_AVX512_polyveck_invntt_so(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        intt_so_avx512(v->vec[i].coeffs);
    }
}

void PQCLEAN_DILITHIUM3_AVX2_polyveck_pointwise_poly_montgomery(polyveck *r, const poly *a, const polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_pointwise_montgomery(&r->vec[i], a, &v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_chknorm
*
* Description: Check infinity norm of polynomials in vector of length K.
*              Assumes input polyveck to be reduced by PQCLEAN_DILITHIUM3_AVX2_polyveck_reduce().
*
* Arguments:   - const polyveck *v: pointer to vector
*              - int32_t B: norm bound
*
* Returns 0 if norm of all polynomials are strictly smaller than B <= (Q-1)/8
* and 1 otherwise.
**************************************************/
int PQCLEAN_DILITHIUM3_AVX2_polyveck_chknorm(const polyveck *v, int32_t bound) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        if (PQCLEAN_DILITHIUM3_AVX2_poly_chknorm(&v->vec[i], bound)) {
            return 1;
        }
    }

    return 0;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_power2round
*
* Description: For all coefficients a of polynomials in vector of length K,
*              compute a0, a1 such that a mod^+ Q = a1*2^D + a0
*              with -2^{D-1} < a0 <= 2^{D-1}. Assumes coefficients to be
*              standard representatives.
*
* Arguments:   - polyveck *v1: pointer to output vector of polynomials with
*                              coefficients a1
*              - polyveck *v0: pointer to output vector of polynomials with
*                              coefficients a0
*              - const polyveck *v: pointer to input vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyveck_power2round(polyveck *v1, polyveck *v0, const polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_power2round(&v1->vec[i], &v0->vec[i], &v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_decompose
*
* Description: For all coefficients a of polynomials in vector of length K,
*              compute high and low bits a0, a1 such a mod^+ Q = a1*ALPHA + a0
*              with -ALPHA/2 < a0 <= ALPHA/2 except a1 = (Q-1)/ALPHA where we
*              set a1 = 0 and -ALPHA/2 <= a0 = a mod Q - Q < 0.
*              Assumes coefficients to be standard representatives.
*
* Arguments:   - polyveck *v1: pointer to output vector of polynomials with
*                              coefficients a1
*              - polyveck *v0: pointer to output vector of polynomials with
*                              coefficients a0
*              - const polyveck *v: pointer to input vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyveck_decompose(polyveck *v1, polyveck *v0, const polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_decompose(&v1->vec[i], &v0->vec[i], &v->vec[i]);
    }
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_make_hint
*
* Description: Compute hint vector.
*
* Arguments:   - uint8_t *hint: pointer to output hint array
*              - const polyveck *v0: pointer to low part of input vector
*              - const polyveck *v1: pointer to high part of input vector
*
* Returns number of 1 bits.
**************************************************/
unsigned int PQCLEAN_DILITHIUM3_AVX2_polyveck_make_hint(uint8_t *hint, const polyveck *v0, const polyveck *v1) {
    unsigned int i, n = 0;

    for (i = 0; i < K; ++i) {
        n += PQCLEAN_DILITHIUM3_AVX2_poly_make_hint(&hint[n], &v0->vec[i], &v1->vec[i]);
    }

    return n;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM3_AVX2_polyveck_use_hint
*
* Description: Use hint vector to correct the high bits of input vector.
*
* Arguments:   - polyveck *w: pointer to output vector of polynomials with
*                             corrected high bits
*              - const polyveck *u: pointer to input vector
*              - const polyveck *h: pointer to input hint vector
**************************************************/
void PQCLEAN_DILITHIUM3_AVX2_polyveck_use_hint(polyveck *w, const polyveck *u, const polyveck *h) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_poly_use_hint(&w->vec[i], &u->vec[i], &h->vec[i]);
    }
}

void PQCLEAN_DILITHIUM3_AVX2_polyveck_pack_w1(uint8_t r[K * POLYW1_PACKEDBYTES], const polyveck *w1) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM3_AVX2_polyw1_pack(&r[i * POLYW1_PACKEDBYTES], &w1->vec[i]);
    }
}


void XURQ_AVX512_polyveck_pack_w1(uint8_t r[K * POLYW1_PACKEDBYTES], const polyveck *w1) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        XURQ_D5_AVX512_polyw1_pack(&r[i * POLYW1_PACKEDBYTES], &w1->vec[i]);
    }
}