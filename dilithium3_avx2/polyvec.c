#include "consts.h"
#include "ntt/ntt.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "packing.h"
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


void XURQ_AVX2_polyvec_matrix_expand_row0(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    poly_uniform_4x_op13(&rowa->vec[0], &rowa->vec[1], &rowa->vec[2], &rowa->vec[3], rho, 0, 1, 2, 3);
    poly_uniform_4x_op13(&rowa->vec[4], &rowb->vec[0], &rowb->vec[1], &rowb->vec[2], rho, 4, 256, 257, 258);
}

void XURQ_AVX2_polyvec_matrix_expand_row1(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    poly_uniform_4x_op13(&rowa->vec[3], &rowa->vec[4], &rowb->vec[0], &rowb->vec[1], rho, 259, 260, 512, 513);

}

void XURQ_AVX2_polyvec_matrix_expand_row2(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    poly_uniform_4x_op13(&rowa->vec[2], &rowa->vec[3], &rowa->vec[4], &rowb->vec[0], rho, 514, 515, 516, 768);

}

void XURQ_AVX2_polyvec_matrix_expand_row3(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    UNUSED(rowb);
    poly_uniform_4x_op13(&rowa->vec[1], &rowa->vec[2], &rowa->vec[3], &rowa->vec[4], rho, 769, 770, 771, 772);

}

void XURQ_AVX2_polyvec_matrix_expand_row4(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    poly_uniform_4x_op13(&rowa->vec[0], &rowa->vec[1], &rowa->vec[2], &rowa->vec[3], rho, 1024, 1025, 1026, 1027);
    poly_uniform_4x_op13(&rowa->vec[4], &rowb->vec[0], &rowb->vec[1], &rowb->vec[2], rho, 1028, 1280, 1281, 1282);

}

void XURQ_AVX2_polyvec_matrix_expand_row5(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    poly_uniform_4x_op13(&rowa->vec[3], &rowa->vec[4], &rowb->vec[0], &rowb->vec[1], rho, 1283, 1284, 1536, 1537);

}

static void XURQ_AVX2_polyvec_matrix_expand_row0_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    poly_uniform_4x_op13(&rowa->vec[0], &rowa->vec[1], &rowa->vec[2], &rowa->vec[3], rho, 0, 1, 2, 3);
    poly_uniform_4x_op13(&rowa->vec[4], &rowb->vec[0], &rowb->vec[1], &rowb->vec[2], rho, 4, 256, 257, 258);
    shuffle(rowa->vec[0].coeffs);
    shuffle(rowa->vec[1].coeffs);
    shuffle(rowa->vec[2].coeffs);
    shuffle(rowa->vec[3].coeffs);
    shuffle(rowa->vec[4].coeffs);
    shuffle(rowb->vec[0].coeffs);
    shuffle(rowb->vec[1].coeffs);
    shuffle(rowb->vec[2].coeffs);
}

static void XURQ_AVX2_polyvec_matrix_expand_row1_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    poly_uniform_4x_op13(&rowa->vec[3], &rowa->vec[4], &rowb->vec[0], &rowb->vec[1], rho, 259, 260, 512, 513);
    shuffle(rowa->vec[3].coeffs);
    shuffle(rowa->vec[4].coeffs);
    shuffle(rowb->vec[0].coeffs);
    shuffle(rowb->vec[1].coeffs);
}

static void XURQ_AVX2_polyvec_matrix_expand_row2_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    poly_uniform_4x_op13(&rowa->vec[2], &rowa->vec[3], &rowa->vec[4], &rowb->vec[0], rho, 514, 515, 516, 768);
    shuffle(rowa->vec[2].coeffs);
    shuffle(rowa->vec[3].coeffs);
    shuffle(rowa->vec[4].coeffs);
    shuffle(rowb->vec[0].coeffs);
}

static void XURQ_AVX2_polyvec_matrix_expand_row3_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    UNUSED(rowb);
    poly_uniform_4x_op13(&rowa->vec[1], &rowa->vec[2], &rowa->vec[3], &rowa->vec[4], rho, 769, 770, 771, 772);
    shuffle(rowa->vec[1].coeffs);
    shuffle(rowa->vec[2].coeffs);
    shuffle(rowa->vec[3].coeffs);
    shuffle(rowa->vec[4].coeffs);
}

static void XURQ_AVX2_polyvec_matrix_expand_row4_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    poly_uniform_4x_op13(&rowa->vec[0], &rowa->vec[1], &rowa->vec[2], &rowa->vec[3], rho, 1024, 1025, 1026, 1027);
    poly_uniform_4x_op13(&rowa->vec[4], &rowb->vec[0], &rowb->vec[1], &rowb->vec[2], rho, 1028, 1280, 1281, 1282);
    shuffle(rowa->vec[0].coeffs);
    shuffle(rowa->vec[1].coeffs);
    shuffle(rowa->vec[2].coeffs);
    shuffle(rowa->vec[3].coeffs);
    shuffle(rowa->vec[4].coeffs);
    shuffle(rowb->vec[0].coeffs);
    shuffle(rowb->vec[1].coeffs);
    shuffle(rowb->vec[2].coeffs);
}

static void XURQ_AVX2_polyvec_matrix_expand_row5_shuffled(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {
    poly_uniform_4x_op13(&rowa->vec[3], &rowa->vec[4], &rowb->vec[0], &rowb->vec[1], rho, 1283, 1284, 1536, 1537);
    shuffle(rowa->vec[3].coeffs);
    shuffle(rowa->vec[4].coeffs);
}

void ExpandA_shuffled(polyvecl mat[6], const uint8_t rho[32]) {
    polyvecl tmp;
    XURQ_AVX2_polyvec_matrix_expand_row0_shuffled(&mat[0], &mat[1], rho);
    XURQ_AVX2_polyvec_matrix_expand_row1_shuffled(&mat[1], &mat[2], rho);
    XURQ_AVX2_polyvec_matrix_expand_row2_shuffled(&mat[2], &mat[3], rho);
    XURQ_AVX2_polyvec_matrix_expand_row3_shuffled(&mat[3], NULL, rho);
    XURQ_AVX2_polyvec_matrix_expand_row4_shuffled(&mat[4], &mat[5], rho);
    XURQ_AVX2_polyvec_matrix_expand_row5_shuffled(&mat[5], &tmp, rho);
}

void ExpandA_row(polyvecl **row, polyvecl buf[2], const uint8_t rho[32], unsigned int i) {
    switch (i) {
        case 0:
            XURQ_AVX2_polyvec_matrix_expand_row0(buf, buf + 1, rho);
            *row = buf;
            break;
        case 1:
            XURQ_AVX2_polyvec_matrix_expand_row1(buf + 1, buf, rho);
            *row = buf + 1;
            break;
        case 2:
            XURQ_AVX2_polyvec_matrix_expand_row2(buf, buf + 1, rho);
            *row = buf;
            break;
        case 3:
            XURQ_AVX2_polyvec_matrix_expand_row3(buf + 1, buf, rho);
            *row = buf + 1;
            break;
        case 4:
            XURQ_AVX2_polyvec_matrix_expand_row4(buf, buf + 1, rho);
            *row = buf;
            break;
        case 5:
            XURQ_AVX2_polyvec_matrix_expand_row5(buf + 1, buf, rho);
            *row = buf + 1;
            break;
    }
}


void polyvec_matrix_pointwise_montgomery(polyveck *t, const polyvecl mat[6], const polyvecl *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        polyvecl_pointwise_acc_montgomery(&t->vec[i], &mat[i], v);
    }
}

/**************************************************************/
/************ Vectors of polynomials of length L **************/
/**************************************************************/

void polyvecl_ntt_bo_avx2(polyvecl *v) {
    for (int i = 0; i < L; ++i) {
        poly_ntt_bo_avx2(&v->vec[i]);
    }
}

void polyvecl_ntt_so_avx2(polyvecl *v) {
    for (int i = 0; i < L; ++i) {
        poly_ntt_so_avx2(&v->vec[i]);
    }
}

/*************************************************
* Name:        polyvecl_pointwise_acc_montgomery
*
* Description: Pointwise multiply vectors of polynomials of length L, multiply
*              resulting vector by 2^{-32} and add (accumulate) polynomials
*              in it. Input/output vectors are in NTT domain representation.
*
* Arguments:   - poly *w: output polynomial
*              - const polyvecl *u: pointer to first input vector
*              - const polyvecl *v: pointer to second input vector
**************************************************/
void polyvecl_pointwise_acc_montgomery(poly *w, const polyvecl *u, const polyvecl *v) {
    PQCLEAN_DILITHIUM3_AVX2_pointwise_acc_avx(w->vec, u->vec->vec, v->vec->vec, PQCLEAN_DILITHIUM3_AVX2_qdata.vec);
}


/**************************************************************/
/************ Vectors of polynomials of length K **************/
/**************************************************************/



/*************************************************
* Name:        polyveck_caddq_avx2
*
* Description: For all coefficients of polynomials in vector of length K
*              add Q if coefficient is negative.
*
* Arguments:   - polyveck *v: pointer to input/output vector
**************************************************/
void polyveck_caddq_avx2(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        poly_caddq_avx2(&v->vec[i]);
    }
}


void polyveck_ntt_so_avx2(polyveck *v) {
    for (int i = 0; i < K; ++i) {
        poly_ntt_so_avx2(&v->vec[i]);
    }
}


void polyveck_intt_so_avx2(polyveck *v) {

    for (int i = 0; i < K; ++i) {
        poly_intt_so_avx2(&v->vec[i]);
    }
}


/*************************************************
* Name:        polyveck_decompose_avx2
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
void polyveck_decompose_avx2(polyveck *v1, polyveck *v0, const polyveck *v) {
    for (int i = 0; i < K; ++i) {
        poly_decompose_avx2(&v1->vec[i], &v0->vec[i], &v->vec[i]);
    }
}



void polyveck_pack_w1_avx2(uint8_t r[768], const polyveck *w1) {
    for (int i = 0; i < K; ++i) {
        polyw1_pack_avx2(&r[i * POLYW1_PACKEDBYTES], &w1->vec[i]);
    }
}
