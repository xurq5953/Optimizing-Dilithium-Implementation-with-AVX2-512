#include "consts.h"
#include "ntt.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "rounding.h"
#include <stdint.h>

#define UNUSED(x) (void)x

/*************************************************
* Name:        expand_mat
*
* Description: Implementation of ExpandA. Generates matrix A with uniformly
*              random coefficients a_{i,j} by performing rejection
*              sampling on the output stream of SHAKE128(rho|j|i)
*              or AES256CTR(rho,j|i).
*
* Arguments:   - polyvecl mat[K]: output matrix
*              - const uint8_t rho[]: byte array containing seed rho
**************************************************/


static void XURQ_D5_polyvec_matrix_expand_row0(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {

    XURQ_AVX512_poly_uniform_8x(&rowa->vec[0],
                                &rowa->vec[1],
                                &rowa->vec[2],
                                &rowa->vec[3],
                                &rowa->vec[4],
                                &rowa->vec[5],
                                &rowa->vec[6],
                                &rowb->vec[0], rho,
                                0, 1, 2, 3,
                                4, 5, 6, 256);
}

static void XURQ_D5_polyvec_matrix_expand_row1(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {

    XURQ_AVX512_poly_uniform_8x(&rowa->vec[1],
                                &rowa->vec[2],
                                &rowa->vec[3],
                                &rowa->vec[4],
                                &rowa->vec[5],
                                &rowa->vec[6],
                                &rowb->vec[0],
                                &rowb->vec[1], rho,
                                257, 258, 259, 260,
                                261, 262, 512, 513);
}

static void XURQ_D5_polyvec_matrix_expand_row2(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {

    XURQ_AVX512_poly_uniform_8x(&rowa->vec[2],
                                &rowa->vec[3],
                                &rowa->vec[4],
                                &rowa->vec[5],
                                &rowa->vec[6],
                                &rowb->vec[0],
                                &rowb->vec[1],
                                &rowb->vec[2], rho,
                                514, 515, 516, 517,
                                518, 768, 769, 770);
}

static void XURQ_D5_polyvec_matrix_expand_row3(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {

    XURQ_AVX512_poly_uniform_8x(&rowa->vec[3],
                                &rowa->vec[4],
                                &rowa->vec[5],
                                &rowa->vec[6],
                                &rowb->vec[0],
                                &rowb->vec[1],
                                &rowb->vec[2],
                                &rowb->vec[3], rho,
                                771, 772, 773, 774,
                                1024, 1025, 1026, 1027);
}

static void XURQ_D5_polyvec_matrix_expand_row4(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {

    XURQ_AVX512_poly_uniform_8x(&rowa->vec[4],
                                &rowa->vec[5],
                                &rowa->vec[6],
                                &rowb->vec[0],
                                &rowb->vec[1],
                                &rowb->vec[2],
                                &rowb->vec[3],
                                &rowb->vec[4], rho,
                                1028, 1029, 1030, 1280,
                                1281, 1282, 1283, 1284);
}

static void XURQ_D5_polyvec_matrix_expand_row5(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {

    XURQ_AVX512_poly_uniform_8x(&rowa->vec[5],
                                &rowa->vec[6],
                                &rowb->vec[0],
                                &rowb->vec[1],
                                &rowb->vec[2],
                                &rowb->vec[3],
                                &rowb->vec[4],
                                &rowb->vec[5], rho,
                                1285, 1286, 1536, 1537,
                                1538, 1539, 1540, 1541);
}


static void XURQ_D5_polyvec_matrix_expand_row6(polyvecl *rowa, polyvecl *rowb, const uint8_t rho[SEEDBYTES]) {

    XURQ_AVX512_poly_uniform_8x(&rowa->vec[6],
                                &rowb->vec[0],
                                &rowb->vec[1],
                                &rowb->vec[2],
                                &rowb->vec[3],
                                &rowb->vec[4],
                                &rowb->vec[5],
                                &rowb->vec[6], rho,
                                1542, 1792, 1793, 1794,
                                1795, 1796, 1797, 1798);
}


void D5_matrix_expand_row(polyvecl **row, polyvecl buf[2], const uint8_t rho[32], unsigned int i) {
    switch (i) {
        case 0:
            XURQ_D5_polyvec_matrix_expand_row0(buf, buf + 1, rho);
            *row = buf;
            break;
        case 1:
            XURQ_D5_polyvec_matrix_expand_row1(buf + 1, buf, rho);
            *row = buf + 1;
            break;
        case 2:
            XURQ_D5_polyvec_matrix_expand_row2(buf, buf + 1, rho);
            *row = buf;
            break;
        case 3:
            XURQ_D5_polyvec_matrix_expand_row3(buf + 1, buf, rho);
            *row = buf + 1;
            break;
        case 4:
            XURQ_D5_polyvec_matrix_expand_row4(buf, buf + 1, rho);
            *row = buf;
            break;
        case 5:
            XURQ_D5_polyvec_matrix_expand_row5(buf + 1, buf, rho);
            *row = buf + 1;
            break;
        case 6:
            XURQ_D5_polyvec_matrix_expand_row6(buf, buf + 1, rho);
            *row = buf;
            break;
        case 7:
            *row = buf + 1;
            break;
    }
}



void ExpandA_so(polyvecl mat[8], const uint8_t rho[32]) {
    XURQ_D5_polyvec_matrix_expand_row0(&mat[0], &mat[1], rho);
    XURQ_D5_polyvec_matrix_expand_row1(&mat[1], &mat[2], rho);
    XURQ_D5_polyvec_matrix_expand_row2(&mat[2], &mat[3], rho);
    XURQ_D5_polyvec_matrix_expand_row3(&mat[3], &mat[4], rho);
    XURQ_D5_polyvec_matrix_expand_row4(&mat[4], &mat[5], rho);
    XURQ_D5_polyvec_matrix_expand_row5(&mat[5], &mat[6], rho);
    XURQ_D5_polyvec_matrix_expand_row6(&mat[6], &mat[7], rho);

    for (int i = 0; i < 8; ++i) {
        shuffle(mat[i].vec[0].coeffs);
        shuffle(mat[i].vec[1].coeffs);
        shuffle(mat[i].vec[2].coeffs);
        shuffle(mat[i].vec[3].coeffs);
        shuffle(mat[i].vec[4].coeffs);
        shuffle(mat[i].vec[5].coeffs);
        shuffle(mat[i].vec[6].coeffs);
    }
}


void ExpandA_bo(polyvecl mat[8], const uint8_t rho[32]) {
    XURQ_D5_polyvec_matrix_expand_row0(&mat[0], &mat[1], rho);
    XURQ_D5_polyvec_matrix_expand_row1(&mat[1], &mat[2], rho);
    XURQ_D5_polyvec_matrix_expand_row2(&mat[2], &mat[3], rho);
    XURQ_D5_polyvec_matrix_expand_row3(&mat[3], &mat[4], rho);
    XURQ_D5_polyvec_matrix_expand_row4(&mat[4], &mat[5], rho);
    XURQ_D5_polyvec_matrix_expand_row5(&mat[5], &mat[6], rho);
    XURQ_D5_polyvec_matrix_expand_row6(&mat[6], &mat[7], rho);

}



void polyvec_matrix_pointwise_mont_avx512(polyveck *t, const polyvecl mat[K], const polyvecl *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        polyvecl_matrix_pointwise_mont(&t->vec[i], &mat[i], v);
    }
}

/**************************************************************/
/************ Vectors of polynomials of length L **************/
/**************************************************************/


void polyvecl_ntt_bo(polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        ntt_bo_avx512(v->vec[i].coeffs);
    }
}

void polyvecl_ntt_so_avx512(polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        ntt_so_avx512(v->vec[i].coeffs);
    }
}

void XURQ_AVX512_polyveck_ntt_so(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        ntt_so_avx512(v->vec[i].coeffs);
    }
}



/**************************************************************/
/************ Vectors of polynomials of length K **************/
/**************************************************************/



void polyveck_caddq_avx512(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        poly_caddq_avx512(&v->vec[i]);
    }
}




void polyveck_ntt_so_avx512(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        ntt_so_avx512(v->vec[i].coeffs);
    }
}

void polyveck_intt_bo(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        intt_bo_avx512(v->vec[i].coeffs);
    }
}

void polyveck_intt_so_avx512(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        intt_so_avx512(v->vec[i].coeffs);
    }
}


/*************************************************
* Name:       polyveck_decompose
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

void XURQ_AVX512_polyveck_decompose(polyveck *v1, polyveck *v0, const polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        XURQ_AVX512_decompose(v1->vec[i].vec2, v0->vec[i].vec2, v->vec[i].vec2);
    }
}




void polyveck_pack_w1_avx512(uint8_t r[1024], const polyveck *w1) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        polyw1_pack_avx512(&r[i * POLYW1_PACKEDBYTES], &w1->vec[i]);
    }
}