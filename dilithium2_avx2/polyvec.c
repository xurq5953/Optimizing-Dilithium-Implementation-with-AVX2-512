#include "consts.h"
#include "ntt/ntt.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "packing.h"
#include <stdint.h>

#define UNUSED(x) (void)x

/*************************************************
* Name:        ExpandA
*
* Description: Implementation of ExpandA_shuffled. Generates matrix A with uniformly
*              random coefficients a_{i,j} by performing rejection
*              sampling on the output stream of SHAKE128(rho|j|i)
*              or AES256CTR(rho,j|i).
*
* Arguments:   - polyvecl mat[K]: output matrix
*              - const uint8_t rho[]: byte array containing seed rho
**************************************************/


void ExpandA_shuffled(polyvecl mat[4], const uint8_t rho[32]) {
    poly_uniform_4x_op13(&mat[0].vec[0], &mat[0].vec[1], &mat[0].vec[2], &mat[0].vec[3], rho, 0, 1, 2, 3);
    shuffle(mat[0].vec[0].coeffs);
    shuffle(mat[0].vec[1].coeffs);
    shuffle(mat[0].vec[2].coeffs);
    shuffle(mat[0].vec[3].coeffs);
    poly_uniform_4x_op13(&mat[1].vec[0], &mat[1].vec[1], &mat[1].vec[2], &mat[1].vec[3], rho, 256, 257,
                         258, 259);
    shuffle(mat[1].vec[0].coeffs);
    shuffle(mat[1].vec[1].coeffs);
    shuffle(mat[1].vec[2].coeffs);
    shuffle(mat[1].vec[3].coeffs);
    poly_uniform_4x_op13(&mat[2].vec[0], &mat[2].vec[1], &mat[2].vec[2], &mat[2].vec[3], rho, 512, 513,
                         514, 515);
    shuffle(mat[2].vec[0].coeffs);
    shuffle(mat[2].vec[1].coeffs);
    shuffle(mat[2].vec[2].coeffs);
    shuffle(mat[2].vec[3].coeffs);
    poly_uniform_4x_op13(&mat[3].vec[0], &mat[3].vec[1], &mat[3].vec[2], &mat[3].vec[3], rho, 768, 769,
                         770, 771);
    shuffle(mat[3].vec[0].coeffs);
    shuffle(mat[3].vec[1].coeffs);
    shuffle(mat[3].vec[2].coeffs);
    shuffle(mat[3].vec[3].coeffs);
}

void ExpandA_row(polyvecl **row, polyvecl buf[2], const uint8_t rho[32], unsigned int i) {
    switch (i) {
        case 0:
            poly_uniform_4x_op13(&buf[0].vec[0], &buf[0].vec[1], &buf[0].vec[2], &buf[0].vec[3], rho, 0,
                                 1, 2, 3);
            *row = buf;
            break;
        case 1:
            poly_uniform_4x_op13(&buf[1].vec[0], &buf[1].vec[1], &buf[1].vec[2], &buf[1].vec[3], rho,
                                 256, 257,
                                 258, 259);
            *row = buf + 1;
            break;
        case 2:
            poly_uniform_4x_op13(&buf[0].vec[0], &buf[0].vec[1], &buf[0].vec[2], &buf[0].vec[3], rho,
                                 512, 513,
                                 514, 515);
            *row = buf;
            break;
        case 3:
            poly_uniform_4x_op13(&buf[1].vec[0], &buf[1].vec[1], &buf[1].vec[2], &buf[1].vec[3], rho,
                                 768, 769,
                                 770, 771);
            *row = buf + 1;
            break;
    }
}

void polyvec_matrix_pointwise_montgomery(polyveck *t, const polyvecl mat[4], const polyvecl *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        polyvecl_pointwise_acc_montgomery(&t->vec[i], &mat[i], v);
    }
}



void polyvecl_ntt_bo(polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        poly_ntt_bo_avx2(&v->vec[i]);
    }
}

void polyvecl_ntt_so(polyvecl *v) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        poly_ntt_so_avx2(&v->vec[i]);
    }
}


void polyvecl_unpack_z(polyvecl *z, const uint8_t *sig) {
    unsigned int i;

    for (i = 0; i < L; ++i) {
        polyz_unpack_avx2(&z->vec[i], sig + SEEDBYTES + i * POLYZ_PACKEDBYTES);
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
    PQCLEAN_DILITHIUM2_AVX2_pointwise_acc_avx(w->vec, u->vec->vec, v->vec->vec, DILITHIUM2_AVX2_qdata.vec);
}



/**************************************************************/
/************ Vectors of polynomials of length K **************/
/**************************************************************/


/*************************************************
* Name:        polyveck_caddq
*
* Description: For all coefficients of polynomials in vector of length K
*              add Q if coefficient is negative.
*
* Arguments:   - polyveck *v: pointer to input/output vector
**************************************************/
void polyveck_caddq(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        poly_caddq_avx2(&v->vec[i]);
    }
}





void polyveck_ntt_so(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        poly_ntt_so_avx2(&v->vec[i]);
    }
}



void polyveck_intt_so_avx2(polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        poly_intt_so_avx2(&v->vec[i]);
    }
}



/*************************************************
* Name:        polyveck_decompose
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
void polyveck_decompose(polyveck *v1, polyveck *v0, const polyveck *v) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        poly_decompose_avx2(&v1->vec[i], &v0->vec[i], &v->vec[i]);
    }
}



void polyveck_pack_w1(uint8_t r[768], const polyveck *w1) {
    unsigned int i;

    for (i = 0; i < K; ++i) {
        polyw1_pack_avx2(&r[i * POLYW1_PACKEDBYTES], &w1->vec[i]);
    }
}


