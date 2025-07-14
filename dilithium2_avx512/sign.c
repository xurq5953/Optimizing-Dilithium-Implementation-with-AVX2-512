#include "align.h"
#include "fips202.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "randombytes.h"
#include "sign.h"
#include "symmetric.h"
#include "align.h"
#include "rejsample.h"
#include "ntt.h"
#include "uniform_eta_pack.h"
#include "PSPMTEE.h"
#include <stdint.h>
#include <string.h>



void XURQ_polyvec_matrix_expand(polyvecl mat[K], const uint8_t rho[SEEDBYTES]) {
    XURQ_AVX512_poly_uniform_8x(&mat[0].vec[0],
                                &mat[0].vec[1],
                                &mat[0].vec[2],
                                &mat[0].vec[3],
                                &mat[1].vec[0],
                                &mat[1].vec[1],
                                &mat[1].vec[2],
                                &mat[1].vec[3], rho,
                                0, 1, 2, 3,
                                256, 257, 258, 259);

    XURQ_AVX512_poly_uniform_8x(&mat[2].vec[0],
                                &mat[2].vec[1],
                                &mat[2].vec[2],
                                &mat[2].vec[3],
                                &mat[3].vec[0],
                                &mat[3].vec[1],
                                &mat[3].vec[2],
                                &mat[3].vec[3], rho,
                                0512, 513, 514, 515,
                                768, 769, 770, 771);

}


void XURQ_polyvec_matrix_expand_so(polyvecl mat[K], const uint8_t rho[SEEDBYTES]) {
    XURQ_AVX512_poly_uniform_8x(&mat[0].vec[0],
                                &mat[0].vec[1],
                                &mat[0].vec[2],
                                &mat[0].vec[3],
                                &mat[1].vec[0],
                                &mat[1].vec[1],
                                &mat[1].vec[2],
                                &mat[1].vec[3], rho,
                                0, 1, 2, 3,
                                256, 257, 258, 259);
    shuffle(&mat[0].vec[0]);
    shuffle(&mat[0].vec[1]);
    shuffle(&mat[0].vec[2]);
    shuffle(&mat[0].vec[3]);
    shuffle(&mat[1].vec[0]);
    shuffle(&mat[1].vec[1]);
    shuffle(&mat[1].vec[2]);
    shuffle(&mat[1].vec[3]);

    XURQ_AVX512_poly_uniform_8x(&mat[2].vec[0],
                                &mat[2].vec[1],
                                &mat[2].vec[2],
                                &mat[2].vec[3],
                                &mat[3].vec[0],
                                &mat[3].vec[1],
                                &mat[3].vec[2],
                                &mat[3].vec[3], rho,
                                0512, 513, 514, 515,
                                768, 769, 770, 771);

    shuffle(&mat[2].vec[0]);
    shuffle(&mat[2].vec[1]);
    shuffle(&mat[2].vec[2]);
    shuffle(&mat[2].vec[3]);
    shuffle(&mat[3].vec[0]);
    shuffle(&mat[3].vec[1]);
    shuffle(&mat[3].vec[2]);
    shuffle(&mat[3].vec[3]);
}

static inline void
XURQ_polyvec_matrix_expand_row_so(polyvecl **row, polyvecl buf[2], const uint8_t rho[SEEDBYTES], unsigned int i) {
    switch (i) {
        case 0:
            XURQ_AVX512_poly_uniform_8x(&buf[0].vec[0],
                                        &buf[0].vec[1],
                                        &buf[0].vec[2],
                                        &buf[0].vec[3],
                                        &buf[1].vec[0],
                                        &buf[1].vec[1],
                                        &buf[1].vec[2],
                                        &buf[1].vec[3], rho,
                                        0, 1, 2, 3,
                                        256, 257, 258, 259);
            shuffle(&buf[0].vec[0]);
            shuffle(&buf[0].vec[1]);
            shuffle(&buf[0].vec[2]);
            shuffle(&buf[0].vec[3]);
            shuffle(&buf[1].vec[0]);
            shuffle(&buf[1].vec[1]);
            shuffle(&buf[1].vec[2]);
            shuffle(&buf[1].vec[3]);
            *row = buf;
            break;
        case 1:
            *row = buf + 1;
            break;
        case 2:
            XURQ_AVX512_poly_uniform_8x(&buf[0].vec[0],
                                        &buf[0].vec[1],
                                        &buf[0].vec[2],
                                        &buf[0].vec[3],
                                        &buf[1].vec[0],
                                        &buf[1].vec[1],
                                        &buf[1].vec[2],
                                        &buf[1].vec[3], rho,
                                        0512, 513, 514, 515,
                                        768, 769, 770, 771);
            shuffle(&buf[0].vec[0]);
            shuffle(&buf[0].vec[1]);
            shuffle(&buf[0].vec[2]);
            shuffle(&buf[0].vec[3]);
            shuffle(&buf[1].vec[0]);
            shuffle(&buf[1].vec[1]);
            shuffle(&buf[1].vec[2]);
            shuffle(&buf[1].vec[3]);
            *row = buf;
            break;
        case 3:
            *row = buf + 1;
            break;
    }
}


static inline void
XURQ_polyvec_matrix_expand_row(polyvecl **row, polyvecl buf[2], const uint8_t rho[SEEDBYTES], unsigned int i) {
    switch (i) {
        case 0:
            XURQ_AVX512_poly_uniform_8x(&buf[0].vec[0],
                                        &buf[0].vec[1],
                                        &buf[0].vec[2],
                                        &buf[0].vec[3],
                                        &buf[1].vec[0],
                                        &buf[1].vec[1],
                                        &buf[1].vec[2],
                                        &buf[1].vec[3], rho,
                                        0, 1, 2, 3,
                                        256, 257, 258, 259);

            *row = buf;
            break;
        case 1:
            *row = buf + 1;
            break;
        case 2:
            XURQ_AVX512_poly_uniform_8x(&buf[0].vec[0],
                                        &buf[0].vec[1],
                                        &buf[0].vec[2],
                                        &buf[0].vec[3],
                                        &buf[1].vec[0],
                                        &buf[1].vec[1],
                                        &buf[1].vec[2],
                                        &buf[1].vec[3], rho,
                                        0512, 513, 514, 515,
                                        768, 769, 770, 771);

            *row = buf;
            break;
        case 3:
            *row = buf + 1;
            break;
    }
}

static inline void XURQ_poly_uniform_gamma1(poly *a0,
                                            poly *a1,
                                            poly *a2,
                                            poly *a3,
                                            const uint8_t seed[64],
                                            uint8_t buf[8][704],
                                            uint16_t nonce,
                                            int flag) {
    if ((flag % 2) == 0) {
        XURQ_AVX512_precompute_gamma1_8x(buf, (uint64_t *) seed,
                                         nonce, nonce + 1, nonce + 2, nonce + 3,
                                         nonce + 4, nonce + 5, nonce + 6, nonce + 7);
        XURQ_AVX512_polyz_unpack4x(a0, a1, a2, a3,
                                   buf[0], buf[1], buf[2], buf[3]);
    } else {
        XURQ_AVX512_polyz_unpack4x(a0, a1, a2, a3,
                                   buf[4], buf[5], buf[6], buf[7]);
    }
}

/*************************************************
* Name:        XURQ_AVX512_crypto_sign_keypair
*
* Description: Generates public and private key.
*
* Arguments:   - uint8_t *pk: pointer to output public key (allocated
*                             array of PQCLEAN_DILITHIUM2_AVX2_CRYPTO_PUBLICKEYBYTES bytes)
*              - uint8_t *sk: pointer to output private key (allocated
*                             array of PQCLEAN_DILITHIUM2_AVX2_CRYPTO_SECRETKEYBYTES bytes)
*
* Returns 0 (success)
**************************************************/
int XURQ_AVX512_crypto_sign_keypair(uint8_t *pk, uint8_t *sk) {
    unsigned int i;
    uint8_t seedbuf[2 * SEEDBYTES + CRHBYTES];
    const uint8_t *rho, *rhoprime, *key;
    polyvecl rowbuf[2];
    polyvecl s1, *row = rowbuf;
    polyveck s2;
    poly t1, t0;

    /* Get randomness for rho, rhoprime and key */
    randombytes(seedbuf, SEEDBYTES);
    shake256(seedbuf, 2 * SEEDBYTES + CRHBYTES, seedbuf, SEEDBYTES);
    rho = seedbuf;
    rhoprime = rho + SEEDBYTES;
    key = rhoprime + CRHBYTES;

    /* Store rho, key */
    memcpy(pk, rho, SEEDBYTES);
    memcpy(sk, rho, SEEDBYTES);
    memcpy(sk + SEEDBYTES, key, SEEDBYTES);

    XURQ_AVX512_poly_uniform_eta8x_with_pack(
            sk + 3 * SEEDBYTES + 0 * POLYETA_PACKEDBYTES,
            sk + 3 * SEEDBYTES + 1 * POLYETA_PACKEDBYTES,
            sk + 3 * SEEDBYTES + 2 * POLYETA_PACKEDBYTES,
            sk + 3 * SEEDBYTES + 3 * POLYETA_PACKEDBYTES,
            sk + 3 * SEEDBYTES + (L + 0) * POLYETA_PACKEDBYTES,
            sk + 3 * SEEDBYTES + (L + 1) * POLYETA_PACKEDBYTES,
            sk + 3 * SEEDBYTES + (L + 2) * POLYETA_PACKEDBYTES,
            sk + 3 * SEEDBYTES + (L + 3) * POLYETA_PACKEDBYTES,
            &s1, &s2, rhoprime);

    /* Transform s1 */
    XURQ_AVX512_polyvecl_ntt(&s1);


    for (i = 0; i < K; i++) {
        /* Expand matrix row */
        XURQ_polyvec_matrix_expand_row(&row, rowbuf, rho, i);

        /* Compute inner-product */
        XURQ_AVX512_polyvecl_pointwise_acc_montgomery(&t1, row, &s1);
        intt_bo_avx512(t1.coeffs);

        /* Add error polynomial */
        XURQ_AVX512_poly_add(&t1, &t1, &s2.vec[i]);

        /* Round t and pack t1, t0 */
        XURQ_AVX512_poly_caddq(&t1);
        XURQ_AVX512_poly_power2round(&t1, &t0, &t1);
        XURQ_AVX512_polyt1_pack(pk + SEEDBYTES + i * POLYT1_PACKEDBYTES, &t1);
        XURQ_AVX512_polyt0_pack(sk + 3 * SEEDBYTES + (L + K) * POLYETA_PACKEDBYTES + i * POLYT0_PACKEDBYTES,
                                            &t0);
    }

    /* Compute H(rho, t1) and store in secret key */
    shake256(sk + 2 * SEEDBYTES, SEEDBYTES, pk, PQCLEAN_DILITHIUM2_AVX2_CRYPTO_PUBLICKEYBYTES);

    return 0;
}

int count = 0;

/*************************************************
* Name:        XURQ_AVX512_crypto_sign_signature
*
* Description: Computes signature.
*
* Arguments:   - uint8_t *sig: pointer to output signature (of length PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES)
*              - size_t *siglen: pointer to output length of signature
*              - uint8_t *m: pointer to message to be signed
*              - size_t mlen: length of message
*              - uint8_t *sk: pointer to bit-packed secret key
*
* Returns 0 (success)
**************************************************/
int XURQ_AVX512_crypto_sign_signature(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen,
                                      const uint8_t *sk) {
    unsigned int n, pos;
    uint8_t seedbuf[3 * SEEDBYTES + 2 * CRHBYTES];
    uint8_t *rho, *tr, *key, *mu, *rhoprime;
    uint8_t hintbuf[N];
    uint8_t *hint = sig + SEEDBYTES + L * POLYZ_PACKEDBYTES;
    uint64_t nonce = 0;
    polyvecl mat[K], s1, z;
    polyveck t0, s2, w1;
    poly c, tmp;
    union {
        polyvecl y;
        polyveck w0;
    } tmpv;
    polyvecl y;
    polyveck r0;
    shake256incctx state;

    rho = seedbuf;
    tr = rho + SEEDBYTES;
    key = tr + SEEDBYTES;
    mu = key + SEEDBYTES;
    rhoprime = mu + CRHBYTES;
    PQCLEAN_DILITHIUM2_AVX2_unpack_sk(rho, tr, key, &t0, &s1, &s2, sk);

    /* Compute CRH(tr, msg) */
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, tr, SEEDBYTES);
    shake256_inc_absorb(&state, m, mlen);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(mu, CRHBYTES, &state);
    shake256_inc_ctx_release(&state);

    shake256(rhoprime, CRHBYTES, key, SEEDBYTES + CRHBYTES);

    /* Expand matrix and transform vectors */
    XURQ_polyvec_matrix_expand_so(mat, rho);

    XURQ_AVX512_polyvecl_ntt_so(&s1);
    XURQ_AVX512_polyveck_ntt_so(&s2);
    XURQ_AVX512_polyveck_ntt_so(&t0);

    int flag = 0;
    ALIGN(64) uint8_t buf[8][704];

rej:

    XURQ_poly_uniform_gamma1(&z.vec[0], &z.vec[1], &z.vec[2], &z.vec[3], rhoprime, buf, nonce, flag);
    nonce += 4;
    flag++;

    /* Matrix-vector product */
    y = z;
    XURQ_AVX512_polyvecl_ntt_so(&y);
    XURQ_AVX512_polyvec_matrix_pointwise_montgomery(&w1, mat, &y);
    XURQ_AVX512_polyveck_invntt_tomont_so(&w1);

    /* Decompose w and call the random oracle */
    XURQ_AVX512_polyveck_caddq(&w1);
    XURQ_AVX512_polyveck_decompose(&w1, &r0, &w1);
    XURQ_AVX512_polyveck_pack_w1(sig, &w1);

    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, CRHBYTES);
    shake256_inc_absorb(&state, sig, K * POLYW1_PACKEDBYTES);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(sig, SEEDBYTES, &state);
    shake256_inc_ctx_release(&state);
    PQCLEAN_DILITHIUM2_AVX2_poly_challenge(&c, sig);
    ntt_so_avx512(c.coeffs);


    /* Zero hint vector in signature */
    pos = 0;
    memset(hint, 0, OMEGA);

    for (int i = 0; i < K; i++) {
        /* Check that subtracting cs2 does not change high bits of w and low bits
         * do not reveal secret information ∥r0∥∞ ≥ γ2 − β*/
        XURQ_AVX512_poly_pointwise_montgomery(&tmp, &c, &s2.vec[i]);
        intt_so_avx512(tmp.coeffs);
        XURQ_AVX512_poly_sub(&r0.vec[i], &r0.vec[i], &tmp);
        XURQ_AVX512_poly_reduce(&r0.vec[i]);
        if (XURQ_AVX512_poly_chknorm(&r0.vec[i], GAMMA2 - BETA)) {
            goto rej;
        }
    }


    /* Compute z, reject if it reveals secret ∥z∥∞ ≥ γ1 − β */
    for (int i = 0; i < L; i++) {
        XURQ_AVX512_poly_pointwise_montgomery(&tmp, &c, &s1.vec[i]);
        intt_so_avx512(tmp.coeffs);
        XURQ_AVX512_poly_add(&z.vec[i], &z.vec[i], &tmp);
        XURQ_AVX512_poly_reduce(&z.vec[i]);
        if (XURQ_AVX512_poly_chknorm(&z.vec[i], GAMMA1 - BETA)) {
            goto rej;
        }
    }


    for (int i = 0; i < K; i++) {


        /* Compute hints ∥ct0∥∞ ≥ γ2*/
        XURQ_AVX512_poly_pointwise_montgomery(&tmp, &c, &t0.vec[i]);
        intt_so_avx512(tmp.coeffs);
        XURQ_AVX512_poly_reduce(&tmp);
        if (XURQ_AVX512_poly_chknorm(&tmp, GAMMA2)) {
            goto rej;
        }

        /*1’s in h is greater than ω*/
        XURQ_AVX512_poly_add(&r0.vec[i], &r0.vec[i], &tmp);
        n = PQCLEAN_DILITHIUM2_AVX2_poly_make_hint(hintbuf, &r0.vec[i], &w1.vec[i]);
        if (pos + n > OMEGA) {
            goto rej;
        }

        /* Store hints in signature */
        memcpy(&hint[pos], hintbuf, n);
        hint[OMEGA + i] = pos = pos + n;
    }

    /* Pack z into signature */
    for (int i = 0; i < L; i++) {
        XURQ_AVX512_polyz_pack(sig + SEEDBYTES + i * POLYZ_PACKEDBYTES, &z.vec[i]);
    }

    count += flag;
    *siglen = PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES;
    return 0;
}

/*************************************************
* Name:        XURQ_AVX512_crypto_sign
*
* Description: Compute signed message.
*
* Arguments:   - uint8_t *sm: pointer to output signed message (allocated
*                             array with PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES + mlen bytes),
*                             can be equal to m
*              - size_t *smlen: pointer to output length of signed
*                               message
*              - const uint8_t *m: pointer to message to be signed
*              - size_t mlen: length of message
*              - const uint8_t *sk: pointer to bit-packed secret key
*
* Returns 0 (success)
**************************************************/
int XURQ_AVX512_crypto_sign(uint8_t *sm, size_t *smlen, const uint8_t *m, size_t mlen, const uint8_t *sk) {
    size_t i;

    for (i = 0; i < mlen; ++i) {
        sm[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES + mlen - 1 - i] = m[mlen - 1 - i];
    }
    XURQ_AVX512_crypto_sign_signature(sm, smlen, sm + PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES, mlen, sk);
    *smlen += mlen;
    return 0;
}

/*************************************************
* Name:        XURQ_AVX512_crypto_sign_verify
*
* Description: Verifies signature.
*
* Arguments:   - uint8_t *m: pointer to input signature
*              - size_t siglen: length of signature
*              - const uint8_t *m: pointer to message
*              - size_t mlen: length of message
*              - const uint8_t *pk: pointer to bit-packed public key
*
* Returns 0 if signature could be verified correctly and -1 otherwise
**************************************************/
int XURQ_AVX512_crypto_sign_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen,
                                   const uint8_t *pk) {
    unsigned int i, j, pos = 0;
    /* PQCLEAN_DILITHIUM2_AVX2_polyw1_pack writes additional 14 bytes */
    ALIGNED_UINT8(K * POLYW1_PACKEDBYTES + 14) buf;
    uint8_t mu[CRHBYTES];
    const uint8_t *hint = sig + SEEDBYTES + L * POLYZ_PACKEDBYTES;
    polyvecl rowbuf[2];
    polyvecl *row = rowbuf;
    polyvecl z;
    poly c, w1, h;
    shake256incctx state;

    if (siglen != PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES) {
        return -1;
    }

    /* Compute CRH(H(rho, t1), msg) */
    shake256(mu, SEEDBYTES, pk, PQCLEAN_DILITHIUM2_AVX2_CRYPTO_PUBLICKEYBYTES);
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, SEEDBYTES);
    shake256_inc_absorb(&state, m, mlen);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(mu, CRHBYTES, &state);
    shake256_inc_ctx_release(&state);

    /* Expand PQCLEAN_DILITHIUM2_AVX2_challenge */
    PQCLEAN_DILITHIUM2_AVX2_poly_challenge(&c, sig);
    ntt_bo_avx512(c.coeffs);

    /* Unpack z; shortness follows from unpacking */
    XURQ_AVX512_polyz_unpack4x(&z.vec[0], &z.vec[1], &z.vec[2], &z.vec[3],
                               sig + SEEDBYTES,
                               sig + SEEDBYTES + POLYZ_PACKEDBYTES,
                               sig + SEEDBYTES + 2 * POLYZ_PACKEDBYTES,
                               sig + SEEDBYTES + 3 * POLYZ_PACKEDBYTES
    );
    for (i = 0; i < L; i++) {
        ntt_bo_avx512(z.vec[i].coeffs);
    }


    for (i = 0; i < K; i++) {
        /* Expand matrix row */
        XURQ_polyvec_matrix_expand_row(&row, rowbuf, pk, i);

        /* Compute i-th row of Az - c2^Dt1 */
        XURQ_AVX512_polyvecl_pointwise_acc_montgomery(&w1, row, &z);

        XURQ_AVX512_polyt1_unpack(&h, pk + SEEDBYTES + i * POLYT1_PACKEDBYTES);
        XURQ_AVX512_poly_shiftl(&h);
        ntt_bo_avx512(h.coeffs);
        XURQ_AVX512_poly_pointwise_montgomery(&h, &c, &h);

        XURQ_AVX512_poly_sub(&w1, &w1, &h);
        XURQ_AVX512_poly_reduce(&w1);
        intt_bo_avx512(w1.coeffs);

        /* Get hint polynomial and reconstruct w1 */
        memset(h.vec, 0, sizeof(poly));
        if (hint[OMEGA + i] < pos || hint[OMEGA + i] > OMEGA) {
            return -1;
        }

        for (j = pos; j < hint[OMEGA + i]; ++j) {
            /* Coefficients are ordered for strong unforgeability */
            if (j > pos && hint[j] <= hint[j - 1]) {
                return -1;
            }
            h.coeffs[hint[j]] = 1;
        }
        pos = hint[OMEGA + i];

        XURQ_AVX512_poly_caddq(&w1);
        PQCLEAN_DILITHIUM2_AVX2_poly_use_hint(&w1, &w1, &h);
        XURQ_AVX512_polyw1_pack(buf.coeffs + i * POLYW1_PACKEDBYTES, &w1);
    }

    /* Extra indices are zero for strong unforgeability */
    for (j = pos; j < OMEGA; ++j) {
        if (hint[j]) {
            return -1;
        }
    }

    /* Call random oracle and verify PQCLEAN_DILITHIUM2_AVX2_challenge */
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, CRHBYTES);
    shake256_inc_absorb(&state, buf.coeffs, K * POLYW1_PACKEDBYTES);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(buf.coeffs, SEEDBYTES, &state);
    shake256_inc_ctx_release(&state);
    for (i = 0; i < SEEDBYTES; ++i) {
        if (buf.coeffs[i] != sig[i]) {
            return -1;
        }
    }

    return 0;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_AVX2_crypto_sign_open
*
* Description: Verify signed message.
*
* Arguments:   - uint8_t *m: pointer to output message (allocated
*                            array with smlen bytes), can be equal to sm
*              - size_t *mlen: pointer to output length of message
*              - const uint8_t *sm: pointer to signed message
*              - size_t smlen: length of signed message
*              - const uint8_t *pk: pointer to bit-packed public key
*
* Returns 0 if signed message could be verified correctly and -1 otherwise
**************************************************/
int
PQCLEAN_DILITHIUM2_AVX2_crypto_sign_open(uint8_t *m, size_t *mlen, const uint8_t *sm, size_t smlen, const uint8_t *pk) {
    size_t i;

    if (smlen < PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES) {
        goto badsig;
    }

    *mlen = smlen - PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES;
    if (XURQ_AVX512_crypto_sign_verify(sm, PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES,
                                       sm + PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES, *mlen, pk)) {
        goto badsig;
    } else {
        /* All good, copy msg, return 0 */
        for (i = 0; i < *mlen; ++i) {
            m[i] = sm[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES + i];
        }
        return 0;
    }

    badsig:
    /* Signature verification failed */
    *mlen = -1;
    for (i = 0; i < smlen; ++i) {
        m[i] = 0;
    }

    return -1;
}
