#include "align.h"
#include "fips202.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "randombytes.h"
#include "sign.h"
#include "symmetric.h"
#include "uniform_eta_pack.h"
#include "rejsample.h"
#include "PSPMTEE.h"
#include "ntt.h"
#include <stdint.h>
#include <string.h>

/*************************************************
* Name:        dilithium5_sign_keypair_avx512
*
* Description: Generates public and private key.
*
* Arguments:   - uint8_t *pk: pointer to output public key (allocated
*                             array of DILITHIUM5_AVX2_CRYPTO_PUBLICKEYBYTES bytes)
*              - uint8_t *sk: pointer to output private key (allocated
*                             array of DILITHIUM5_AVX2_CRYPTO_SECRETKEYBYTES bytes)
*
* Returns 0 (success)
**************************************************/
int dilithium5_sign_keypair_avx512(uint8_t *pk, uint8_t *sk) {
    unsigned int i;
    uint8_t seedbuf[2 * SEEDBYTES + CRHBYTES];
    const uint8_t *rho, *rhoprime, *key;
    polyvecl rowbuf[2];
    polyvecl s1, *row = rowbuf;
    polyveck s2;
    poly t1, t0;

    /* Get randomness for rho, rhoprime and key */
    randombytes(seedbuf, SEEDBYTES);
    memset(seedbuf, 0, SEEDBYTES);
    seedbuf[0] = 1;
    shake256(seedbuf, 2 * SEEDBYTES + CRHBYTES, seedbuf, SEEDBYTES);
    rho = seedbuf;
    rhoprime = rho + SEEDBYTES;
    key = rhoprime + CRHBYTES;

    /* Store rho, key */
    memcpy(pk, rho, SEEDBYTES);
    memcpy(sk, rho, SEEDBYTES);
    memcpy(sk + SEEDBYTES, key, SEEDBYTES);

    /* Sample short vectors s1 and s2 */
    /* Pack secret vectors */
    poly_uniform_eta8x_with_pack(sk, &s1, &s2, rhoprime);

    /* Transform s1 */
    polyvecl_ntt_bo(&s1);

    for (i = 0; i < K; i++) {
        /* Expand matrix row */
        D5_matrix_expand_row(&row, rowbuf, rho, i);

        /* Compute inner-product */
        polyvecl_matrix_pointwise_mont(&t1, row, &s1);
        poly_intt_bo_avx512(&t1);

        /* Add error polynomial */
        poly_add_avx512(&t1, &t1, &s2.vec[i]);

        /* Round t and pack t1, t0 */
        poly_caddq_avx512(&t1);
        poly_power2round_avx512(&t1, &t0, &t1);
        polyt1_pack_avx512(pk + SEEDBYTES + i * POLYT1_PACKEDBYTES, &t1);
        polyt0_pack_avx512(sk + 3 * SEEDBYTES + (L + K) * POLYETA_PACKEDBYTES + i * POLYT0_PACKEDBYTES, &t0);
    }

    /* Compute H(rho, t1) and store in secret key */
    shake256(sk + 2 * SEEDBYTES, SEEDBYTES, pk, DILITHIUM5_AVX2_CRYPTO_PUBLICKEYBYTES);

    return 0;
}

/*************************************************
* Name:        dilithium5_sign_signature_avx512
*
* Description: Computes signature.
*
* Arguments:   - uint8_t *sig: pointer to output signature (of length DILITHIUM5_AVX2_CRYPTO_BYTES)
*              - size_t *siglen: pointer to output length of signature
*              - uint8_t *m: pointer to message to be signed
*              - size_t mlen: length of message
*              - uint8_t *sk: pointer to bit-packed secret key
*
* Returns 0 (success)
**************************************************/
int dilithium5_sign_signature_avx512(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen,
                                     const uint8_t *sk) {
    unsigned int i, n, pos;
    uint8_t seedbuf[3 * SEEDBYTES + 2 * CRHBYTES];
    uint8_t *rho, *tr, *key, *mu, *rhoprime;
    ALIGN(64) uint8_t hintbuf[N];
    uint8_t *hint = sig + SEEDBYTES + L * POLYZ_PACKEDBYTES;
    uint64_t nonce = 0;
    polyvecl mat[K], s1, z;
    polyveck t0, s2, w1;
    poly c, tmp;
    polyvecl y;
    polyveck r0;
    shake256incctx state;

    rho = seedbuf;
    tr = rho + SEEDBYTES;
    key = tr + SEEDBYTES;
    mu = key + SEEDBYTES;
    rhoprime = mu + CRHBYTES;
    XURQ_unpack_sk(rho, tr, key, &t0, &s1, &s2, sk);

    /* Compute CRH(tr, msg) */
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, tr, SEEDBYTES);
    shake256_inc_absorb(&state, m, mlen);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(mu, CRHBYTES, &state);
    shake256_inc_ctx_release(&state);

    shake256(rhoprime, CRHBYTES, key, SEEDBYTES + CRHBYTES);

    /* Expand matrix and transform vectors */
    ExpandA_so(mat, rho);


    polyveck_ntt_so_avx512(&t0);

    rej:
    /* Sample intermediate vector y */
    ExpandMask(&z, rhoprime, nonce);
    nonce += 7;

    /* Matrix-vector product */
    y = z;
    polyvecl_ntt_so_avx512(&y);
    polyvec_matrix_pointwise_mont_avx512(&w1, mat, &y);
    polyveck_intt_so_avx512(&w1);

    /* Decompose w and call the random oracle */
    polyveck_caddq_avx512(&w1);
    XURQ_AVX512_polyveck_decompose(&w1, &r0, &w1);
    polyveck_pack_w1_avx512(sig, &w1);

    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, CRHBYTES);
    shake256_inc_absorb(&state, sig, K * POLYW1_PACKEDBYTES);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(sig, SEEDBYTES, &state);
    shake256_inc_ctx_release(&state);
    poly_challenge(&c, sig);


    if (pspm_tee_r0(&c, &s2, &r0, &r0)) goto rej;
    if (pspm_tee_z(&c, &s1, &z, &z)) goto rej;


    pos = 0;
    memset(hint, 0, OMEGA);

    ntt_so_avx512(&c);


    for (i = 0; i < K; i++) {
        /* Compute hints */
        poly_pointwise_montgomery_avx512(&tmp, &c, &t0.vec[i]);
        intt_so_avx512(tmp.coeffs);
        poly_reduce_avx512(&tmp);
        if (poly_chknorm_avx512(&tmp, GAMMA2)) {
            goto rej;
        }

        poly_add_avx512(&r0.vec[i], &r0.vec[i], &tmp);
        n = poly_make_hint_avx512(hintbuf, &r0.vec[i], &w1.vec[i]);
        if (pos + n > OMEGA) {
            goto rej;
        }

        /* Store hints in signature */
        memcpy(&hint[pos], hintbuf, n);
        hint[OMEGA + i] = pos = pos + n;
    }

    /* Pack z into signature */
    for (i = 0; i < L; i++) {
        polyz_pack_avx512(sig + SEEDBYTES + i * POLYZ_PACKEDBYTES, &z.vec[i]);
    }

    *siglen = DILITHIUM5_AVX2_CRYPTO_BYTES;
    return 0;
}

/*************************************************
* Name:        dilithium5_sign_avx512
*
* Description: Compute signed message.
*
* Arguments:   - uint8_t *sm: pointer to output signed message (allocated
*                             array with DILITHIUM5_AVX2_CRYPTO_BYTES + mlen bytes),
*                             can be equal to m
*              - size_t *smlen: pointer to output length of signed
*                               message
*              - const uint8_t *m: pointer to message to be signed
*              - size_t mlen: length of message
*              - const uint8_t *sk: pointer to bit-packed secret key
*
* Returns 0 (success)
**************************************************/
int dilithium5_sign_avx512(uint8_t *sm, size_t *smlen, const uint8_t *m, size_t mlen, const uint8_t *sk) {
    size_t i;

    for (i = 0; i < mlen; ++i) {
        sm[DILITHIUM5_AVX2_CRYPTO_BYTES + mlen - 1 - i] = m[mlen - 1 - i];
    }
    dilithium5_sign_signature_avx512(sm, smlen, sm + DILITHIUM5_AVX2_CRYPTO_BYTES, mlen, sk);
    *smlen += mlen;
    return 0;
}

/*************************************************
* Name:        dilithium5_sign_verify_avx512
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
int dilithium5_sign_verify_avx512(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen,
                                              const uint8_t *pk) {
    unsigned int i, j, pos = 0;
    /* DILITHIUM5_AVX2_polyw1_pack writes additional 14 bytes */
    ALIGNED_UINT8(K * POLYW1_PACKEDBYTES + 14) buf;
    uint8_t mu[CRHBYTES];
    const uint8_t *hint = sig + SEEDBYTES + L * POLYZ_PACKEDBYTES;
    polyvecl rowbuf[2];
    polyvecl *row = rowbuf;
    polyvecl z;
    poly c, w1, h;
    shake256incctx state;
    polyvecl mat[K];

    if (siglen != DILITHIUM5_AVX2_CRYPTO_BYTES) {
        return -1;
    }

    /* Compute CRH(H(rho, t1), msg) */
    shake256(mu, SEEDBYTES, pk, DILITHIUM5_AVX2_CRYPTO_PUBLICKEYBYTES);
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, SEEDBYTES);
    shake256_inc_absorb(&state, m, mlen);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(mu, CRHBYTES, &state);
    shake256_inc_ctx_release(&state);


    poly_challenge(&c, sig);
    ntt_bo_avx512(&c);

    /* Unpack z; shortness follows from unpacking */
    for (i = 0; i < L; i++) {
        polyz_unpack_avx512(&z.vec[i], sig + SEEDBYTES + i * POLYZ_PACKEDBYTES);
        ntt_bo_avx512(&z.vec[i]);
    }

    ExpandA_bo(mat, pk);


    for (i = 0; i < K; i++) {

        /* Compute i-th row of Az - c2^Dt1 */

        polyvecl_matrix_pointwise_mont(&w1, &mat[i], &z);

        polyt1_unpack_avx512(&h, pk + SEEDBYTES + i * POLYT1_PACKEDBYTES);
        poly_shiftl_avx512(&h);
        ntt_bo_avx512(h.coeffs);
        poly_pointwise_montgomery_avx512(&h, &c, &h);

        poly_sub_avx512(&w1, &w1, &h);
        poly_reduce_avx512(&w1);
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

        poly_caddq_avx512(&w1);
        poly_use_hint_avx512(&w1, &w1, &h);
        polyw1_pack_avx512(buf.coeffs + i * POLYW1_PACKEDBYTES, &w1);
    }

    /* Extra indices are zero for strong unforgeability */
    for (j = pos; j < OMEGA; ++j) {
        if (hint[j]) {
            return -1;
        }
    }

    /* Call random oracle and verify DILITHIUM5_AVX2_challenge */
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



