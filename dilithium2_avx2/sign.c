#include "align.h"
#include "fips202.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "randombytes.h"
#include "sign.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>


static inline void
ExpandA_row(polyvecl **row, polyvecl buf[2], const uint8_t rho[32], unsigned int i) {
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



int dilithium2_sign_avx2(uint8_t *sm, size_t *smlen, const uint8_t *m, size_t mlen, const uint8_t *sk) {
    size_t i;

    for (i = 0; i < mlen; ++i) {
        sm[DILITHIUM2_AVX2_CRYPTO_BYTES + mlen - 1 - i] = m[mlen - 1 - i];
    }
    dilithium2_signature_avx2(sm, smlen, sm + DILITHIUM2_AVX2_CRYPTO_BYTES, mlen, sk);
    *smlen += mlen;
    return 0;
}


int dilithium2_keypair_avx2(uint8_t *pk, uint8_t *sk) {
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

    /* Sample short vectors s1 and s2 and Pack secret vectors*/
    ExpandS_with_pack(&s1, &s2, sk + 3 * SEEDBYTES, rhoprime);

    /* Transform s1 */
    polyvecl_ntt_bo(&s1);

    for (i = 0; i < K; i++) {
        /* Expand matrix row */
        ExpandA_row(&row, rowbuf, rho, i);

        /* Compute inner-product */
        polyvecl_pointwise_acc_montgomery(&t1, row, &s1);
        poly_intt_bo_avx2(&t1);

        /* Add error polynomial */
        poly_add(&t1, &t1, &s2.vec[i]);

        /* Round t and pack t1, t0 */
        poly_caddq(&t1);
        poly_power2round(&t1, &t0, &t1);
        polyt1_pack_avx2(pk + SEEDBYTES + i * POLYT1_PACKEDBYTES, &t1);
        polyt0_pack_avx2(sk + 3 * SEEDBYTES + (L + K) * POLYETA_PACKEDBYTES + i * POLYT0_PACKEDBYTES,
                    &t0);
    }

    /* Compute H(rho, t1) and store in secret key */
    shake256(sk + 2 * SEEDBYTES, SEEDBYTES, pk, DILITHIUM2_CRYPTO_PUBLICKEYBYTES);

    return 0;
}


int dilithium2_signature_avx2(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk) {
    unsigned int i, n, pos;
    uint8_t seedbuf[3 * SEEDBYTES + 2 * CRHBYTES];
    uint8_t *rho, *tr, *key, *mu, *rhoprime;
    uint8_t hintbuf[N];
    uint8_t hintbuf2[POLYVECH_PACKEDBYTES];
    uint8_t *hint = sig + SEEDBYTES + L * POLYZ_PACKEDBYTES;
    uint64_t nonce = 0;
    polyvecl mat[K], s1, z;
    polyveck t0, s2, w1;
    poly c, tmp;
    union {
        polyvecl y;
        polyveck w0;
    } tmpv;
    shake256incctx state;

    rho = seedbuf;
    tr = rho + SEEDBYTES;
    key = tr + SEEDBYTES;
    mu = key + SEEDBYTES;
    rhoprime = mu + CRHBYTES;
    unpack_sk(rho, tr, key, &t0, &s1, &s2, sk);

    /* Compute CRH(tr, msg) */
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, tr, SEEDBYTES);
    shake256_inc_absorb(&state, m, mlen);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(mu, CRHBYTES, &state);
    shake256_inc_ctx_release(&state);

    shake256(rhoprime, CRHBYTES, key, SEEDBYTES + CRHBYTES);

    /* Expand matrix and transform vectors */
    ExpandA_shuffled(mat, rho);
    polyvecl_ntt_so(&s1);
    polyveck_ntt_so(&s2);
    polyveck_ntt_so(&t0);


    rej:
    /* Sample intermediate vector y */
    ExpandMask(&z.vec[0], &z.vec[1], &z.vec[2], &z.vec[3],
               rhoprime, nonce, nonce + 1, nonce + 2, nonce + 3);
    nonce += 4;

    /* Matrix-vector product */
    tmpv.y = z;
    polyvecl_ntt_so(&tmpv.y);
    polyvec_matrix_pointwise_montgomery(&w1, mat, &tmpv.y);
    polyveck_intt_so_avx2(&w1);

    /* Decompose w and call the random oracle */
    polyveck_caddq(&w1);
    polyveck_decompose(&w1, &tmpv.w0, &w1);
    polyveck_pack_w1(sig, &w1);

    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, CRHBYTES);
    shake256_inc_absorb(&state, sig, K * POLYW1_PACKEDBYTES);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(sig, SEEDBYTES, &state);
    shake256_inc_ctx_release(&state);
    poly_challenge(&c, sig);
    poly_ntt_so_avx2(&c);

    /* Compute z, reject if it reveals secret */
    for (i = 0; i < L; i++) {
        poly_pointwise_montgomery(&tmp, &c, &s1.vec[i]);
        poly_intt_so_avx2(&tmp);
        poly_add(&z.vec[i], &z.vec[i], &tmp);
        poly_reduce(&z.vec[i]);
        if (poly_chknorm_avx2(&z.vec[i], GAMMA1 - BETA)) {
            goto rej;
        }
    }

    /* Zero hint vector in signature */
    pos = 0;
    memset(hint, 0, OMEGA);

    for (i = 0; i < K; i++) {
        /* Check that subtracting cs2 does not change high bits of w and low bits
         * do not reveal secret information */
        poly_pointwise_montgomery(&tmp, &c, &s2.vec[i]);
        poly_intt_so_avx2(&tmp);
        poly_sub(&tmpv.w0.vec[i], &tmpv.w0.vec[i], &tmp);
        poly_reduce(&tmpv.w0.vec[i]);
        if (poly_chknorm_avx2(&tmpv.w0.vec[i], GAMMA2 - BETA)) {
            goto rej;
        }

        /* Compute hints */
        poly_pointwise_montgomery(&tmp, &c, &t0.vec[i]);
        poly_intt_so_avx2(&tmp);
        poly_reduce(&tmp);
        if (poly_chknorm_avx2(&tmp, GAMMA2)) {
            goto rej;
        }

        poly_add(&tmpv.w0.vec[i], &tmpv.w0.vec[i], &tmp);
        n = poly_make_hint_avx2(hintbuf, &tmpv.w0.vec[i], &w1.vec[i]);
        if (pos + n > OMEGA) {
            goto rej;
        }

        /* Store hints in signature */
//        memcpy(&hint[pos], hintbuf, n);
//        hint[OMEGA + i] = pos = pos + n;

        memcpy(&hintbuf2[pos], hintbuf, n);
        hintbuf2[OMEGA + i] = pos = pos + n;
    }

    /* Pack z into signature */
    for (i = 0; i < L; i++) {
        polyz_pack_avx2(sig + SEEDBYTES + i * POLYZ_PACKEDBYTES, &z.vec[i]);
    }
    memcpy(hint, hintbuf2, POLYVECH_PACKEDBYTES);

    *siglen = DILITHIUM2_AVX2_CRYPTO_BYTES;
    return 0;
}


int dilithium2_verify_avx2(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk) {
    unsigned int i, j, pos = 0;
    /* polyw1_pack_avx2 writes additional 14 bytes */
    ALIGNED_UINT8(K * POLYW1_PACKEDBYTES + 14) buf;
    uint8_t mu[CRHBYTES];
    const uint8_t *hint = sig + SEEDBYTES + L * POLYZ_PACKEDBYTES;
    polyvecl rowbuf[2];
    polyvecl *row = rowbuf;
    polyvecl z;
    poly c, w1, h;
    shake256incctx state;

    if (siglen != DILITHIUM2_AVX2_CRYPTO_BYTES) {
        return -1;
    }

    /* Compute CRH(H(rho, t1), msg) */
    shake256(mu, SEEDBYTES, pk, DILITHIUM2_CRYPTO_PUBLICKEYBYTES);
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, SEEDBYTES);
    shake256_inc_absorb(&state, m, mlen);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(mu, CRHBYTES, &state);
    shake256_inc_ctx_release(&state);

    /* Expand challenge */
    poly_challenge(&c, sig);
    poly_ntt_bo_avx2(&c);

    /* Unpack z; shortness follows from unpacking */
    polyvecl_unpack_z(&z, sig);
    polyvecl_ntt_bo(&z);

    for (i = 0; i < K; i++) {
        /* Expand matrix row */
        ExpandA_row(&row, rowbuf, pk, i);

        /* Compute i-th row of Az - c2^Dt1 */
        polyvecl_pointwise_acc_montgomery(&w1, row, &z);

        XURQ_AVX2_polyt1_unpack(&h, pk + SEEDBYTES + i * POLYT1_PACKEDBYTES);
        poly_shiftl_avx2(&h);
        poly_ntt_bo_avx2(&h);
        poly_pointwise_montgomery(&h, &c, &h);

        poly_sub(&w1, &w1, &h);
        poly_reduce(&w1);
        poly_intt_bo_avx2(&w1);

        /* Get hint polynomial and reconstruct w1 */
        memset(h.vec, 0, sizeof(poly));
        if (hint[OMEGA + i] < pos || hint[OMEGA + i] > OMEGA) {
            return -1;
        }

        for (j = pos; j < hint[OMEGA + i]; ++j) {
                /* Coefficients are ordered for strong unforgeability */
                if (j > pos && hint[j] <= hint[j - 1])
                    return -1;
                h.coeffs[hint[j]] = 1;
        }

        pos = hint[OMEGA + i];

        poly_caddq(&w1);
        poly_use_hint_avx2(&w1, &w1, &h);
        polyw1_pack_avx2(buf.coeffs + i * POLYW1_PACKEDBYTES, &w1);
    }

    /* Extra indices are zero for strong unforgeability */
    for (j = pos; j < OMEGA; ++j) {
        if (hint[j]) return -1;
    }

    /* Call random oracle and verify DILITHIUM2_AVX2_challenge */
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
* Name:        dilithium2_sign_open_avx2
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
int dilithium2_sign_open_avx2(uint8_t *m, size_t *mlen, const uint8_t *sm, size_t smlen, const uint8_t *pk) {
    size_t i;

    if (smlen < DILITHIUM2_AVX2_CRYPTO_BYTES) {
        goto badsig;
    }

    *mlen = smlen - DILITHIUM2_AVX2_CRYPTO_BYTES;
    if (dilithium2_verify_avx2(sm, DILITHIUM2_AVX2_CRYPTO_BYTES, sm + DILITHIUM2_AVX2_CRYPTO_BYTES, *mlen, pk)) {
        goto badsig;
    } else {
        /* All good, copy msg, return 0 */
        for (i = 0; i < *mlen; ++i) {
            m[i] = sm[DILITHIUM2_AVX2_CRYPTO_BYTES + i];
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