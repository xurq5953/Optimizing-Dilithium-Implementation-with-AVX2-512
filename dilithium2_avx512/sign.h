#ifndef PQCLEAN_DILITHIUM2_AVX2_SIGN_H
#define PQCLEAN_DILITHIUM2_AVX2_SIGN_H
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include <stddef.h>
#include <stdint.h>

void PQCLEAN_DILITHIUM2_AVX2_challenge(poly *c, const uint8_t seed[SEEDBYTES]);

int XURQ_AVX512_crypto_sign_keypair(uint8_t *pk, uint8_t *sk);

int XURQ_AVX512_crypto_sign_signature(uint8_t *sig, size_t *siglen,
                                      const uint8_t *m, size_t mlen,
                                      const uint8_t *sk);

int XURQ_AVX512_crypto_sign(uint8_t *sm, size_t *smlen,
                            const uint8_t *m, size_t mlen,
                            const uint8_t *sk);

int XURQ_AVX512_crypto_sign_verify(const uint8_t *sig, size_t siglen,
                                   const uint8_t *m, size_t mlen,
                                   const uint8_t *pk);

int PQCLEAN_DILITHIUM2_AVX2_crypto_sign_open(uint8_t *m, size_t *mlen,
        const uint8_t *sm, size_t smlen,
        const uint8_t *pk);

void XURQ_polyvec_matrix_expand_so(polyvecl mat[K], const uint8_t rho[SEEDBYTES]);

#endif
