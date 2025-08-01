#ifndef PQCLEAN_DILITHIUM3_AVX2_API_H
#define PQCLEAN_DILITHIUM3_AVX2_API_H

#include <stddef.h>
#include <stdint.h>

#define PQCLEAN_DILITHIUM3_AVX2_CRYPTO_PUBLICKEYBYTES 1952
#define PQCLEAN_DILITHIUM3_AVX2_CRYPTO_SECRETKEYBYTES 4000
#define PQCLEAN_DILITHIUM3_AVX2_CRYPTO_BYTES 3293

#define PQCLEAN_DILITHIUM3_AVX2_CRYPTO_ALGNAME "Dilithium3"

int XURQ_AVX512_crypto_sign_keypair(uint8_t *pk, uint8_t *sk);

int XURQ_AVX512_crypto_sign_signature(
    uint8_t *sig, size_t *siglen,
    const uint8_t *m, size_t mlen, const uint8_t *sk);

int XURQ_AVX512_crypto_sign_verify(
    const uint8_t *sig, size_t siglen,
    const uint8_t *m, size_t mlen, const uint8_t *pk);

int XURQ_AVX512_crypto_sign(
    uint8_t *sm, size_t *smlen,
    const uint8_t *m, size_t mlen, const uint8_t *sk);

int PQCLEAN_DILITHIUM3_AVX2_crypto_sign_open(
    uint8_t *m, size_t *mlen,
    const uint8_t *sm, size_t smlen, const uint8_t *pk);

#endif
