#ifndef DILITHIUM2_AVX2_API_H
#define DILITHIUM2_AVX2_API_H

#include <stddef.h>
#include <stdint.h>


#define DILITHIUM2_AVX2_CRYPTO_SECRETKEYBYTES 2528



int dilithium2_keypair_avx2(uint8_t *pk, uint8_t *sk);

int dilithium2_signature_avx2(
    uint8_t *sig, size_t *siglen,
    const uint8_t *m, size_t mlen, const uint8_t *sk);

int dilithium2_verify_avx2(
    const uint8_t *sig, size_t siglen,
    const uint8_t *m, size_t mlen, const uint8_t *pk);

int dilithium2_sign_avx2(
    uint8_t *sm, size_t *smlen,
    const uint8_t *m, size_t mlen, const uint8_t *sk);

int dilithium2_sign_open_avx2(
    uint8_t *m, size_t *mlen,
    const uint8_t *sm, size_t smlen, const uint8_t *pk);

#endif
