#ifndef DILITHIUM5_AVX2_SIGN_H
#define DILITHIUM5_AVX2_SIGN_H
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include <stddef.h>
#include <stdint.h>

int dilithium5_sign_keypair_avx512(uint8_t *pk, uint8_t *sk);

int dilithium5_sign_signature_avx512(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk);

int dilithium5_sign_avx512(uint8_t *sm, size_t *smlen, const uint8_t *m, size_t mlen, const uint8_t *sk);

int dilithium5_sign_verify_avx512(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk);

#endif
