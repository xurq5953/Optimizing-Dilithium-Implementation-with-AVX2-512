#ifndef DILITHIUM2_AVX2_SIGN_H
#define DILITHIUM2_AVX2_SIGN_H
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include <stddef.h>
#include <stdint.h>


int dilithium2_keypair_avx2(uint8_t *pk, uint8_t *sk);

int dilithium2_sign_avx2(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk);

int dilithium2_verify_avx2(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk);
#endif
