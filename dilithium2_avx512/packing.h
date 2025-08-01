#ifndef PQCLEAN_DILITHIUM2_AVX2_PACKING_H
#define PQCLEAN_DILITHIUM2_AVX2_PACKING_H
#include "params.h"
#include "polyvec.h"
#include <stdint.h>

void PQCLEAN_DILITHIUM2_AVX2_pack_pk(uint8_t pk[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_PUBLICKEYBYTES], const uint8_t rho[SEEDBYTES], const polyveck *t1);

void PQCLEAN_DILITHIUM2_AVX2_pack_sk(uint8_t sk[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_SECRETKEYBYTES],
                                     const uint8_t rho[SEEDBYTES],
                                     const uint8_t tr[SEEDBYTES],
                                     const uint8_t key[SEEDBYTES],
                                     const polyveck *t0,
                                     const polyvecl *s1,
                                     const polyveck *s2);

void PQCLEAN_DILITHIUM2_AVX2_pack_sig(uint8_t sig[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES], const uint8_t c[SEEDBYTES], const polyvecl *z, const polyveck *h);

void PQCLEAN_DILITHIUM2_AVX2_unpack_pk(uint8_t rho[SEEDBYTES], polyveck *t1, const uint8_t pk[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_PUBLICKEYBYTES]);

void PQCLEAN_DILITHIUM2_AVX2_unpack_sk(uint8_t rho[SEEDBYTES],
                                       uint8_t tr[SEEDBYTES],
                                       uint8_t key[SEEDBYTES],
                                       polyveck *t0,
                                       polyvecl *s1,
                                       polyveck *s2,
                                       const uint8_t sk[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_SECRETKEYBYTES]);

int PQCLEAN_DILITHIUM2_AVX2_unpack_sig(uint8_t c[SEEDBYTES], polyvecl *z, polyveck *h, const uint8_t sig[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES]);


void XURQ_AVX512_pack_sk(uint8_t sk[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_SECRETKEYBYTES],
                         const uint8_t rho[SEEDBYTES],
                         const uint8_t tr[SEEDBYTES],
                         const uint8_t key[SEEDBYTES],
                         const polyveck *t0,
                         const polyvecl *s1,
                         const polyveck *s2);

#endif
