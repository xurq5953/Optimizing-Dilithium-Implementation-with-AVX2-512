#ifndef DILITHIUM5_AVX2_PACKING_H
#define DILITHIUM5_AVX2_PACKING_H
#include "params.h"
#include "polyvec.h"
#include <stdint.h>

void XURQ_unpack_sk(uint8_t rho[SEEDBYTES],
                    uint8_t tr[SEEDBYTES],
                    uint8_t key[SEEDBYTES],
                    polyveck *t0,
                    polyvecl *s1,
                    polyveck *s2,
                    const uint8_t sk[DILITHIUM5_AVX2_CRYPTO_SECRETKEYBYTES]);

void XURQ_AVX512_polyeta_unpack(poly *restrict r, const uint8_t a[POLYETA_PACKEDBYTES]);

void XURQ_AVX512_polyt0_unpack(poly *restrict r, const uint8_t a[POLYT0_PACKEDBYTES]);
#endif
