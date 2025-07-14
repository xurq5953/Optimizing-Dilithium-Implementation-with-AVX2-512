#ifndef DILITHIUM2_AVX2_PACKING_H
#define DILITHIUM2_AVX2_PACKING_H
#include "params.h"
#include "polyvec.h"
#include <stdint.h>



void unpack_sk(uint8_t rho[32],
               uint8_t tr[32],
               uint8_t key[32],
               polyveck *t0,
               polyvecl *s1,
               polyveck *s2,
               const uint8_t sk[2528]);


void polyz_unpack_avx2(poly *r, const uint8_t *a);

void polyw1_pack_avx2(uint8_t *r, const poly *a);

void polyz_pack_avx2(uint8_t r[POLYZ_PACKEDBYTES+7], const poly *restrict a);

void polyt0_pack_avx2(uint8_t r[416], const poly *restrict a);

void polyt1_unpack_avx2(poly *restrict r, const uint8_t a[320]);

void polyt1_pack_avx2(uint8_t r[320], const poly *restrict a);

void polyeta_unpack_avx2(poly *restrict r, const uint8_t a[POLYETA_PACKEDBYTES]);

void polyt0_unpack_avx2(poly *restrict r, const uint8_t a[POLYT0_PACKEDBYTES]);



#endif
