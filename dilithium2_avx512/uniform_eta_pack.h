//
// Created by xurq on 2023/3/4.
//

#ifndef OUR_2_AVX512_UNIFORM_ETA_PACK_H
#define OUR_2_AVX512_UNIFORM_ETA_PACK_H

#include <x86intrin.h>
#include <stdint.h>
#include "polyvec.h"

void XURQ_AVX512_poly_uniform_eta8x_with_pack(
        uint8_t r0[POLYETA_PACKEDBYTES],
        uint8_t r1[POLYETA_PACKEDBYTES],
        uint8_t r2[POLYETA_PACKEDBYTES],
        uint8_t r3[POLYETA_PACKEDBYTES],
        uint8_t r4[POLYETA_PACKEDBYTES],
        uint8_t r5[POLYETA_PACKEDBYTES],
        uint8_t r6[POLYETA_PACKEDBYTES],
        uint8_t r7[POLYETA_PACKEDBYTES],
        polyvecl *s1,
        polyveck *s2,
        const uint64_t seed[4]);
#endif //OUR_2_AVX512_UNIFORM_ETA_PACK_H
