//
// Created by xurq on 2023/3/4.
//

#ifndef OUR_2_AVX512_UNIFORM_ETA_PACK_H
#define OUR_2_AVX512_UNIFORM_ETA_PACK_H

#include <x86intrin.h>
#include <stdint.h>
#include "polyvec.h"

void poly_uniform_eta8x_with_pack(
        uint8_t *r,
        polyvecl *s1,
        polyveck *s2,
        const uint64_t seed[4]);

int rej_eta_with_pipe_avx512(poly *s, uint8_t *p, const uint8_t *buf);



#endif //OUR_2_AVX512_UNIFORM_ETA_PACK_H
