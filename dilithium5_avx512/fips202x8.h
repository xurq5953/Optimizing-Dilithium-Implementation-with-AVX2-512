//
// Created by xurq on 2022/10/19.
//

#ifndef DILITHIUM2AVX_FIPS202X8_H
#define DILITHIUM2AVX_FIPS202X8_H

#include <x86intrin.h>
#include <stdint.h>
#include "align.h"
#include "keccak8x.h"


#define SHAKE128_RATE 168
#define SHAKE256_RATE 136



typedef struct {
    __m512i s[25];
} keccakx8_state;




void XURQ_AVX512_shake128x8_squeezeblocks(keccakx8_state *state,
                              uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              int nblocks);


void XURQ_AVX512_shake256x8_squeezeblocks(keccakx8_state *state,
                              uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              int nblocks);

#endif //DILITHIUM2AVX_FIPS202X8_H
