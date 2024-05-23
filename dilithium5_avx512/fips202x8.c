//
// Created by xurq on 2022/10/19.
//

#include <string.h>
#include <stdio.h>
#include "fips202x8.h"



/**
 *  only works for bits of in_i < shake128_rate
 * @param state
 * @param shake_rate  bytes of shake rate
 * @param in0
 * @param in1
 * @param in2
 * @param in3
 * @param in4
 * @param in5
 * @param in6
 * @param in7
 * @param inlen  bytes of in_i
 */



void XURQ_AVX512_shake128x8_squeezeblocks(keccakx8_state *state,
                              uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              int nblocks) {
    uint8_t *temp;
    __m512i t, t4, t5, t6, t7;
    __m512i t0, t1, t2, t3;
    __m256i z0,z1,z2,z3;
    for (int i = 0; i < nblocks; ++i) {
        XURQ_keccak8x_function(state->s);

        t0 = _mm512_unpacklo_epi64(state->s[0], state->s[1]); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(state->s[0], state->s[1]); //77 55 33 11
        t2 = _mm512_unpacklo_epi64(state->s[2], state->s[3]); //66 44 22 00
        t3 = _mm512_unpackhi_epi64(state->s[2], state->s[3]); //77 55 33 11
        t4 = _mm512_unpacklo_epi64(state->s[4], state->s[5]); //66 44 22 00
        t5 = _mm512_unpackhi_epi64(state->s[4], state->s[5]); //77 55 33 11
        t6 = _mm512_unpacklo_epi64(state->s[6], state->s[7]); //66 44 22 00
        t7 = _mm512_unpackhi_epi64(state->s[6], state->s[7]); //77 55 33 11

        t = _mm512_shuffle_i32x4(t0, t2, 0x44);//22 00 22 00
        t2 = _mm512_shuffle_i32x4(t0, t2, 0xee);//66 44 66 44
        t0 = t;
        t = _mm512_shuffle_i32x4(t4, t6, 0x44);//22 00 22 00
        t6 = _mm512_shuffle_i32x4(t4, t6, 0xee);//66 44 66 44
        t4 = t;
        t = _mm512_shuffle_i32x4(t1, t3, 0x44); //33 11 33 11
        t3 = _mm512_shuffle_i32x4(t1, t3, 0xee);//77 55 77 55
        t1 = t;
        t = _mm512_shuffle_i32x4(t5, t7, 0x44); //33 11 33 11
        t7 = _mm512_shuffle_i32x4(t5, t7, 0xee);//77 55 77 55
        t5 = t;
        _mm512_storeu_si512(out0, _mm512_shuffle_i32x4(t0, t4, 0x88));
        _mm512_storeu_si512(out1, _mm512_shuffle_i32x4(t1, t5, 0x88));
        _mm512_storeu_si512(out2, _mm512_shuffle_i32x4(t0, t4, 0xdd));
        _mm512_storeu_si512(out3, _mm512_shuffle_i32x4(t1, t5, 0xdd));
        _mm512_storeu_si512(out4, _mm512_shuffle_i32x4(t2, t6, 0x88));
        _mm512_storeu_si512(out5, _mm512_shuffle_i32x4(t3, t7, 0x88));
        _mm512_storeu_si512(out6, _mm512_shuffle_i32x4(t2, t6, 0xdd));
        _mm512_storeu_si512(out7, _mm512_shuffle_i32x4(t3, t7, 0xdd));

        t0 = _mm512_unpacklo_epi64(state->s[8], state->s[9]); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(state->s[8], state->s[9]); //77 55 33 11
        t2 = _mm512_unpacklo_epi64(state->s[10], state->s[11]); //66 44 22 00
        t3 = _mm512_unpackhi_epi64(state->s[10], state->s[11]); //77 55 33 11
        t4 = _mm512_unpacklo_epi64(state->s[12], state->s[13]); //66 44 22 00
        t5 = _mm512_unpackhi_epi64(state->s[12], state->s[13]); //77 55 33 11
        t6 = _mm512_unpacklo_epi64(state->s[14], state->s[15]); //66 44 22 00
        t7 = _mm512_unpackhi_epi64(state->s[14], state->s[15]); //77 55 33 11

        t = _mm512_shuffle_i32x4(t0, t2, 0x44);//22 00 22 00
        t2 = _mm512_shuffle_i32x4(t0, t2, 0xee);//66 44 66 44
        t0 = t;
        t = _mm512_shuffle_i32x4(t4, t6, 0x44);//22 00 22 00
        t6 = _mm512_shuffle_i32x4(t4, t6, 0xee);//66 44 66 44
        t4 = t;
        t = _mm512_shuffle_i32x4(t1, t3, 0x44); //33 11 33 11
        t3 = _mm512_shuffle_i32x4(t1, t3, 0xee);//77 55 77 55
        t1 = t;
        t = _mm512_shuffle_i32x4(t5, t7, 0x44); //33 11 33 11
        t7 = _mm512_shuffle_i32x4(t5, t7, 0xee);//77 55 77 55
        t5 = t;
        _mm512_storeu_si512(out0 + 64, _mm512_shuffle_i32x4(t0, t4, 0x88));
        _mm512_storeu_si512(out1 + 64, _mm512_shuffle_i32x4(t1, t5, 0x88));
        _mm512_storeu_si512(out2 + 64, _mm512_shuffle_i32x4(t0, t4, 0xdd));
        _mm512_storeu_si512(out3 + 64, _mm512_shuffle_i32x4(t1, t5, 0xdd));
        _mm512_storeu_si512(out4 + 64, _mm512_shuffle_i32x4(t2, t6, 0x88));
        _mm512_storeu_si512(out5 + 64, _mm512_shuffle_i32x4(t3, t7, 0x88));
        _mm512_storeu_si512(out6 + 64, _mm512_shuffle_i32x4(t2, t6, 0xdd));
        _mm512_storeu_si512(out7 + 64, _mm512_shuffle_i32x4(t3, t7, 0xdd));

        t0 = _mm512_unpacklo_epi64(state->s[16], state->s[17]); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(state->s[16], state->s[17]); //77 55 33 11
        t2 = _mm512_unpacklo_epi64(state->s[18], state->s[19]); //66 44 22 00
        t3 = _mm512_unpackhi_epi64(state->s[18], state->s[19]); //77 55 33 11

        z0 = _mm512_extracti32x8_epi32(t0,0);//22 00
        z1 = _mm512_extracti32x8_epi32(t2,0);//22 00
        z2 = _mm256_permute2x128_si256(z0,z1,0x20);// 00 00
        z3 = _mm256_permute2x128_si256(z0,z1,0x31);// 22 22
        _mm256_storeu_si256(out0 + 128,z2);
        _mm256_storeu_si256(out2 + 128,z3);

        z0 = _mm512_extracti32x8_epi32(t0,1);//66 44
        z1 = _mm512_extracti32x8_epi32(t2,1);//66 44
        z2 = _mm256_permute2x128_si256(z0,z1,0x20);// 44 44
        z3 = _mm256_permute2x128_si256(z0,z1,0x31);// 66 66
        _mm256_storeu_si256(out4 + 128,z2);
        _mm256_storeu_si256(out6 + 128,z3);

        z0 = _mm512_extracti32x8_epi32(t1,0);//33 11
        z1 = _mm512_extracti32x8_epi32(t3,0);//33 11
        z2 = _mm256_permute2x128_si256(z0,z1,0x20);// 11 11
        z3 = _mm256_permute2x128_si256(z0,z1,0x31);// 33 33
        _mm256_storeu_si256(out1 + 128,z2);
        _mm256_storeu_si256(out3 + 128,z3);

        z0 = _mm512_extracti32x8_epi32(t1,1);//77 55
        z1 = _mm512_extracti32x8_epi32(t3,1);//77 55
        z2 = _mm256_permute2x128_si256(z0,z1,0x20);// 55 55
        z3 = _mm256_permute2x128_si256(z0,z1,0x31);// 77 77
        _mm256_storeu_si256(out5 + 128,z2);
        _mm256_storeu_si256(out7 + 128,z3);


        temp = (uint8_t *) (&state->s[20]);
        memcpy(out0 + 160, temp, 8);
        memcpy(out1 + 160, temp + 8, 8);
        memcpy(out2 + 160, temp + 16, 8);
        memcpy(out3 + 160, temp + 24, 8);
        memcpy(out4 + 160, temp + 32, 8);
        memcpy(out5 + 160, temp + 40, 8);
        memcpy(out6 + 160, temp + 48, 8);
        memcpy(out7 + 160, temp + 56, 8);

        out0 += 168;
        out1 += 168;
        out2 += 168;
        out3 += 168;
        out4 += 168;
        out5 += 168;
        out6 += 168;
        out7 += 168;

    }
}



void XURQ_AVX512_shake256x8_squeezeblocks(keccakx8_state *state,
                              uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              uint8_t *out4,
                              uint8_t *out5,
                              uint8_t *out6,
                              uint8_t *out7,
                              int nblocks) {
    uint8_t *temp;
    __m512i t, t4, t5, t6, t7;
    __m512i t0, t1, t2, t3;
    for (int i = 0; i < nblocks; ++i) {
        XURQ_keccak8x_function(state->s);

        t0 = _mm512_unpacklo_epi64(state->s[0], state->s[1]); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(state->s[0], state->s[1]); //77 55 33 11
        t2 = _mm512_unpacklo_epi64(state->s[2], state->s[3]); //66 44 22 00
        t3 = _mm512_unpackhi_epi64(state->s[2], state->s[3]); //77 55 33 11
        t4 = _mm512_unpacklo_epi64(state->s[4], state->s[5]); //66 44 22 00
        t5 = _mm512_unpackhi_epi64(state->s[4], state->s[5]); //77 55 33 11
        t6 = _mm512_unpacklo_epi64(state->s[6], state->s[7]); //66 44 22 00
        t7 = _mm512_unpackhi_epi64(state->s[6], state->s[7]); //77 55 33 11

        t = _mm512_shuffle_i32x4(t0, t2, 0x44);//22 00 22 00
        t2 = _mm512_shuffle_i32x4(t0, t2, 0xee);//66 44 66 44
        t0 = t;
        t = _mm512_shuffle_i32x4(t4, t6, 0x44);//22 00 22 00
        t6 = _mm512_shuffle_i32x4(t4, t6, 0xee);//66 44 66 44
        t4 = t;
        t = _mm512_shuffle_i32x4(t1, t3, 0x44); //33 11 33 11
        t3 = _mm512_shuffle_i32x4(t1, t3, 0xee);//77 55 77 55
        t1 = t;
        t = _mm512_shuffle_i32x4(t5, t7, 0x44); //33 11 33 11
        t7 = _mm512_shuffle_i32x4(t5, t7, 0xee);//77 55 77 55
        t5 = t;
        _mm512_storeu_si512(out0, _mm512_shuffle_i32x4(t0, t4, 0x88));
        _mm512_storeu_si512(out1, _mm512_shuffle_i32x4(t1, t5, 0x88));
        _mm512_storeu_si512(out2, _mm512_shuffle_i32x4(t0, t4, 0xdd));
        _mm512_storeu_si512(out3, _mm512_shuffle_i32x4(t1, t5, 0xdd));
        _mm512_storeu_si512(out4, _mm512_shuffle_i32x4(t2, t6, 0x88));
        _mm512_storeu_si512(out5, _mm512_shuffle_i32x4(t3, t7, 0x88));
        _mm512_storeu_si512(out6, _mm512_shuffle_i32x4(t2, t6, 0xdd));
        _mm512_storeu_si512(out7, _mm512_shuffle_i32x4(t3, t7, 0xdd));

        t0 = _mm512_unpacklo_epi64(state->s[8], state->s[9]); //66 44 22 00
        t1 = _mm512_unpackhi_epi64(state->s[8], state->s[9]); //77 55 33 11
        t2 = _mm512_unpacklo_epi64(state->s[10], state->s[11]); //66 44 22 00
        t3 = _mm512_unpackhi_epi64(state->s[10], state->s[11]); //77 55 33 11
        t4 = _mm512_unpacklo_epi64(state->s[12], state->s[13]); //66 44 22 00
        t5 = _mm512_unpackhi_epi64(state->s[12], state->s[13]); //77 55 33 11
        t6 = _mm512_unpacklo_epi64(state->s[14], state->s[15]); //66 44 22 00
        t7 = _mm512_unpackhi_epi64(state->s[14], state->s[15]); //77 55 33 11

        t = _mm512_shuffle_i32x4(t0, t2, 0x44);//22 00 22 00
        t2 = _mm512_shuffle_i32x4(t0, t2, 0xee);//66 44 66 44
        t0 = t;
        t = _mm512_shuffle_i32x4(t4, t6, 0x44);//22 00 22 00
        t6 = _mm512_shuffle_i32x4(t4, t6, 0xee);//66 44 66 44
        t4 = t;
        t = _mm512_shuffle_i32x4(t1, t3, 0x44); //33 11 33 11
        t3 = _mm512_shuffle_i32x4(t1, t3, 0xee);//77 55 77 55
        t1 = t;
        t = _mm512_shuffle_i32x4(t5, t7, 0x44); //33 11 33 11
        t7 = _mm512_shuffle_i32x4(t5, t7, 0xee);//77 55 77 55
        t5 = t;
        _mm512_storeu_si512(out0 + 64, _mm512_shuffle_i32x4(t0, t4, 0x88));
        _mm512_storeu_si512(out1 + 64, _mm512_shuffle_i32x4(t1, t5, 0x88));
        _mm512_storeu_si512(out2 + 64, _mm512_shuffle_i32x4(t0, t4, 0xdd));
        _mm512_storeu_si512(out3 + 64, _mm512_shuffle_i32x4(t1, t5, 0xdd));
        _mm512_storeu_si512(out4 + 64, _mm512_shuffle_i32x4(t2, t6, 0x88));
        _mm512_storeu_si512(out5 + 64, _mm512_shuffle_i32x4(t3, t7, 0x88));
        _mm512_storeu_si512(out6 + 64, _mm512_shuffle_i32x4(t2, t6, 0xdd));
        _mm512_storeu_si512(out7 + 64, _mm512_shuffle_i32x4(t3, t7, 0xdd));


        temp = (uint8_t *) (&state->s[16]);
        memcpy(out0 + 128, temp, 8);
        memcpy(out1 + 128, temp + 8, 8);
        memcpy(out2 + 128, temp + 16, 8);
        memcpy(out3 + 128, temp + 24, 8);
        memcpy(out4 + 128, temp + 32, 8);
        memcpy(out5 + 128, temp + 40, 8);
        memcpy(out6 + 128, temp + 48, 8);
        memcpy(out7 + 128, temp + 56, 8);

        out0 += 136;
        out1 += 136;
        out2 += 136;
        out3 += 136;
        out4 += 136;
        out5 += 136;
        out6 += 136;
        out7 += 136;

    }
}


