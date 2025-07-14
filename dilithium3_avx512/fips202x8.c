//
// Created by xurq on 2022/10/19.
//

#include <string.h>
#include <stdio.h>
#include "fips202x8.h"
#include "common/keccak/KeccakP-1600-times8-SnP.h"


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
void shake128x8_absorb(keccakx8_state *state,
                       const uint8_t *in0,
                       const uint8_t *in1,
                       const uint8_t *in2,
                       const uint8_t *in3,
                       const uint8_t *in4,
                       const uint8_t *in5,
                       const uint8_t *in6,
                       const uint8_t *in7,
                       int inlen) {
    __m512i idx = _mm512_set_epi64(in7, in6, in5, in4, in3, in2, in1, in0);
    int tail = inlen & 7;
    int lanes = inlen >> 3;

    for (int i = 0; i < lanes; ++i)
        state->s[i] = _mm512_i64gather_epi64(idx, i * 8, 1);

    if (tail) {
        state->s[lanes] = _mm512_i64gather_epi64(idx, lanes * 8, 1) & _mm512_set1_epi64((1ULL << (tail * 8)) - 1);
    }
    lanes++;

    for (int i = lanes; i < 25; ++i)
        state->s[i] = _mm512_setzero_si512();

    state->s[lanes - 1] ^= _mm512_set1_epi64(0x1fULL << (tail * 8));
    state->s[20] ^= _mm512_set1_epi64(1ULL << 63);
}

void shake256x8_absorb(keccakx8_state *state,
                       const uint8_t *in0,
                       const uint8_t *in1,
                       const uint8_t *in2,
                       const uint8_t *in3,
                       const uint8_t *in4,
                       const uint8_t *in5,
                       const uint8_t *in6,
                       const uint8_t *in7,
                       int inlen) {
    __m512i idx = _mm512_set_epi64(in7, in6, in5, in4, in3, in2, in1, in0);
    int tail = inlen & 7;
    int lanes = inlen >> 3;

    for (int i = 0; i < lanes; ++i)
        state->s[i] = _mm512_i64gather_epi64(idx, i * 8, 1);

    if (tail) {
        state->s[lanes] = _mm512_i64gather_epi64(idx, lanes * 8, 1) & _mm512_set1_epi64((1ULL << (tail * 8)) - 1);
    }
    lanes++;

    for (int i = lanes; i < 25; ++i)
        state->s[i] = _mm512_setzero_si512();

    state->s[lanes - 1] ^= _mm512_set1_epi64(0x1fULL << (tail * 8));
    state->s[16] ^= _mm512_set1_epi64(1ULL << 63);
}


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



#define HTOLE_64(i) (i)
#define LETOH_64(i) (i)
#define VXOR(X, Y)            _mm512_xor_si512(X, Y)

static uint64_t load64(const uint8_t *x)
{
    return LETOH_64(*((uint64_t*)x));
}

static void store64(uint8_t *x, uint64_t u)
{
    *(uint64_t*)x = HTOLE_64(u);
}


static __m512i set_vector(const uint64_t a7, const uint64_t a6, const uint64_t a5, const uint64_t a4,
                          const uint64_t a3, const uint64_t a2, const uint64_t a1, const uint64_t a0)
{
    __m512i r;

    ((uint64_t *)&r)[0] = a0; ((uint64_t *)&r)[1] = a1;
    ((uint64_t *)&r)[2] = a2; ((uint64_t *)&r)[3] = a3;
    ((uint64_t *)&r)[4] = a4; ((uint64_t *)&r)[5] = a5;
    ((uint64_t *)&r)[6] = a6; ((uint64_t *)&r)[7] = a7;

    return r;
}

void SIKE_keccak_absorb_8x1w(__m512i *s, unsigned int r, const uint8_t *m0, const uint8_t *m1,
                             const uint8_t *m2, const uint8_t *m3, const uint8_t *m4, const uint8_t *m5,
                             const uint8_t *m6, const uint8_t *m7, unsigned long long int mlen, unsigned char p)
{
    unsigned long long i;
    unsigned char t0[200], t1[200], t2[200], t3[200], t4[200], t5[200], t6[200], t7[200];
    __m512i a;

    while (mlen >= r)
    {
        for (i = 0; i < r/8; i++) {
            a = set_vector(load64(m7+8*i), load64(m6+8*i), load64(m5+8*i), load64(m4+8*i), \
                     load64(m3+8*i), load64(m2+8*i), load64(m1+8*i), load64(m0+8*i));
            s[i] = VXOR(s[i], a);
        }

        KeccakP1600times8_PermuteAll_24rounds(s);
        mlen -= r;
        m0 += r; m1 += r; m2 += r; m3 += r;
        m4 += r; m5 += r; m6 += r; m7 += r;
    }

    for (i = 0; i < r; i++) {
        t0[i] = t1[i] = t2[i] = t3[i] = 0;
        t4[i] = t5[i] = t6[i] = t7[i] = 0;
    }

    for (i = 0; i < mlen; i++) {
        t0[i] = m0[i]; t1[i] = m1[i]; t2[i] = m2[i]; t3[i] = m3[i];
        t4[i] = m4[i]; t5[i] = m5[i]; t6[i] = m6[i]; t7[i] = m7[i];
    }

    t0[i] = t1[i] = t2[i] = t3[i] = p;
    t4[i] = t5[i] = t6[i] = t7[i] = p;

    t0[r-1] |= 128; t1[r-1] |= 128; t2[r-1] |= 128; t3[r-1] |= 128;
    t4[r-1] |= 128; t5[r-1] |= 128; t6[r-1] |= 128; t7[r-1] |= 128;

    for (i = 0; i < r/8; i++) {
        a = set_vector(load64(t7+8*i), load64(t6+8*i), load64(t5+8*i), load64(t4+8*i), \
                   load64(t3+8*i), load64(t2+8*i), load64(t1+8*i), load64(t0+8*i));
        s[i] = VXOR(s[i], a);
    }
}

void SIKE_keccak_squeezeblocks_8x1w(uint8_t *h0, uint8_t *h1, uint8_t *h2, uint8_t *h3,
                                    uint8_t *h4, uint8_t *h5, uint8_t *h6, uint8_t *h7,
                                    unsigned long long int nblocks, __m512i *s, unsigned int r)
{
    unsigned int i;

    while (nblocks > 0) {
        KeccakP1600times8_PermuteAll_24rounds(s);
        for (i = 0; i < (r>>3); i++) {
            store64(h0+8*i, ((uint64_t *)&s[i])[0]);
            store64(h1+8*i, ((uint64_t *)&s[i])[1]);
            store64(h2+8*i, ((uint64_t *)&s[i])[2]);
            store64(h3+8*i, ((uint64_t *)&s[i])[3]);
            store64(h4+8*i, ((uint64_t *)&s[i])[4]);
            store64(h5+8*i, ((uint64_t *)&s[i])[5]);
            store64(h6+8*i, ((uint64_t *)&s[i])[6]);
            store64(h7+8*i, ((uint64_t *)&s[i])[7]);
        }
        h0 += r; h1 += r; h2 += r; h3 += r;
        h4 += r; h5 += r; h6 += r; h7 += r;
        nblocks--;
    }
}

#define VZERO                 _mm512_setzero_si512()

void SIKE_shake256_8x1w(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3,
                        uint8_t *out4, uint8_t *out5, uint8_t *out6, uint8_t *out7,
                        unsigned long long outlen,
                        const uint8_t *in0, const uint8_t *in1, const uint8_t *in2, const uint8_t *in3,
                        const uint8_t *in4, const uint8_t *in5, const uint8_t *in6, const uint8_t *in7,
                        unsigned long long inlen)
{
    __m512i s[25];
    uint8_t t0[SHAKE256_RATE], t1[SHAKE256_RATE], t2[SHAKE256_RATE], t3[SHAKE256_RATE];
    uint8_t t4[SHAKE256_RATE], t5[SHAKE256_RATE], t6[SHAKE256_RATE], t7[SHAKE256_RATE];
    unsigned long long nblocks = outlen/SHAKE256_RATE;
    unsigned int i;

    for (i = 0; i < 25; i++) s[i] = VZERO;

    /* Absorb input */
    SIKE_keccak_absorb_8x1w(s, SHAKE256_RATE, in0, in1, in2, in3, in4, in5, in6, in7, inlen, 0x1F);

    /* Squeeze output */
    SIKE_keccak_squeezeblocks_8x1w(out0, out1, out2, out3, out4, out5, out6, out7, nblocks, s, SHAKE256_RATE);

    out0 += nblocks*SHAKE256_RATE;  out1 += nblocks*SHAKE256_RATE;
    out2 += nblocks*SHAKE256_RATE;  out3 += nblocks*SHAKE256_RATE;
    out4 += nblocks*SHAKE256_RATE;  out5 += nblocks*SHAKE256_RATE;
    out6 += nblocks*SHAKE256_RATE;  out7 += nblocks*SHAKE256_RATE;
    outlen -= nblocks*SHAKE256_RATE;

    if (outlen) {
        SIKE_keccak_squeezeblocks_8x1w(t0, t1, t2, t3, t4, t5, t6, t7, 1, s, SHAKE256_RATE);

        for (i = 0; i < outlen; i++) {
            out0[i] = t0[i]; out1[i] = t1[i];
            out2[i] = t2[i]; out3[i] = t3[i];
            out4[i] = t4[i]; out5[i] = t5[i];
            out6[i] = t6[i]; out7[i] = t7[i];
        }
    }
}
