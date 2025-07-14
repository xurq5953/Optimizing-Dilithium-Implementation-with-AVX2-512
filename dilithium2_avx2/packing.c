#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"


/*************************************************
* Name:        unpack_sk
*
* Description: Unpack secret key sk = (rho, tr, key, t0, s1, s2).
*
* Arguments:   - const uint8_t rho[]: output byte array for rho
*              - const uint8_t tr[]: output byte array for tr
*              - const uint8_t key[]: output byte array for key
*              - const polyveck *t0: pointer to output vector t0
*              - const polyvecl *s1: pointer to output vector s1
*              - const polyveck *s2: pointer to output vector s2
*              - uint8_t sk[]: byte array containing bit-packed sk
**************************************************/
void unpack_sk(uint8_t rho[32],
               uint8_t tr[32],
               uint8_t key[32],
               polyveck *t0,
               polyvecl *s1,
               polyveck *s2,
               const uint8_t sk[2528]) {
    unsigned int i;

    for (i = 0; i < SEEDBYTES; ++i) {
        rho[i] = sk[i];
    }
    sk += SEEDBYTES;

    for (i = 0; i < SEEDBYTES; ++i) {
        key[i] = sk[i];
    }
    sk += SEEDBYTES;

    for (i = 0; i < SEEDBYTES; ++i) {
        tr[i] = sk[i];
    }
    sk += SEEDBYTES;

    for (i = 0; i < L; ++i) {
        polyeta_unpack_avx2(&s1->vec[i], sk + i * POLYETA_PACKEDBYTES);
    }
    sk += L * POLYETA_PACKEDBYTES;

    for (i = 0; i < K; ++i) {
        polyeta_unpack_avx2(&s2->vec[i], sk + i * POLYETA_PACKEDBYTES);
    }
    sk += K * POLYETA_PACKEDBYTES;

    for (i = 0; i < K; ++i) {
        polyt0_unpack_avx2(&t0->vec[i], sk + i * POLYT0_PACKEDBYTES);
    }
}





/*************************************************
* Name:        polyz_unpack_avx2
*
* Description: Unpack polynomial z with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void polyz_unpack_avx2(poly *restrict r, const uint8_t *a) {
    __m256i f0, f1, f2;
    const __m256i shufbidx = _mm256_set_epi8(-1, 9, 8, 7, -1, 7, 6, 5, -1, 5, 4, 3, -1, 3, 2, 1,
                                             -1, 8, 7, 6, -1, 6, 5, 4, -1, 4, 3, 2, -1, 2, 1, 0);
    const __m256i srlvdidx = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
    const __m256i mask = _mm256_set1_epi32(0x3FFFF);
    const __m256i gamma1 = _mm256_set1_epi32(GAMMA1);

    for (int i = 0; i < 30; i += 3) {
        f0 = _mm256_loadu_si256((__m256i *) &a[18 * i]);
        f1 = _mm256_loadu_si256((__m256i *) &a[18 * i + 18]);
        f2 = _mm256_loadu_si256((__m256i *) &a[18 * i + 36]);

        f0 = _mm256_permute4x64_epi64(f0, 0x94);
        f1 = _mm256_permute4x64_epi64(f1, 0x94);
        f2 = _mm256_permute4x64_epi64(f2, 0x94);

        f0 = _mm256_shuffle_epi8(f0, shufbidx);
        f1 = _mm256_shuffle_epi8(f1, shufbidx);
        f2 = _mm256_shuffle_epi8(f2, shufbidx);

        f0 = _mm256_srlv_epi32(f0, srlvdidx);
        f1 = _mm256_srlv_epi32(f1, srlvdidx);
        f2 = _mm256_srlv_epi32(f2, srlvdidx);

        f0 = _mm256_and_si256(f0, mask);
        f1 = _mm256_and_si256(f1, mask);
        f2 = _mm256_and_si256(f2, mask);

        f0 = _mm256_sub_epi32(gamma1, f0);
        f1 = _mm256_sub_epi32(gamma1, f1);
        f2 = _mm256_sub_epi32(gamma1, f2);

        _mm256_store_si256(&r->vec[i], f0);
        _mm256_store_si256(&r->vec[i + 1], f1);
        _mm256_store_si256(&r->vec[i + 2], f2);
    }

    f0 = _mm256_loadu_si256((__m256i *) &a[18 * 30]);
    f1 = _mm256_loadu_si256((__m256i *) &a[18 * 31]);

    f0 = _mm256_permute4x64_epi64(f0, 0x94);
    f1 = _mm256_permute4x64_epi64(f1, 0x94);

    f0 = _mm256_shuffle_epi8(f0, shufbidx);
    f1 = _mm256_shuffle_epi8(f1, shufbidx);

    f0 = _mm256_srlv_epi32(f0, srlvdidx);
    f1 = _mm256_srlv_epi32(f1, srlvdidx);

    f0 = _mm256_and_si256(f0, mask);
    f1 = _mm256_and_si256(f1, mask);

    f0 = _mm256_sub_epi32(gamma1, f0);
    f1 = _mm256_sub_epi32(gamma1, f1);

    _mm256_store_si256(&r->vec[30], f0);
    _mm256_store_si256(&r->vec[31], f1);
}



/*************************************************
* Name:        polyw1_pack_avx2
*
* Description: Bit-pack polynomial w1 with coefficients in [0,15] or [0,43].
*              Input coefficients are assumed to be positive standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYW1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyw1_pack_avx2(uint8_t *r, const poly *restrict a) {
    unsigned int i;
    __m256i f0, f1, f2, f3;
    const __m256i shift1 = _mm256_set1_epi16((64 << 8) + 1);
    const __m256i shift2 = _mm256_set1_epi32((4096 << 16) + 1);
    const __m256i shufdidx1 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    const __m256i shufdidx2 = _mm256_set_epi32(-1, -1, 6, 5, 4, 2, 1, 0);
    const __m256i shufbidx = _mm256_set_epi8(-1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0,
                                             -1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);

    for (i = 0; i < N / 32; i++) {
        f0 = _mm256_load_si256(&a->vec[4 * i + 0]);
        f1 = _mm256_load_si256(&a->vec[4 * i + 1]);
        f2 = _mm256_load_si256(&a->vec[4 * i + 2]);
        f3 = _mm256_load_si256(&a->vec[4 * i + 3]);
        f0 = _mm256_packus_epi32(f0, f1);
        f1 = _mm256_packus_epi32(f2, f3);
        f0 = _mm256_packus_epi16(f0, f1);
        f0 = _mm256_maddubs_epi16(f0, shift1);
        f0 = _mm256_madd_epi16(f0, shift2);
        f0 = _mm256_permutevar8x32_epi32(f0, shufdidx1);
        f0 = _mm256_shuffle_epi8(f0, shufbidx);
        f0 = _mm256_permutevar8x32_epi32(f0, shufdidx2);
        _mm256_storeu_si256((__m256i *) &r[24 * i], f0);
    }

}

/*************************************************
* Name:        polyz_pack_avx2
 * need 7 bytes redundancy
*
* Description: Bit-pack polynomial with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYZ_PACKEDBYTES+7 bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyz_pack_avx2(uint8_t r[POLYZ_PACKEDBYTES+7], const poly *restrict a) {
    __m256i f0,f1,f2,f3;
    __m256i t0,t1,t2,t3;
    __m128i g0,g1;
    const __m256i gamma1 = _mm256_set1_epi32(GAMMA1);
    const __m256i mask0 = _mm256_set1_epi64x(0xffffffff00000000UL);
    const __m256i idx0 = _mm256_setr_epi64x(0,28,0,28);

    for (int i = 0; i < 8; ++i) {
        //concatenate32(r,18)
        f0 = _mm256_load_si256(&a->vec[i * 4 + 0]);
        f1 = _mm256_load_si256(&a->vec[i * 4 + 1]);
        f2 = _mm256_load_si256(&a->vec[i * 4 + 2]);
        f3 = _mm256_load_si256(&a->vec[i * 4 + 3]);
        f0 = _mm256_sub_epi32(gamma1, f0);
        f1 = _mm256_sub_epi32(gamma1, f1);
        f2 = _mm256_sub_epi32(gamma1, f2);
        f3 = _mm256_sub_epi32(gamma1, f3);
        t0 = _mm256_srli_epi64(f0 & mask0, 14);
        t1 = _mm256_srli_epi64(f1 & mask0, 14);
        t2 = _mm256_srli_epi64(f2 & mask0, 14);
        t3 = _mm256_srli_epi64(f3 & mask0, 14);
        f0 = _mm256_andnot_si256(mask0,f0) ^ t0;
        f1 = _mm256_andnot_si256(mask0,f1) ^ t1;
        f2 = _mm256_andnot_si256(mask0,f2) ^ t2;
        f3 = _mm256_andnot_si256(mask0,f3) ^ t3;

        //concatenate64(r,36)
        t0 = _mm256_srli_si256(f0,4);
        t1 = _mm256_srli_si256(f1,4);
        t2 = _mm256_srli_si256(f2,4);
        t3 = _mm256_srli_si256(f3,4);
        t0 = t0 & mask0;
        t1 = t1 & mask0;
        t2 = t2 & mask0;
        t3 = t3 & mask0;
        t0 = _mm256_slli_epi32(t0, 4);
        t1 = _mm256_slli_epi32(t1, 4);
        t2 = _mm256_slli_epi32(t2, 4);
        t3 = _mm256_slli_epi32(t3, 4);
        f0 = f0 ^ t0;
        f1 = f1 ^ t1;
        f2 = f2 ^ t2;
        f3 = f3 ^ t3;
        f0 = _mm256_srlv_epi64(f0,idx0);
        f1 = _mm256_srlv_epi64(f1,idx0);
        f2 = _mm256_srlv_epi64(f2,idx0);
        f3 = _mm256_srlv_epi64(f3,idx0);


        g0 = _mm256_castsi256_si128(f1);
        g1 = _mm256_extractf128_si256(f1,1);
        _mm_storeu_si128((__m128i_u *) (r + i * 72 + 18), g0);
        _mm_storeu_si128((__m128i_u *) (r + i * 72 + 27), g1);

        g0 = _mm256_castsi256_si128(f2);
        g1 = _mm256_extractf128_si256(f2,1);
        _mm_storeu_si128((__m128i_u *) (r + i * 72 + 36), g0);
        _mm_storeu_si128((__m128i_u *) (r + i * 72 + 45), g1);

        g0 = _mm256_castsi256_si128(f3);
        g1 = _mm256_extractf128_si256(f3,1);
        _mm_storeu_si128((__m128i_u *) (r + i * 72 + 54), g0);
        _mm_storeu_si128((__m128i_u *) (r + i * 72 + 63), g1);
    }

}


/*************************************************
* Name:        polyt0_pack_avx2  r need 3 bytes redundancy
*
* Description: Bit-pack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT0_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyt0_pack_avx2(uint8_t r[416], const poly *restrict a) {
    __m256i f0,f1,f2,f3;
    __m256i t0,t1,t2,t3;
    __m128i g0,g1;
    const __m256i d = _mm256_set1_epi32(1 << (D - 1));
    const __m256i mask0 = _mm256_set1_epi64x(0xffffffff00000000UL);
    const __m256i idx0 = _mm256_setr_epi64x(0,12,0,12);

    for (int i = 0; i < 32; ++i) {
        //concatenate32(r,13)
        f0 = _mm256_load_si256(&a->vec[i]);
        f0 = _mm256_sub_epi32(d, f0);
        t0 = _mm256_srli_epi64(f0, 19);
        f0 = t0 ^ f0;
        f0 = _mm256_andnot_si256(mask0, f0);

        //concatenate64(r, 26)
        t0 = _mm256_srli_si256(f0, 4);
        f0 = f0 ^ t0;
        t0 = f0 & mask0;
        t0 = _mm256_srli_epi64(t0, 6);
        f0 = _mm256_andnot_si256(mask0, f0);
        f0 = f0 ^ t0;
        f0 = _mm256_permute4x64_epi64(f0,0xd8);

        //concatenate64(r,52)
        t0 = _mm256_srli_si256(f0,4);
        t0 = t0 & mask0;
        t0 = _mm256_slli_epi32(t0, 20);
        f0 = f0 ^ t0;
        f0 = _mm256_srlv_epi64(f0,idx0);

        g0 = _mm256_castsi256_si128(f0);
        _mm_storeu_si128((__m128i_u *) (r + i * 13), g0);
    }

}

void polyt1_unpack_avx2(poly *restrict r, const uint8_t a[320]) {
    __m256i b, b0;
    __m256i mask0 = _mm256_set_epi32(4, 4, 3, 2, 3, 2, 1, 0);
    __m256i mask1 = _mm256_set_epi8(11, 10, 10, 9, 9, 8, 8, 7, 6, 5, 5, 4, 4, 3, 3, 2,9, 8, 8, 7, 7, 6, 6, 5, 4, 3, 3, 2, 2, 1, 1, 0);
    __m256i mask2 = _mm256_set1_epi32(0x3ff);
    __m256i index = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);

    for (int i = 0; i < 16; ++i) {
        b = _mm256_loadu_si256((__m256i *) (a + 20 * i));
        b = _mm256_permutevar8x32_epi32(b, mask0);
        b = _mm256_shuffle_epi8(b, mask1);
        b0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b));
        b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b, 1));
        b0 = _mm256_srlv_epi32(b0, index);
        b = _mm256_srlv_epi32(b, index);
        b0 &= mask2;
        b &= mask2;
        _mm256_store_si256(&r->vec[i * 2], b0);
        _mm256_store_si256(&r->vec[i * 2 + 1], b);
    }

}


/*************************************************
* Name:        polyt1_pack_avx2
*
* Description: Bit-pack polynomial t1 with coefficients fitting in 10 bits.
*              Input coefficients are assumed to be positive standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyt1_pack_avx2(uint8_t r[320], const poly *restrict a) {
    __m256i f0,f1,f2,f3;
    __m256i t0,t1,t2,t3;
    __m128i g0,g1;
    const __m256i mask0 = _mm256_set1_epi64x(0xffffffff00000000UL);
    const __m256i idx0 = _mm256_setr_epi64x(0,20,0,20);

    for (int i = 0; i < 32; ++i) {
        //concatenate32(r,10)
        f0 = _mm256_load_si256(&a->vec[i]);
        t0 = _mm256_srli_epi64(f0, 22);
        f0 = t0 ^ f0;
        f0 = _mm256_andnot_si256(mask0,f0);

        //concatenate64(r, 20)
        t0 = _mm256_srli_si256(f0, 4);
        t0 = _mm256_srli_epi64(t0, 12);
        f0 = f0 ^ t0;
        f0 = _mm256_srlv_epi64(f0,idx0);

        g0 = _mm256_castsi256_si128(f0);
        g1 = _mm256_extractf128_si256(f0,1);
        _mm_storeu_si128((__m128i_u *) (r + i * 10), g0);
        _mm_storeu_si128((__m128i_u *) (r + i * 10 + 5), g1);
    }
}


/*************************************************
* Name:        polyeta_unpack_avx2
*
* Description: Unpack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void polyeta_unpack_avx2(poly *restrict r, const uint8_t a[POLYETA_PACKEDBYTES]) {
    __m256i f0,f1,f2,f3,f4,f5,f6;

    const __m256i mask0 = _mm256_set1_epi32(0x7);
    const __m256i idx0 = _mm256_setr_epi32(0,3,6,9,12,15,18,21);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i eta = _mm256_set1_epi32(ETA);

    for (int i = 0; i < 6; ++i) {
        f0 = _mm256_loadu_si256((__m256i *) (a + 15 * i));

        f1 = _mm256_permutevar8x32_epi32(f0,zero);
        f1 = _mm256_srlv_epi32(f1,idx0);
        f1 = f1 & mask0;
        f1 = _mm256_sub_epi32(eta,f1);
        _mm256_store_si256(&r->vec[i * 5],f1);

        f2 = _mm256_srli_si256(f0,3);
        f2 = _mm256_permutevar8x32_epi32(f2,zero);
        f2 = _mm256_srlv_epi32(f2,idx0);
        f2 = f2 & mask0;
        f2 = _mm256_sub_epi32(eta,f2);
        _mm256_store_si256(&r->vec[i * 5 + 1],f2);

        f3 = _mm256_srli_si256(f0,6);
        f3 = _mm256_permutevar8x32_epi32(f3,zero);
        f3 = _mm256_srlv_epi32(f3,idx0);
        f3 = f3 & mask0;
        f3 = _mm256_sub_epi32(eta,f3);
        _mm256_store_si256(&r->vec[i * 5 + 2],f3);

        f4 = _mm256_srli_si256(f0,9);
        f4 = _mm256_permutevar8x32_epi32(f4,zero);
        f4 = _mm256_srlv_epi32(f4,idx0);
        f4 = f4 & mask0;
        f4 = _mm256_sub_epi32(eta,f4);
        _mm256_store_si256(&r->vec[i * 5 + 3],f4);

        f5 = _mm256_srli_si256(f0,12);
        f5 = _mm256_permutevar8x32_epi32(f5,zero);
        f5 = _mm256_srlv_epi32(f5,idx0);
        f5 = f5 & mask0;
        f5 = _mm256_sub_epi32(eta,f5);
        _mm256_store_si256(&r->vec[i * 5 + 4],f5);
    }

    f0 = _mm256_loadu_si256((__m256i *) (a + 90));

    f1 = _mm256_permutevar8x32_epi32(f0,zero);
    f1 = _mm256_srlv_epi32(f1,idx0);
    f1 = f1 & mask0;
    f1 = _mm256_sub_epi32(eta,f1);
    _mm256_store_si256(&r->vec[30],f1);

    f2 = _mm256_srli_si256(f0,3);
    f2 = _mm256_permutevar8x32_epi32(f2,zero);
    f2 = _mm256_srlv_epi32(f2,idx0);
    f2 = f2 & mask0;
    f2 = _mm256_sub_epi32(eta,f2);
    _mm256_store_si256(&r->vec[31],f2);

}


/*************************************************
* Name:        polyt0_unpack_avx2
*
* Description: Unpack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void polyt0_unpack_avx2(poly *restrict r, const uint8_t a[POLYT0_PACKEDBYTES]) {
    __m256i f0,f1,f2,f3,f4,f5,f6;

    const __m256i mask0 = _mm256_set1_epi32(0x1fff);
    const __m256i idx0 = _mm256_setr_epi8(0,1,2,3,0,1,2,3,2,3,4,5,4,5,6,7,
                                          6,7,8,9,8,9,8,9,8,9,10,11,10,11,12,13);
    const __m256i idx1 = _mm256_setr_epi32(0,13,10,7,4,1,14,11);
    const __m256i d = _mm256_set1_epi32(1 << (D - 1));

    for (int i = 0; i < N / 8; ++i) {
        f0 = _mm256_loadu_si256((__m256i *) (a + 13 * i));

        f1 = _mm256_permute4x64_epi64(f0, 0x44);
        f1 = _mm256_shuffle_epi8(f1,idx0);
        f1 = _mm256_srlv_epi32(f1,idx1);
        f1 = f1 & mask0;
        f1 = _mm256_sub_epi32(d, f1);

        _mm256_store_si256(&r->vec[i],f1);
    }

}

