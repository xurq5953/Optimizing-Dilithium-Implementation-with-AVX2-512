#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"



void XURQ_unpack_sk(uint8_t rho[SEEDBYTES],
                    uint8_t tr[SEEDBYTES],
                    uint8_t key[SEEDBYTES],
                    polyveck *t0,
                    polyvecl *s1,
                    polyveck *s2,
                    const uint8_t sk[DILITHIUM5_AVX2_CRYPTO_SECRETKEYBYTES]) {
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
        XURQ_AVX512_polyeta_unpack(&s1->vec[i], sk + i * POLYETA_PACKEDBYTES);
    }
    sk += L * POLYETA_PACKEDBYTES;

    for (i = 0; i < K; ++i) {
        XURQ_AVX512_polyeta_unpack(&s2->vec[i], sk + i * POLYETA_PACKEDBYTES);
    }
    sk += K * POLYETA_PACKEDBYTES;

    for (i = 0; i < K; ++i) {
        XURQ_AVX512_polyt0_unpack(&t0->vec[i], sk + i * POLYT0_PACKEDBYTES);
    }
}



ALIGN(64) static const uint8_t idx_eta_up[64] = {
        0, 0xff, 0, 0xff, 0, 1, 1, 0xff, 1, 0xff, 1, 2, 2, 0xff, 2, 0xff,
        3, 0xff, 3, 0xff, 3, 4, 4, 0xff, 4, 0xff, 4, 5, 5, 0xff, 5, 0xff,
        6, 0xff, 6, 0xff, 6, 7, 7, 0xff, 7, 0xff, 7, 8, 8, 0xff, 8, 0xff,
        9, 0xff, 9, 0xff, 9, 10, 10, 0xff, 10, 0xff, 10, 11, 11, 0xff, 11, 0xff,
};

ALIGN(64) static const uint16_t idx_eta_up1[32] = {
        0, 3, 6, 1, 4, 7, 2, 5,
        0, 3, 6, 1, 4, 7, 2, 5,
        0, 3, 6, 1, 4, 7, 2, 5,
        0, 3, 6, 1, 4, 7, 2, 5,
};

void XURQ_AVX512_polyeta_unpack(poly *restrict r, const uint8_t a[POLYETA_PACKEDBYTES]) {
    __m512i b0, b1, b2, b3;
    int pos = 0;
    int ctr = 0;
    const __m512i idx0 = _mm512_load_si512(idx_eta_up);
    const __m512i idx1 = _mm512_load_si512(idx_eta_up1);
    const __m512i mask = _mm512_set1_epi16(7);
    const __m512i eta = _mm512_set1_epi16(2);

    b0 = _mm512_loadu_si512(a + pos);
    b1 = _mm512_loadu_si512(a + pos + 12);
    b2 = _mm512_loadu_si512(a + pos + 24);
    b3 = _mm512_loadu_si512(a + pos + 36);

    b0 = _mm512_permutexvar_epi8(idx0, b0);
    b1 = _mm512_permutexvar_epi8(idx0, b1);
    b2 = _mm512_permutexvar_epi8(idx0, b2);
    b3 = _mm512_permutexvar_epi8(idx0, b3);

    b0 = _mm512_srlv_epi16(b0, idx1);
    b1 = _mm512_srlv_epi16(b1, idx1);
    b2 = _mm512_srlv_epi16(b2, idx1);
    b3 = _mm512_srlv_epi16(b3, idx1);

    b0 &= mask;
    b1 &= mask;
    b2 &= mask;
    b3 &= mask;

    b0 = _mm512_sub_epi16(eta, b0);
    b1 = _mm512_sub_epi16(eta, b1);
    b2 = _mm512_sub_epi16(eta, b2);
    b3 = _mm512_sub_epi16(eta, b3);

    _mm512_storeu_epi32(r->coeffs + ctr, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b0, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 16, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b0, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 32, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b1, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 48, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b1, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 64, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b2, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 80, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b2, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 96, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b3, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 112, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b3, 1)));

    pos += 48;
    ctr += 128;

    b0 = _mm512_loadu_si512(a + pos);
    b1 = _mm512_loadu_si512(a + pos + 12);
    b2 = _mm512_loadu_si512(a + pos + 24);
    b3 = _mm512_loadu_si512(a + pos + 36);

    b0 = _mm512_permutexvar_epi8(idx0, b0);
    b1 = _mm512_permutexvar_epi8(idx0, b1);
    b2 = _mm512_permutexvar_epi8(idx0, b2);
    b3 = _mm512_permutexvar_epi8(idx0, b3);

    b0 = _mm512_srlv_epi16(b0, idx1);
    b1 = _mm512_srlv_epi16(b1, idx1);
    b2 = _mm512_srlv_epi16(b2, idx1);
    b3 = _mm512_srlv_epi16(b3, idx1);

    b0 &= mask;
    b1 &= mask;
    b2 &= mask;
    b3 &= mask;

    b0 = _mm512_sub_epi16(eta, b0);
    b1 = _mm512_sub_epi16(eta, b1);
    b2 = _mm512_sub_epi16(eta, b2);
    b3 = _mm512_sub_epi16(eta, b3);

    _mm512_storeu_epi32(r->coeffs + ctr, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b0, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 16, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b0, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 32, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b1, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 48, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b1, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 64, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b2, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 80, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b2, 1)));
    _mm512_storeu_epi32(r->coeffs + ctr + 96, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b3, 0)));
    _mm512_storeu_epi32(r->coeffs + ctr + 112, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(b3, 1)));
}



ALIGN(64) static const uint8_t idx_t0_u[64] = {
        0, 1, 0xff, 0xff,
        1, 2, 3, 0xff,
        3, 4, 0xff, 0xff,
        4, 5, 6, 0xff,
        6, 7, 8, 0xff,
        8, 9, 0xff, 0xff,
        9, 10, 11, 0xff,
        11, 12, 0xff, 0xff,

        13, 14, 0xff, 0xff,
        14, 15, 16, 0xff,
        16, 17, 0xff, 0xff,
        17, 18, 19, 0xff,
        19, 20, 21, 0xff,
        21, 22, 0xff, 0xff,
        22, 23, 24, 0xff,
        24, 25, 0xff, 0xff
};


void XURQ_AVX512_polyt0_unpack(poly *restrict r, const uint8_t a[POLYT0_PACKEDBYTES]) {
    int pos = 0;
    int ctr = 0;
    __m512i f0, f1, f2, f3;
    const __m512i idx0 = _mm512_load_si512(idx_t0_u);
    const __m512i idx1 = _mm512_setr_epi32(0, 5, 2, 7,
                                           4, 1, 6, 3,
                                           0, 5, 2, 7,
                                           4, 1, 6, 3);
    const __m512i mask = _mm512_set1_epi32(0x1FFF);
    const __m512i n = _mm512_set1_epi32((1 << (D - 1)));


    for (int i = 0; i < 4; ++i) {
        f0 = _mm512_load_si512(a + pos);
        f1 = _mm512_load_si512(a + pos + 26);
        f2 = _mm512_load_si512(a + pos + 52);
        f3 = _mm512_load_si512(a + pos + 78);

        f0 = _mm512_permutexvar_epi8(idx0, f0);
        f1 = _mm512_permutexvar_epi8(idx0, f1);
        f2 = _mm512_permutexvar_epi8(idx0, f2);
        f3 = _mm512_permutexvar_epi8(idx0, f3);

        f0 = _mm512_srlv_epi32(f0, idx1);
        f1 = _mm512_srlv_epi32(f1, idx1);
        f2 = _mm512_srlv_epi32(f2, idx1);
        f3 = _mm512_srlv_epi32(f3, idx1);

        f0 &= mask;
        f1 &= mask;
        f2 &= mask;
        f3 &= mask;

        f0 = _mm512_sub_epi32(n, f0);
        f1 = _mm512_sub_epi32(n, f1);
        f2 = _mm512_sub_epi32(n, f2);
        f3 = _mm512_sub_epi32(n, f3);

        _mm512_storeu_epi32(r->coeffs + ctr, f0);
        _mm512_storeu_epi32(r->coeffs + ctr + 16, f1);
        _mm512_storeu_epi32(r->coeffs + ctr + 32, f2);
        _mm512_storeu_epi32(r->coeffs + ctr + 48, f3);

        pos += 104;
        ctr += 64;
    }

}