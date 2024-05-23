//
// Created by xurq on 2023/3/8.
//

#include "ntt.h"
#include "consts.h"
#include "polyvec.h"
#include <x86intrin.h>


#define LOADAB(m) \
z0 = _mm512_load_epi32(a + (m));\
z1 = _mm512_load_epi32(a + (m) + 16);\
z2 = _mm512_load_epi32(a + (m) + 32);\
z3 = _mm512_load_epi32(a + (m) + 48);\
z4 = _mm512_load_epi32(a + (m) + 64);\
z5 = _mm512_load_epi32(a + (m) + 80);\
z6 = _mm512_load_epi32(a + (m) + 96);\
z7 = _mm512_load_epi32(a + (m) + 112);\
\
z8 =  _mm512_load_epi32(b + (m));\
z9 =  _mm512_load_epi32(b + (m) + 16);\
z10 = _mm512_load_epi32(b + (m) + 32);\
z11 = _mm512_load_epi32(b + (m) + 48);\
z12 = _mm512_load_epi32(b + (m) + 64);\
z13 = _mm512_load_epi32(b + (m) + 80);\
z14 = _mm512_load_epi32(b + (m) + 96);\
z15 = _mm512_load_epi32(b + (m) + 112);\
\


#define STOREX(c, m) \
_mm512_store_si512(c + (m),       x0);\
_mm512_store_si512(c + (m) + 16,  x1);\
_mm512_store_si512(c + (m) + 32,  x2);\
_mm512_store_si512(c + (m) + 48,  x3);\
_mm512_store_si512(c + (m) + 64,  x4);\
_mm512_store_si512(c + (m) + 80,  x5);\
_mm512_store_si512(c + (m) + 96,  x6);\
_mm512_store_si512(c + (m) + 112, x7);\
\



#define mont_mul(a, b, m) \
t0 = _mm512_mul_epi32(a, b); \
t1 = _mm512_mul_epi32(_mm512_srli_epi64(a,32),_mm512_srli_epi64(b,32)); \
r0 = _mm512_mul_epi32(t0, qinv) ;                                       \
r1 = _mm512_mul_epi32(t1, qinv) ;                                       \
r0 = _mm512_mul_epi32(r0, q);\
r1 = _mm512_mul_epi32(r1, q);\
r0 = _mm512_sub_epi32(t0, r0);\
r1 = _mm512_sub_epi32(t1, r1);\
x##m = _mm512_mask_blend_epi32(0xAAAA,_mm512_srli_epi64(r0,32),r1); \
\


void pointwise_avx512(int32_t c[N],
                      const int32_t a[N],
                      const int32_t b[N]) {
    __m512i z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15;
    __m512i t0, t1, r0, r1;
    __m512i x0, x1, x2, x3, x4, x5, x6, x7;

    const __m512i q = _mm512_set1_epi32(Q);
    const __m512i qinv = _mm512_set1_epi32(QINV);

    LOADAB(0)

    mont_mul(z0, z8, 0)
    mont_mul(z1, z9, 1)
    mont_mul(z2, z10, 2)
    mont_mul(z3, z11, 3)
    mont_mul(z4, z12, 4)
    mont_mul(z5, z13, 5)
    mont_mul(z6, z14, 6)
    mont_mul(z7, z15, 7)

    STOREX(c, 0)

    LOADAB(128)

    mont_mul(z0, z8, 0)
    mont_mul(z1, z9, 1)
    mont_mul(z2, z10, 2)
    mont_mul(z3, z11, 3)
    mont_mul(z4, z12, 4)
    mont_mul(z5, z13, 5)
    mont_mul(z6, z14, 6)
    mont_mul(z7, z15, 7)

    STOREX(c, 128)
}

#define LOADAB2(a, b, m) \
z0 = _mm512_load_epi32(a + (m));\
z1 = _mm512_load_epi32(a + (m) + 16);\
z2 = _mm512_load_epi32(a + (m) + 32);\
z3 = _mm512_load_epi32(a + (m) + 48);\
\
z4 =  _mm512_load_epi32(b + (m));\
z5 =  _mm512_load_epi32(b + (m) + 16);\
z6 = _mm512_load_epi32(b + (m) + 32);\
z7 = _mm512_load_epi32(b + (m) + 48);\
\



#define MUL(a, b, m) \
x##m = _mm512_mul_epi32(a,b);\
y##m = _mm512_mul_epi32(_mm512_srli_epi64(a, 32),_mm512_srli_epi64(b, 32)); \
\


#define reduce(a, b) \
r0 = _mm512_mul_epi32(a, qinv) ;                                       \
r1 = _mm512_mul_epi32(b, qinv) ;                                       \
r0 = _mm512_mul_epi32(r0, q);\
r1 = _mm512_mul_epi32(r1, q);\
r0 = _mm512_sub_epi32(a, r0);\
r1 = _mm512_sub_epi32(b, r1);\
a = _mm512_mask_blend_epi32(0xAAAA,_mm512_srli_epi64(r0,32),r1); \
\


void polyvecl_matrix_pointwise_mont(poly *w, const polyvecl *u, const polyvecl *v) {
    __m512i z0, z1, z2, z3, z4, z5, z6, z7;
    __m512i x0, x1, x2, x3, x4, x5, x6, x7;
    __m512i y0, y1, y2, y3, y4, y5, y6, y7;
    __m512i r0, r1;
    const __m512i q = _mm512_set1_epi32(Q);
    const __m512i qinv = _mm512_set1_epi32(QINV);

    LOADAB2(u->vec[0].coeffs, v->vec[0].coeffs, 0)
    MUL(z0, z4, 0)
    MUL(z1, z5, 1)
    MUL(z2, z6, 2)
    MUL(z3, z7, 3)


    for (int i = 1; i < L; ++i) {
        LOADAB2(u->vec[i].coeffs, v->vec[i].coeffs, 0)
        MUL(z0, z4, 4)
        MUL(z1, z5, 5)
        MUL(z2, z6, 6)
        MUL(z3, z7, 7)

        x0 = _mm512_add_epi64(x0,x4);
        x1 = _mm512_add_epi64(x1,x5);
        x2 = _mm512_add_epi64(x2,x6);
        x3 = _mm512_add_epi64(x3,x7);

        y0 = _mm512_add_epi64(y0,y4);
        y1 = _mm512_add_epi64(y1,y5);
        y2 = _mm512_add_epi64(y2,y6);
        y3 = _mm512_add_epi64(y3,y7);
    }

    reduce(x0, y0)
    reduce(x1, y1)
    reduce(x2, y2)
    reduce(x3, y3)

    STOREX(w->coeffs, 0)

    LOADAB2(u->vec[0].coeffs, v->vec[0].coeffs, 64)
    MUL(z0, z4, 0)
    MUL(z1, z5, 1)
    MUL(z2, z6, 2)
    MUL(z3, z7, 3)


    for (int i = 1; i < L; ++i) {
        LOADAB2(u->vec[i].coeffs, v->vec[i].coeffs, 64)
        MUL(z0, z4, 4)
        MUL(z1, z5, 5)
        MUL(z2, z6, 6)
        MUL(z3, z7, 7)

        x0 = _mm512_add_epi64(x0,x4);
        x1 = _mm512_add_epi64(x1,x5);
        x2 = _mm512_add_epi64(x2,x6);
        x3 = _mm512_add_epi64(x3,x7);

        y0 = _mm512_add_epi64(y0,y4);
        y1 = _mm512_add_epi64(y1,y5);
        y2 = _mm512_add_epi64(y2,y6);
        y3 = _mm512_add_epi64(y3,y7);
    }

    reduce(x0, y0)
    reduce(x1, y1)
    reduce(x2, y2)
    reduce(x3, y3)

    STOREX(w->coeffs, 64)

    LOADAB2(u->vec[0].coeffs, v->vec[0].coeffs, 128)
    MUL(z0, z4, 0)
    MUL(z1, z5, 1)
    MUL(z2, z6, 2)
    MUL(z3, z7, 3)


    for (int i = 1; i < L; ++i) {
        LOADAB2(u->vec[i].coeffs, v->vec[i].coeffs, 128)
        MUL(z0, z4, 4)
        MUL(z1, z5, 5)
        MUL(z2, z6, 6)
        MUL(z3, z7, 7)

        x0 = _mm512_add_epi64(x0,x4);
        x1 = _mm512_add_epi64(x1,x5);
        x2 = _mm512_add_epi64(x2,x6);
        x3 = _mm512_add_epi64(x3,x7);

        y0 = _mm512_add_epi64(y0,y4);
        y1 = _mm512_add_epi64(y1,y5);
        y2 = _mm512_add_epi64(y2,y6);
        y3 = _mm512_add_epi64(y3,y7);
    }

    reduce(x0, y0)
    reduce(x1, y1)
    reduce(x2, y2)
    reduce(x3, y3)

    STOREX(w->coeffs, 128)

    LOADAB2(u->vec[0].coeffs, v->vec[0].coeffs, 192)
    MUL(z0, z4, 0)
    MUL(z1, z5, 1)
    MUL(z2, z6, 2)
    MUL(z3, z7, 3)

    for (int i = 1; i < L; ++i) {
        LOADAB2(u->vec[i].coeffs, v->vec[i].coeffs, 192)
        MUL(z0, z4, 4)
        MUL(z1, z5, 5)
        MUL(z2, z6, 6)
        MUL(z3, z7, 7)

        x0 = _mm512_add_epi64(x0,x4);
        x1 = _mm512_add_epi64(x1,x5);
        x2 = _mm512_add_epi64(x2,x6);
        x3 = _mm512_add_epi64(x3,x7);

        y0 = _mm512_add_epi64(y0,y4);
        y1 = _mm512_add_epi64(y1,y5);
        y2 = _mm512_add_epi64(y2,y6);
        y3 = _mm512_add_epi64(y3,y7);
    }

    reduce(x0, y0)
    reduce(x1, y1)
    reduce(x2, y2)
    reduce(x3, y3)

    STOREX(w->coeffs, 192)
}