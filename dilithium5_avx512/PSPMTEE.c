//
// Created by xurq on 2024/1/12.
//

#include "PSPMTEE.h"
#include <x86intrin.h>

int pspm(poly *c, poly *s, poly *y) {
    ALIGN(64) uint8_t stable[512];
    __m512i w0, w1, w2, w3;
    __m512i f0, f1, f2, f3, f4, f5, f6, f7;
    __m512i t;
    __m128i t0;
    const __m128i zero = _mm_setzero_si128();
    uint16_t g;

    //prepare table
    for (int j = 0; j < 16; ++j) {
        t = s->vec2[j];
        t0 = _mm512_cvtepi32_epi8(t);
        _mm_storeu_epi8(stable + 256 + 16 * j, t0);
        _mm_storeu_epi8(stable + 16 * j, _mm_sub_epi8(zero,t0));
    }

    w0 = _mm512_setzero_epi32();
    w1 = _mm512_setzero_epi32();
    w2 = _mm512_setzero_epi32();
    w3 = _mm512_setzero_epi32();

    //compute cs1[i]
    for (int j = 0; j < N; ++j) {
        if (c->coeffs[j] == 1) {
            f0 = _mm512_loadu_epi8(stable + N - j);
            f1 = _mm512_loadu_epi8(stable + 64  + N - j);
            f2 = _mm512_loadu_epi8(stable + 128 + N - j);
            f3 = _mm512_loadu_epi8(stable + 192 + N - j);

            w0 = _mm512_add_epi8(w0, f0);
            w1 = _mm512_add_epi8(w1, f1);
            w2 = _mm512_add_epi8(w2, f2);
            w3 = _mm512_add_epi8(w3, f3);
        } else if (c->coeffs[j] == -1) {
            f0 = _mm512_loadu_epi8(stable + N - j);
            f1 = _mm512_loadu_epi8(stable + 64  + N - j);
            f2 = _mm512_loadu_epi8(stable + 128 + N - j);
            f3 = _mm512_loadu_epi8(stable + 192 + N - j);

            w0 = _mm512_sub_epi8(w0, f0);
            w1 = _mm512_sub_epi8(w1, f1);
            w2 = _mm512_sub_epi8(w2, f2);
            w3 = _mm512_sub_epi8(w3, f3);
        }
    }


    //recover
    f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 0));
    f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 1));
    f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 2));
    f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 3));


    _mm512_store_epi32(&y->vec2[0],f4);
    _mm512_store_epi32(&y->vec2[1],f5);
    _mm512_store_epi32(&y->vec2[2],f6);
    _mm512_store_epi32(&y->vec2[3],f7);

    f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 0));
    f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 1));
    f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 2));
    f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 3));


    _mm512_store_epi32(&y->vec2[4],f4);
    _mm512_store_epi32(&y->vec2[5],f5);
    _mm512_store_epi32(&y->vec2[6],f6);
    _mm512_store_epi32(&y->vec2[7],f7);

    f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 0));
    f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 1));
    f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 2));
    f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 3));


    _mm512_store_epi32(&y->vec2[8 ],f4);
    _mm512_store_epi32(&y->vec2[9 ],f5);
    _mm512_store_epi32(&y->vec2[10],f6);
    _mm512_store_epi32(&y->vec2[11],f7);

    f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 0));
    f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 1));
    f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 2));
    f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 3));


    _mm512_store_epi32(&y->vec2[12],f4);
    _mm512_store_epi32(&y->vec2[13],f5);
    _mm512_store_epi32(&y->vec2[14],f6);
    _mm512_store_epi32(&y->vec2[15],f7);

}


int pspm_tee_z(poly *c, polyvecl *s1, polyvecl *y, polyvecl *z) {
    ALIGN(64) uint8_t stable[512];
    __m512i w0, w1, w2, w3;
    __m512i f0, f1, f2, f3, f4, f5, f6, f7;
    __m512i t;
    __m128i t0;
    const __m512i check = _mm512_set1_epi32(GAMMA1 - BETA);
    const __m128i zero = _mm_setzero_si128();
    uint16_t g;


    for (int i = 0; i < L; ++i) {
        //prepare table
        for (int j = 0; j < 16; ++j) {
            t = s1->vec[i].vec2[j];
            t0 = _mm512_cvtepi32_epi8(t);
            _mm_storeu_epi8(stable + 256 + 16 * j, t0);
            _mm_storeu_epi8(stable + 16 * j, _mm_sub_epi8(zero,t0));
        }

        w0 = _mm512_setzero_epi32();
        w1 = _mm512_setzero_epi32();
        w2 = _mm512_setzero_epi32();
        w3 = _mm512_setzero_epi32();

        //compute cs1[i]
        for (int j = 0; j < N; ++j) {
            if (c->coeffs[j] == 1) {
                f0 = _mm512_loadu_epi8(stable + N - j);
                f1 = _mm512_loadu_epi8(stable + N + 64 - j);
                f2 = _mm512_loadu_epi8(stable + N + 128 - j);
                f3 = _mm512_loadu_epi8(stable + N + 182 - j);

                w0 = _mm512_add_epi8(w0, f0);
                w1 = _mm512_add_epi8(w1, f1);
                w2 = _mm512_add_epi8(w2, f2);
                w3 = _mm512_add_epi8(w3, f3);
            } else if (c->coeffs[j] == -1) {
                f0 = _mm512_loadu_epi8(stable + N - j);
                f1 = _mm512_loadu_epi8(stable + N + 64 - j);
                f2 = _mm512_loadu_epi8(stable + N + 128 - j);
                f3 = _mm512_loadu_epi8(stable + N + 182 - j);

                w0 = _mm512_sub_epi8(w0, f0);
                w1 = _mm512_sub_epi8(w1, f1);
                w2 = _mm512_sub_epi8(w2, f2);
                w3 = _mm512_sub_epi8(w3, f3);
            }
        }


        //recover
        f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 0));
        f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 1));
        f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 2));
        f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 3));


        //compute y[i] + cs1[i]

        f0 = _mm512_loadu_epi32(&y->vec[i].vec2[0]);
        f1 = _mm512_loadu_epi32(&y->vec[i].vec2[1]);
        f2 = _mm512_loadu_epi32(&y->vec[i].vec2[2]);
        f3 = _mm512_loadu_epi32(&y->vec[i].vec2[3]);

        f0 = _mm512_add_epi32(f0, f4);
        f1 = _mm512_add_epi32(f1, f5);
        f2 = _mm512_add_epi32(f2, f6);
        f3 = _mm512_add_epi32(f3, f7);

        //check

        f0 = _mm512_abs_epi32(f0);
        f1 = _mm512_abs_epi32(f1);
        f2 = _mm512_abs_epi32(f2);
        f3 = _mm512_abs_epi32(f3);

        g = _mm512_cmpgt_epi32_mask(f0,   check)
            | _mm512_cmpgt_epi32_mask(f1, check)
            | _mm512_cmpgt_epi32_mask(f2, check)
            | _mm512_cmpgt_epi32_mask(f3, check);

        if (g != 0) return 1;

        _mm512_store_epi32(&z->vec[i].vec2[0],f0);
        _mm512_store_epi32(&z->vec[i].vec2[1],f1);
        _mm512_store_epi32(&z->vec[i].vec2[2],f2);
        _mm512_store_epi32(&z->vec[i].vec2[3],f3);

        /*---------------------------------------------------------------------------------------------------------*/

        f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 0));
        f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 1));
        f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 2));
        f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 3));

        f0 = _mm512_loadu_epi32(&y->vec[i].vec2[4]);
        f1 = _mm512_loadu_epi32(&y->vec[i].vec2[5]);
        f2 = _mm512_loadu_epi32(&y->vec[i].vec2[6]);
        f3 = _mm512_loadu_epi32(&y->vec[i].vec2[7]);

        f0 = _mm512_add_epi32(f0, f4);
        f1 = _mm512_add_epi32(f1, f5);
        f2 = _mm512_add_epi32(f2, f6);
        f3 = _mm512_add_epi32(f3, f7);

        f0 = _mm512_abs_epi32(f0);
        f1 = _mm512_abs_epi32(f1);
        f2 = _mm512_abs_epi32(f2);
        f3 = _mm512_abs_epi32(f3);

        g = _mm512_cmpgt_epi32_mask(f0,   check)
            | _mm512_cmpgt_epi32_mask(f1, check)
            | _mm512_cmpgt_epi32_mask(f2, check)
            | _mm512_cmpgt_epi32_mask(f3, check);

        if (g != 0) return 1;

        _mm512_store_epi32(&z->vec[i].vec2[4],f0);
        _mm512_store_epi32(&z->vec[i].vec2[5],f1);
        _mm512_store_epi32(&z->vec[i].vec2[6],f2);
        _mm512_store_epi32(&z->vec[i].vec2[7],f3);

        /*---------------------------------------------------------------------------------------------------------*/

        f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 0));
        f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 1));
        f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 2));
        f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 3));

        f0 = _mm512_loadu_epi32(&y->vec[i].vec2[8 ]);
        f1 = _mm512_loadu_epi32(&y->vec[i].vec2[9 ]);
        f2 = _mm512_loadu_epi32(&y->vec[i].vec2[10]);
        f3 = _mm512_loadu_epi32(&y->vec[i].vec2[11]);

        f0 = _mm512_add_epi32(f0, f4);
        f1 = _mm512_add_epi32(f1, f5);
        f2 = _mm512_add_epi32(f2, f6);
        f3 = _mm512_add_epi32(f3, f7);

        f0 = _mm512_abs_epi32(f0);
        f1 = _mm512_abs_epi32(f1);
        f2 = _mm512_abs_epi32(f2);
        f3 = _mm512_abs_epi32(f3);

        g = _mm512_cmpgt_epi32_mask(f0,   check)
            | _mm512_cmpgt_epi32_mask(f1, check)
            | _mm512_cmpgt_epi32_mask(f2, check)
            | _mm512_cmpgt_epi32_mask(f3, check);

        if (g != 0) return 1;

        _mm512_store_epi32(&z->vec[i].vec2[8 ],f0);
        _mm512_store_epi32(&z->vec[i].vec2[9 ],f1);
        _mm512_store_epi32(&z->vec[i].vec2[10],f2);
        _mm512_store_epi32(&z->vec[i].vec2[11],f3);

        /*---------------------------------------------------------------------------------------------------------*/

        f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 0));
        f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 1));
        f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 2));
        f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 3));

        f0 = _mm512_loadu_epi32(&y->vec[i].vec2[12]);
        f1 = _mm512_loadu_epi32(&y->vec[i].vec2[13]);
        f2 = _mm512_loadu_epi32(&y->vec[i].vec2[14]);
        f3 = _mm512_loadu_epi32(&y->vec[i].vec2[15]);

        f0 = _mm512_add_epi32(f0, f4);
        f1 = _mm512_add_epi32(f1, f5);
        f2 = _mm512_add_epi32(f2, f6);
        f3 = _mm512_add_epi32(f3, f7);

        f0 = _mm512_abs_epi32(f0);
        f1 = _mm512_abs_epi32(f1);
        f2 = _mm512_abs_epi32(f2);
        f3 = _mm512_abs_epi32(f3);

        g = _mm512_cmpgt_epi32_mask(f0,   check)
            | _mm512_cmpgt_epi32_mask(f1, check)
            | _mm512_cmpgt_epi32_mask(f2, check)
            | _mm512_cmpgt_epi32_mask(f3, check);

        if (g != 0) return 1;

        _mm512_store_epi32(&z->vec[i].vec2[12],f0);
        _mm512_store_epi32(&z->vec[i].vec2[13],f1);
        _mm512_store_epi32(&z->vec[i].vec2[14],f2);
        _mm512_store_epi32(&z->vec[i].vec2[15],f3);
    }

    return 0;
}


int pspm_tee_r0(poly *c, polyveck *s2, polyveck *w, polyveck *r0) {
    ALIGN(64) uint8_t stable[512];
    __m512i w0, w1, w2, w3;
    __m512i f0, f1, f2, f3, f4, f5, f6, f7;
    __m512i t;
    __m128i t0;
    const __m128i zero = _mm_setzero_si128();
    const __m512i check = _mm512_set1_epi32(GAMMA2 - BETA);
    uint16_t g;


    for (int i = 0; i < K; ++i) {
        //prepare table
        for (int j = 0; j < 16; ++j) {
            t = s2->vec[i].vec2[j];
            t0 = _mm512_cvtepi32_epi8(t);
            _mm_storeu_epi8(stable + 256 + 16 * j, t0);
            _mm_storeu_epi8(stable + 16 * j, _mm_sub_epi8(zero, t0));
        }

        w0 = _mm512_setzero_epi32();
        w1 = _mm512_setzero_epi32();
        w2 = _mm512_setzero_epi32();
        w3 = _mm512_setzero_epi32();

        //compute cs1[i]
        for (int j = 0; j < N; ++j) {
            if (c->coeffs[j] == 1) {
                f0 = _mm512_loadu_epi8(stable + N - j);
                f1 = _mm512_loadu_epi8(stable + N + 64 - j);
                f2 = _mm512_loadu_epi8(stable + N + 128 - j);
                f3 = _mm512_loadu_epi8(stable + N + 182 - j);

                w0 = _mm512_add_epi8(w0, f0);
                w1 = _mm512_add_epi8(w1, f1);
                w2 = _mm512_add_epi8(w2, f2);
                w3 = _mm512_add_epi8(w3, f3);
            } else if (c->coeffs[j] == -1) {
                f0 = _mm512_loadu_epi8(stable + N - j);
                f1 = _mm512_loadu_epi8(stable + N + 64 - j);
                f2 = _mm512_loadu_epi8(stable + N + 128 - j);
                f3 = _mm512_loadu_epi8(stable + N + 182 - j);

                w0 = _mm512_sub_epi8(w0, f0);
                w1 = _mm512_sub_epi8(w1, f1);
                w2 = _mm512_sub_epi8(w2, f2);
                w3 = _mm512_sub_epi8(w3, f3);
            }
        }


        //recover
        f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 0));
        f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 1));
        f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 2));
        f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w0, 3));

        //compute   w[i] − cs2[i]

        f0 = _mm512_loadu_epi32(&w->vec[i].vec2[0]);
        f1 = _mm512_loadu_epi32(&w->vec[i].vec2[1]);
        f2 = _mm512_loadu_epi32(&w->vec[i].vec2[2]);
        f3 = _mm512_loadu_epi32(&w->vec[i].vec2[3]);

        f0 = _mm512_sub_epi32(f0, f4);
        f1 = _mm512_sub_epi32(f1, f5);
        f2 = _mm512_sub_epi32(f2, f6);
        f3 = _mm512_sub_epi32(f3, f7);

        //check norm

        f0 = _mm512_abs_epi32(f0);
        f1 = _mm512_abs_epi32(f1);
        f2 = _mm512_abs_epi32(f2);
        f3 = _mm512_abs_epi32(f3);

        g = _mm512_cmpgt_epi32_mask(f0,   check)
            | _mm512_cmpgt_epi32_mask(f1, check)
            | _mm512_cmpgt_epi32_mask(f2, check)
            | _mm512_cmpgt_epi32_mask(f3, check);

        if (g != 0) return 1;

        _mm512_store_epi32(&r0->vec[i].vec2[0],f0);
        _mm512_store_epi32(&r0->vec[i].vec2[1],f1);
        _mm512_store_epi32(&r0->vec[i].vec2[2],f2);
        _mm512_store_epi32(&r0->vec[i].vec2[3],f3);

        /*---------------------------------------------------------------------------------------------------------*/


        f0 = _mm512_loadu_epi32(&w->vec[i].vec2[4]);
        f1 = _mm512_loadu_epi32(&w->vec[i].vec2[5]);
        f2 = _mm512_loadu_epi32(&w->vec[i].vec2[6]);
        f3 = _mm512_loadu_epi32(&w->vec[i].vec2[7]);

        f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 0));
        f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 1));
        f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 2));
        f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w1, 3));

        f0 = _mm512_sub_epi32(f0, f4);
        f1 = _mm512_sub_epi32(f1, f5);
        f2 = _mm512_sub_epi32(f2, f6);
        f3 = _mm512_sub_epi32(f3, f7);

        f0 = _mm512_abs_epi32(f0);
        f1 = _mm512_abs_epi32(f1);
        f2 = _mm512_abs_epi32(f2);
        f3 = _mm512_abs_epi32(f3);

        g = _mm512_cmpgt_epi32_mask(f0,   check)
            | _mm512_cmpgt_epi32_mask(f1, check)
            | _mm512_cmpgt_epi32_mask(f2, check)
            | _mm512_cmpgt_epi32_mask(f3, check);

        if (g != 0) return 1;

        _mm512_store_epi32(&r0->vec[i].vec2[4],f0);
        _mm512_store_epi32(&r0->vec[i].vec2[5],f1);
        _mm512_store_epi32(&r0->vec[i].vec2[6],f2);
        _mm512_store_epi32(&r0->vec[i].vec2[7],f3);

        /*---------------------------------------------------------------------------------------------------------*/

        f0 = _mm512_loadu_epi32(&w->vec[i].vec2[8 ]);
        f1 = _mm512_loadu_epi32(&w->vec[i].vec2[9 ]);
        f2 = _mm512_loadu_epi32(&w->vec[i].vec2[10]);
        f3 = _mm512_loadu_epi32(&w->vec[i].vec2[11]);

        f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 0));
        f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 1));
        f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 2));
        f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w2, 3));

        f0 = _mm512_sub_epi32(f0, f4);
        f1 = _mm512_sub_epi32(f1, f5);
        f2 = _mm512_sub_epi32(f2, f6);
        f3 = _mm512_sub_epi32(f3, f7);

        f0 = _mm512_abs_epi32(f0);
        f1 = _mm512_abs_epi32(f1);
        f2 = _mm512_abs_epi32(f2);
        f3 = _mm512_abs_epi32(f3);

        g = _mm512_cmpgt_epi32_mask(f0,   check)
            | _mm512_cmpgt_epi32_mask(f1, check)
            | _mm512_cmpgt_epi32_mask(f2, check)
            | _mm512_cmpgt_epi32_mask(f3, check);

        if (g != 0) return 1;

        _mm512_store_epi32(&r0->vec[i].vec2[8 ],f0);
        _mm512_store_epi32(&r0->vec[i].vec2[9 ],f1);
        _mm512_store_epi32(&r0->vec[i].vec2[10],f2);
        _mm512_store_epi32(&r0->vec[i].vec2[11],f3);

        /*---------------------------------------------------------------------------------------------------------*/


        f0 = _mm512_loadu_epi32(&w->vec[i].vec2[12]);
        f1 = _mm512_loadu_epi32(&w->vec[i].vec2[13]);
        f2 = _mm512_loadu_epi32(&w->vec[i].vec2[14]);
        f3 = _mm512_loadu_epi32(&w->vec[i].vec2[15]);

        f4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 0));
        f5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 1));
        f6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 2));
        f7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(w3, 3));

        f0 = _mm512_sub_epi32(f0, f4);
        f1 = _mm512_sub_epi32(f1, f5);
        f2 = _mm512_sub_epi32(f2, f6);
        f3 = _mm512_sub_epi32(f3, f7);

        f0 = _mm512_abs_epi32(f0);
        f1 = _mm512_abs_epi32(f1);
        f2 = _mm512_abs_epi32(f2);
        f3 = _mm512_abs_epi32(f3);

        g = _mm512_cmpgt_epi32_mask(f0,   check)
            | _mm512_cmpgt_epi32_mask(f1, check)
            | _mm512_cmpgt_epi32_mask(f2, check)
            | _mm512_cmpgt_epi32_mask(f3, check);

        if (g != 0) return 1;

        _mm512_store_epi32(&r0->vec[i].vec2[12],f0);
        _mm512_store_epi32(&r0->vec[i].vec2[13],f1);
        _mm512_store_epi32(&r0->vec[i].vec2[14],f2);
        _mm512_store_epi32(&r0->vec[i].vec2[15],f3);
    }

    return 0;
}