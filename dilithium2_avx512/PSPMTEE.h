//
// Created by xurq on 2024/1/12.
//

#ifndef OUR_5_AVX512_PSPMTEE_H
#define OUR_5_AVX512_PSPMTEE_H

#include "poly.h"
#include "polyvec.h"

int pspm_tee_z(poly *c, polyvecl *s1, polyvecl *y, polyvecl *z);

int pspm_tee_r0(poly *c, polyveck *s2, polyveck *w, polyveck *r0);

int pspm(poly *c, poly *s, poly *y);

#endif //OUR_5_AVX512_PSPMTEE_H
