#ifndef DILITHIUM2_AVX2_ROUNDING_H
#define DILITHIUM2_AVX2_ROUNDING_H
#include "params.h"
#include <immintrin.h>
#include <stdint.h>

void power2round_avx2(__m256i *a1, __m256i *a0, const __m256i *a);
void decompose_avx2(__m256i *a1, __m256i *a0, const __m256i *a);
unsigned int make_hint_avx2(uint8_t hint[256], const __m256i *a0, const __m256i *a1);
void use_hint_avx2(__m256i *b, const __m256i *a, const __m256i *hint);

#endif
