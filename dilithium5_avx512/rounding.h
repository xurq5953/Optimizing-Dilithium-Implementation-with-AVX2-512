#ifndef DILITHIUM5_AVX2_ROUNDING_H
#define DILITHIUM5_AVX2_ROUNDING_H
#include "params.h"
#include <immintrin.h>
#include <stdint.h>

void XURQ_AVX512_decompose(__m512i *a1, __m512i *a0, const __m512i *a);

void XURQ_AVX512_power2round_avx(__m512i *a1, __m512i *a0, const __m512i *a);

unsigned int make_hint_avx512(uint8_t hint[N], const __m512i *restrict a0, const __m512i *restrict a1);

void use_hint_avx512(__m512i *b, const __m512i *a, const __m512i *restrict hint);
#endif
