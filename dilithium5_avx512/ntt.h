#ifndef DILITHIUM5_AVX2_NTT_H
#define DILITHIUM5_AVX2_NTT_H

#include <immintrin.h>
#include <stdint.h>
#include "params.h"
#include "polyvec.h"



void shuffle(int32_t a[N]);

void ntt_bo_avx512(int32_t a[256]);

void ntt_so_avx512(int32_t a[N]);

void intt_bo_avx512(int32_t a[256]);

void intt_so_avx512(int32_t a[256]);

void pointwise_avx512(int32_t c[N],
                      const int32_t a[N],
                      const int32_t b[N]);


#endif
