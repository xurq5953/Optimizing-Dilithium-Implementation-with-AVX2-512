#ifndef PQCLEAN_DILITHIUM2_AVX2_NTT_H
#define PQCLEAN_DILITHIUM2_AVX2_NTT_H

#include <immintrin.h>
#include <stdint.h>
#include "params.h"

void PQCLEAN_DILITHIUM2_AVX2_ntt_avx(__m256i *a, const __m256i *PQCLEAN_DILITHIUM2_AVX2_qdata);
void PQCLEAN_DILITHIUM2_AVX2_invntt_avx(__m256i *a, const __m256i *PQCLEAN_DILITHIUM2_AVX2_qdata);

void PQCLEAN_DILITHIUM2_AVX2_nttunpack_avx(__m256i *a);

void PQCLEAN_DILITHIUM2_AVX2_pointwise_avx(__m256i *c, const __m256i *a, const __m256i *b, const __m256i *PQCLEAN_DILITHIUM2_AVX2_qdata);
void PQCLEAN_DILITHIUM2_AVX2_pointwise_acc_avx(__m256i *c, const __m256i *a, const __m256i *b, const __m256i *PQCLEAN_DILITHIUM2_AVX2_qdata);


void shuffle(int32_t a[N]);

void ntt_bo_avx512(int32_t a[N]);

void ntt_so_avx512(int32_t a[N]);

void intt_bo_avx512(int32_t a[N]);

void intt_so_avx512(int32_t a[N]);

void pointwise_avx512(int32_t c[N],
                      const int32_t a[N],
                      const int32_t b[N]);






#endif
