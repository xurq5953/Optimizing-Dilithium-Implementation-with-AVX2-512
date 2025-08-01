#ifndef PQCLEAN_DILITHIUM3_AVX2_NTT_H
#define PQCLEAN_DILITHIUM3_AVX2_NTT_H

#include <immintrin.h>



void PQCLEAN_DILITHIUM3_AVX2_pointwise_avx(__m256i *c, const __m256i *a, const __m256i *b, const __m256i *PQCLEAN_DILITHIUM3_AVX2_qdata);
void PQCLEAN_DILITHIUM3_AVX2_pointwise_acc_avx(__m256i *c, const __m256i *a, const __m256i *b, const __m256i *PQCLEAN_DILITHIUM3_AVX2_qdata);


void ntt_avx2_bo(int32_t c[256]);  //use our self method

void intt_bo_avx2(int32_t c[256]);

void ntt_avx2_so(int32_t c[256]);

void intt_so_avx2(int32_t c[256]);

void shuffle(int32_t  *c);

#endif
