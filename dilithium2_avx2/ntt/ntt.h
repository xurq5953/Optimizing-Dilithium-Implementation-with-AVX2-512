#ifndef DILITHIUM2_AVX2_NTT_H
#define DILITHIUM2_AVX2_NTT_H

#include <immintrin.h>
#include <stdint.h>
#include "../params.h"


void PQCLEAN_DILITHIUM2_AVX2_pointwise_avx(__m256i *c, const __m256i *a, const __m256i *b, const __m256i *PQCLEAN_DILITHIUM2_AVX2_qdata);

void PQCLEAN_DILITHIUM2_AVX2_pointwise_acc_avx(__m256i *c, const __m256i *a, const __m256i *b, const __m256i *PQCLEAN_DILITHIUM2_AVX2_qdata);

void XRQ_ntt_avx2_bo(int32_t c[256]);  //use our self method

void XRQ_intt_avx2_bo(int32_t c[256]);

void XRQ_ntt_avx2_so(int32_t c[256]);

void XRQ_intt_avx2_so(int32_t c[256]);

void shuffle(int32_t  *c);

#endif
