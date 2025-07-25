#ifndef PQCLEAN_DILITHIUM3_AVX2_ALIGN_H
#define PQCLEAN_DILITHIUM3_AVX2_ALIGN_H

#include <immintrin.h>
#include <stdint.h>

#define ALIGNED_UINT8(N)        \
    union {                     \
        uint8_t coeffs[N];      \
        __m256i vec[((N)+31)/32]; \
        __m512i vec2[4];\
    }

#define ALIGNED_INT32(N)        \
    union {                     \
        int32_t coeffs[N];      \
        __m256i vec[((N)+7)/8]; \
        __m512i vec2[16];\
    }

#define ALIGN(x) __attribute__ ((aligned(x)))


#endif
