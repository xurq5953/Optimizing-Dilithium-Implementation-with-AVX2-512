#ifndef PQCLEAN_DILITHIUM2_AVX2_ALIGN_H
#define PQCLEAN_DILITHIUM2_AVX2_ALIGN_H

#include <immintrin.h>
#include <stdint.h>


#ifdef ALIGN
#undef ALIGN
#endif

#if defined(__GNUC__)
#define ALIGN(x) __attribute__ ((aligned(x)))
#elif defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#elif defined(__ARMCC_VERSION)
#define ALIGN(x) __align(x)
#else
#define ALIGN(x)
#endif




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

#endif
