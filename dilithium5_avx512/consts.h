#ifndef DILITHIUM2_AVX2_CONSTS_H
#define DILITHIUM2_AVX2_CONSTS_H
#include "align.h"


typedef ALIGNED_INT32(624) qdata_t;


extern const int32_t qdata2[];

#define QINV16X 0
#define Q16X 16
#define DIV16X 32
#define ZETAS 48
#define ZETAS_INV 560

extern const int32_t psi[];

extern const int32_t ipsi[];

extern const int64_t ipsi2[];

#endif
