#ifndef PQCLEAN_DILITHIUM5_AVX2_CONSTS_H
#define PQCLEAN_DILITHIUM5_AVX2_CONSTS_H
#include "align.h"
#include "cdecl.h"

#define QINV 58728449 // q^(-1) mod 2^32
#define MONT (-4186625) // 2^32 mod q
#define DIV 41978 // mont^2/256
#define DIV_QINV (-8395782)


typedef ALIGNED_INT32(624) qdata_t;
extern const qdata_t PQCLEAN_DILITHIUM5_AVX2_qdata;

extern const int32_t  inte_qdata[512];

extern const int32_t  inv_qdata[512];

extern const int64_t inte_data2[32];

#endif
