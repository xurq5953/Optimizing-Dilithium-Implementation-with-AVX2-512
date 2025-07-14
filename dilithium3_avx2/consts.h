#ifndef DILITHIUM3_AVX2_CONSTS_H
#define DILITHIUM3_AVX2_CONSTS_H
#include "align.h"
#include "cdecl.h"


typedef ALIGNED_INT32(624) qdata_t;
extern const qdata_t PQCLEAN_DILITHIUM3_AVX2_qdata;

extern const int32_t  inte_qdata[512];

extern const int32_t  inv_qdata[512];

extern const int64_t inte_data2[32];

#endif
