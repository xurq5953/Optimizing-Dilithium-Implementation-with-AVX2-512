#ifndef DILITHIUM3_AVX2_REJSAMPLE_H
#define DILITHIUM3_AVX2_REJSAMPLE_H
#include "params.h"
#include "symmetric.h"
#include <stdint.h>

#define REJ_UNIFORM_NBLOCKS ((768+STREAM128_BLOCKBYTES-1)/STREAM128_BLOCKBYTES)
#define REJ_UNIFORM_BUFLEN (REJ_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES)

#define REJ_UNIFORM_ETA_NBLOCKS ((227+STREAM256_BLOCKBYTES-1)/STREAM256_BLOCKBYTES)
#define REJ_UNIFORM_ETA_BUFLEN (REJ_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES)

extern const uint8_t DILITHIUM3_AVX2_idxlut[256][8];

unsigned int XURQ_AVX2_rej_uniform_avx_s1s3(int32_t *restrict r, const uint8_t *buf, unsigned int num);

unsigned int XURQ_AVX2_rej_uniform_avx_s1s3_final(int32_t *restrict r, const uint8_t *buf, unsigned int num);

unsigned int XURQ_AVX2_rej_eta_avx_with_pack(int32_t *restrict r, uint8_t *pipe, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]);
#endif
