#ifndef DILITHIUM5_AVX2_REJSAMPLE_H
#define DILITHIUM5_AVX2_REJSAMPLE_H
#include "params.h"
#include "symmetric.h"
#include "polyvec.h"
#include <stdint.h>

#define REJ_UNIFORM_NBLOCKS ((768+STREAM128_BLOCKBYTES-1)/STREAM128_BLOCKBYTES)
#define REJ_UNIFORM_BUFLEN (REJ_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES)

#define REJ_UNIFORM_ETA_NBLOCKS ((136+STREAM256_BLOCKBYTES-1)/STREAM256_BLOCKBYTES)
#define REJ_UNIFORM_ETA_BUFLEN (REJ_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES)


void XURQ_AVX512_rej_uniform8x(int32_t * r0,
                               int32_t * r1,
                               int32_t * r2,
                               int32_t * r3,
                               int32_t * r4,
                               int32_t * r5,
                               int32_t * r6,
                               int32_t * r7,
                               uint32_t ctr[8],
                               uint8_t buf[8][864]);

void ExpandMask(polyvecl *z,
                const uint8_t seed[64],
                uint16_t nonce);

int XURQ_rej_uniform_avx512(int32_t *r, uint8_t *buf);


#endif
