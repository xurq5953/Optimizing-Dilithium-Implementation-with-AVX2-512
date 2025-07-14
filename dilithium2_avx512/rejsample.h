#ifndef PQCLEAN_DILITHIUM2_AVX2_REJSAMPLE_H
#define PQCLEAN_DILITHIUM2_AVX2_REJSAMPLE_H
#include "params.h"
#include "symmetric.h"
#include "poly.h"
#include <stdint.h>

#define REJ_UNIFORM_NBLOCKS ((768+STREAM128_BLOCKBYTES-1)/STREAM128_BLOCKBYTES)
#define REJ_UNIFORM_BUFLEN (REJ_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES)

#define REJ_UNIFORM_ETA_NBLOCKS ((136+STREAM256_BLOCKBYTES-1)/STREAM256_BLOCKBYTES)
#define REJ_UNIFORM_ETA_BUFLEN (REJ_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES)

extern const uint8_t PQCLEAN_DILITHIUM2_AVX2_idxlut[256][8];

unsigned int PQCLEAN_DILITHIUM2_AVX2_rej_uniform_avx(int32_t *r, const uint8_t buf[REJ_UNIFORM_BUFLEN + 8]);

unsigned int PQCLEAN_DILITHIUM2_AVX2_rej_eta_avx(int32_t *r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]);


// void XURQ_AVX512_polyz_unpack4x(poly *r0,poly *r1,poly *r2,poly *r3,
//                                 const uint8_t buf[4][704]);

void XURQ_AVX512_polyz_unpack4x(poly *r0, poly *r1, poly *r2, poly *r3,
                                const uint8_t *buf0,
                                const uint8_t *buf1,
                                const uint8_t *buf2,
                                const uint8_t *buf3);

        void XURQ_AVX512_rej_eta8x(int32_t * r0,
                           int32_t * r1,
                           int32_t * r2,
                           int32_t * r3,
                           int32_t * r4,
                           int32_t * r5,
                           int32_t * r6,
                           int32_t * r7,
                           uint32_t ctr[8],
                           uint8_t buf[8][136]);

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

int XURQ_rej_uniform_avx512(int32_t *r, uint8_t *buf);

#endif
