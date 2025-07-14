#ifndef PQCLEAN_DILITHIUM3_AVX2_REJSAMPLE_H
#define PQCLEAN_DILITHIUM3_AVX2_REJSAMPLE_H
#include "params.h"
#include "symmetric.h"
#include "polyvec.h"
#include <stdint.h>

#define REJ_UNIFORM_NBLOCKS ((768+STREAM128_BLOCKBYTES-1)/STREAM128_BLOCKBYTES)
#define REJ_UNIFORM_BUFLEN (REJ_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES)

#define REJ_UNIFORM_ETA_NBLOCKS ((227+STREAM256_BLOCKBYTES-1)/STREAM256_BLOCKBYTES)
#define REJ_UNIFORM_ETA_BUFLEN (REJ_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES)

extern const uint8_t PQCLEAN_DILITHIUM3_AVX2_idxlut[256][8];

unsigned int PQCLEAN_DILITHIUM3_AVX2_rej_uniform_avx(int32_t *r, const uint8_t buf[REJ_UNIFORM_BUFLEN + 8]);

unsigned int PQCLEAN_DILITHIUM3_AVX2_rej_eta_avx(int32_t *r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]);

unsigned int XURQ_AVX2_rej_uniform_avx_s1s3(int32_t *restrict r, const uint8_t *buf, unsigned int num);

unsigned int XURQ_AVX2_rej_uniform_avx_s1s3_final(int32_t *restrict r, const uint8_t *buf, unsigned int num);

unsigned int XURQ_AVX2_rej_eta_avx_with_pack(int32_t *restrict r, uint8_t *pipe, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]);

int XURQ_rej_uniform_avx512(int32_t *r, uint8_t *buf);

void XURQ_poly_uniform_gamma1(polyvecl *z,
                              const uint8_t seed[64],
                              uint16_t nonce);

void XURQ_D3_AVX512_polyz_unpack(poly *r, const uint8_t *buf);

unsigned int rej_eta_avx512(int32_t * restrict r, const uint8_t buf[REJ_UNIFORM_ETA_BUFLEN]);
#endif
