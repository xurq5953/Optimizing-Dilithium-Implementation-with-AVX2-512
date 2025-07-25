#ifndef DILITHIUM3_AVX2_SYMMETRIC_H
#define DILITHIUM3_AVX2_SYMMETRIC_H
#include "keccak/fips202.h"
#include "params.h"
#include <stdint.h>



typedef shake128incctx stream128_state;
typedef shake256incctx stream256_state;


void PQCLEAN_DILITHIUM3_AVX2_dilithium_shake256_stream_init(shake256incctx *state, const uint8_t seed[CRHBYTES], uint16_t nonce);

#define STREAM128_BLOCKBYTES SHAKE128_RATE
#define STREAM256_BLOCKBYTES SHAKE256_RATE


#define stream256_init(STATE, SEED, NONCE) PQCLEAN_DILITHIUM3_AVX2_dilithium_shake256_stream_init(STATE, SEED, NONCE)
#define stream256_squeezeblocks(OUT, OUTBLOCKS, STATE) shake256_inc_squeeze(OUT, (OUTBLOCKS)*(SHAKE256_RATE), STATE)
#define stream256_release(STATE) shake256_inc_ctx_release(STATE)


#endif
