#include <stdint.h>
#include <stdio.h>
#include "../sign.h"
#include "../poly.h"
#include "../polyvec.h"
#include "../params.h"
#include "cpucycles.h"
#include "speed_print.h"
#include "../randombytes.h"
#include "../ntt.h"
#include "../fips202x8.h"
#include "../common/keccak/KeccakP-1600-times8-SnP.h"
#include "../uniform_eta_pack.h"
#include "../rejsample.h"
#include "../PSPMTEE.h"

#define NTESTS 10000

#define TEST 1

uint64_t t[NTESTS];

int main(void) {
    unsigned int i;
    size_t smlen;
    uint8_t pk[DILITHIUM5_AVX2_CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[DILITHIUM5_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t sk2[DILITHIUM5_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t pk2[DILITHIUM5_AVX2_CRYPTO_PUBLICKEYBYTES];
    uint8_t m[DILITHIUM5_AVX2_CRYPTO_BYTES + CRHBYTES];
    uint8_t sm[DILITHIUM5_AVX2_CRYPTO_BYTES + CRHBYTES];
    uint8_t sm2[DILITHIUM5_AVX2_CRYPTO_BYTES + CRHBYTES];
    __attribute__((aligned(64)))
    uint8_t seed[CRHBYTES] = {0};
    polyvecl mat[K], z;
    polyvecl mat1[K];
    poly *a = &mat[0].vec[0];
    poly *b = &mat[0].vec[1];
    poly *c = &mat[0].vec[2];
    polyvecl s1;
    polyveck s2;
    keccakx8_state state8x;
    ALIGN(64) uint8_t buf[8][704];
    randombytes(seed, SEEDBYTES);
    __m256i f;
    uint8_t seedbuf[2 * SEEDBYTES + CRHBYTES];

#if TEST == 1



    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        dilithium5_sign_keypair_avx512(pk, sk);
    }
    print_results("XURQ Keypair:", t, NTESTS);


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        dilithium5_sign_avx512(sm, &smlen, sm, CRHBYTES, sk);
    }
    print_results("XURQ Sign:", t, NTESTS);


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        dilithium5_sign_verify_avx512(sm, DILITHIUM5_AVX2_CRYPTO_BYTES, sm, CRHBYTES, pk);
    }
    print_results("XURQ Verify:", t, NTESTS);


#endif

    return 0;
}
