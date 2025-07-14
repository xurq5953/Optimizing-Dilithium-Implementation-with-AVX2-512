#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../sign.h"
#include "../poly.h"
#include "../polyvec.h"
#include "../params.h"
#include "cpucycles.h"
#include "speed_print.h"
#include "../randombytes.h"
#include "../keccak/fips202x4.h"
#include "../rejsample.h"
#include "../ntt/ntt.h"
#include "../cdecl.h"
#include "../consts.h"
#include "../keccak/KeccakP-1600-times4-SnP.h"
#include "../packing.h"

#define NTESTS 10000

#define TEST 1

#define TEST_STRATEGY 1

uint64_t t[NTESTS];



int main(void) {
    unsigned int i;
    size_t smlen;
    uint8_t pk[DILITHIUM3_AVX2_CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[DILITHIUM3_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t sk2[DILITHIUM3_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t sm[DILITHIUM3_AVX2_CRYPTO_BYTES + CRHBYTES];
    __attribute__((aligned(32)))
    uint8_t seed[64] = {0};
    polyvecl mat[K],z;
    poly *a = &mat[0].vec[0];
    poly *b = &mat[0].vec[1];
    poly *c = &mat[0].vec[2];
    poly *d = &mat[0].vec[3];
    polyvecl s1,s11;
    polyveck s2,s22;
    ALIGNED_UINT8(864) buf[4];
    const uint8_t *rho, *rhoprime, *key;
    polyvecl rowbuf[2];
    polyvecl *row = rowbuf;
    uint64_t nonce = 0;
    keccakx4_state state;
    poly t1, t0;
    randombytes(seed, 64);
     memset(seed,0,64);

    rhoprime = seed;
    randombytes(buf[0].coeffs, 128);




#if TEST == 1


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        for (int j  = 0; j < K; j++) {
            ExpandA_row(&row, rowbuf, seed, j);
        }
    }
    print_results("ours ExpandA:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < L; ++k) {
                shuffle(mat[j].vec[k].coeffs);
            }
        }
    }
    print_results("XRQ shuffle A:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        for (int j  = 0; j < K; j++) {
            ExpandA_row(&row, rowbuf, seed, j);
        }
    }
    print_results("ours ExpandA:", t, NTESTS);


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        ExpandS_with_pack(&s11,&s22,sk+ 3 * SEEDBYTES, seed);
    }
    print_results("ours ExpandS_with_pack:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        dilithium3_keypair_avx2(pk, sk);
    }
    print_results("ours  Keypair:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        dilithium3_sign_avx2(sm, &smlen, sm, CRHBYTES, sk);
    }

    print_results("ours  Sign:", t, NTESTS);
    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        dilithium3_verify_avx2(sm, DILITHIUM3_AVX2_CRYPTO_BYTES, sm, CRHBYTES, pk);
    }
    print_results("ours  Verify:", t, NTESTS);



#endif

    return 0;
}
