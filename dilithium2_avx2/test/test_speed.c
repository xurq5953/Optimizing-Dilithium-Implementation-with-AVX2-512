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



#define NROUNDS 24
static const uint64_t KeccakF_RoundConstants[NROUNDS] = {
        (uint64_t) 0x0000000000000001ULL,
        (uint64_t) 0x0000000000008082ULL,
        (uint64_t) 0x800000000000808aULL,
        (uint64_t) 0x8000000080008000ULL,
        (uint64_t) 0x000000000000808bULL,
        (uint64_t) 0x0000000080000001ULL,
        (uint64_t) 0x8000000080008081ULL,
        (uint64_t) 0x8000000000008009ULL,
        (uint64_t) 0x000000000000008aULL,
        (uint64_t) 0x0000000000000088ULL,
        (uint64_t) 0x0000000080008009ULL,
        (uint64_t) 0x000000008000000aULL,
        (uint64_t) 0x000000008000808bULL,
        (uint64_t) 0x800000000000008bULL,
        (uint64_t) 0x8000000000008089ULL,
        (uint64_t) 0x8000000000008003ULL,
        (uint64_t) 0x8000000000008002ULL,
        (uint64_t) 0x8000000000000080ULL,
        (uint64_t) 0x000000000000800aULL,
        (uint64_t) 0x800000008000000aULL,
        (uint64_t) 0x8000000080008081ULL,
        (uint64_t) 0x8000000000008080ULL,
        (uint64_t) 0x0000000080000001ULL,
        (uint64_t) 0x8000000080008008ULL
};

static unsigned int rej_eta(int32_t *a,
                            unsigned int len,
                            const uint8_t *buf,
                            unsigned int buflen) {
    unsigned int ctr, pos;
    uint32_t t0, t1;

    ctr = pos = 0;
    while (ctr < len && pos < buflen) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

        if (t0 < 15) {
            t0 = t0 - (205 * t0 >> 10) * 5;
            a[ctr++] = 2 - t0;
        }
        if (t1 < 15 && ctr < len) {
            t1 = t1 - (205 * t1 >> 10) * 5;
            a[ctr++] = 2 - t1;
        }
    }

    return ctr;
}


int main(void) {
    unsigned int i;
    size_t smlen;
    uint8_t pk[DILITHIUM2_CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[DILITHIUM2_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t sk2[DILITHIUM2_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t sm[DILITHIUM2_AVX2_CRYPTO_BYTES + CRHBYTES];
    __attribute__((aligned(32)))
    uint8_t seed[CRHBYTES] = {0};
    polyvecl mat[K],z;
    poly *a = &mat[0].vec[0];
    poly *b = &mat[0].vec[1];
    poly *c = &mat[0].vec[2];
    poly *d = &mat[0].vec[3];
    polyvecl s1,s11;
    polyveck s2,s22;
    ALIGNED_UINT8(864) buf[4];

    keccakx4_state state;

   randombytes(seed, SEEDBYTES);
    // memset(seed,0,SEEDBYTES);



    uint8_t r0[1300];
    uint8_t r1[1300];



#if TEST == 1



    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        dilithium2_keypair_avx2(pk, sk);
    }
    print_results("ours  Keypair:", t, NTESTS);

    __m256i s[25];
    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        KeccakP1600times4_PermuteAll_24rounds(s);
    }
    print_results(" KeccakP1600times4_PermuteAll_24rounds:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM2_AVX2_f1600x4(s, KeccakF_RoundConstants);
    }
    print_results(" KeccakP1600times4_PermuteAll_24rounds:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        dilithium2_sign_avx2(sm, &smlen, sm, CRHBYTES, sk);
    }
    print_results("ours  Sign:", t, NTESTS);


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        dilithium2_verify_avx2(sm, DILITHIUM2_AVX2_CRYPTO_BYTES, sm, CRHBYTES, pk);
    }
    print_results("ours  Verify:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        dilithium2_verify_avx2(sm, DILITHIUM2_AVX2_CRYPTO_BYTES, sm, CRHBYTES, pk);
    }
    print_results("ours  Verify:", t, NTESTS);


#endif

    return 0;
}
