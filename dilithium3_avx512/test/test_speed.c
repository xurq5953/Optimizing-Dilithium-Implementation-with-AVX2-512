#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../sign.h"
#include "../api.h"
#include "../poly.h"
#include "../polyvec.h"
#include "../params.h"
#include "cpucycles.h"
#include "speed_print.h"
#include "../randombytes.h"
#include "../fips202x4.h"
#include "../rejsample.h"
#include "../ntt.h"
#include "../cdecl.h"
#include "../consts.h"
#include "../keccak4x/KeccakP-1600-times4-SnP.h"

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

static inline void
ExpandA_row(polyvecl **row, polyvecl buf[2], const uint8_t rho[32], unsigned int i) {
    switch (i) {
        case 0:
            XURQ_AVX512_polyvec_matrix_expand_row0(buf, buf + 1, rho);
            *row = buf;
            break;
        case 1:
            XURQ_AVX512_polyvec_matrix_expand_row1(buf + 1, buf, rho);
            *row = buf + 1;
            break;
        case 2:
            XURQ_AVX512_polyvec_matrix_expand_row2(buf, buf + 1, rho);
            *row = buf;
            break;
        case 3:
            XURQ_AVX512_polyvec_matrix_expand_row3(buf + 1, buf, rho);
            *row = buf + 1;
            break;
        case 4:
            XURQ_AVX512_polyvec_matrix_expand_row4(buf, buf + 1, rho);
            *row = buf;
            break;
        case 5:
            XURQ_AVX512_polyvec_matrix_expand_row5(buf + 1, buf, rho);
            *row = buf + 1;
            break;
    }
}


int main(void) {
    unsigned int i;
    size_t smlen;
    uint8_t pk[PQCLEAN_DILITHIUM3_AVX2_CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[PQCLEAN_DILITHIUM3_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t sk2[PQCLEAN_DILITHIUM3_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t sm[PQCLEAN_DILITHIUM3_AVX2_CRYPTO_BYTES + CRHBYTES];
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
    uint8_t seedbuf[2 * SEEDBYTES + CRHBYTES];
    rhoprime = seed;

//    ExpandS_with_pack(&s11,&s22,sk+ 3 * SEEDBYTES, seed);
//
//    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_eta_4x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], seed, 0, 1, 2, 3);
//    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_eta_4x(&s1.vec[4], &s2.vec[0], &s2.vec[1], &s2.vec[2], seed, 4, 5, 6, 7);
//    PQCLEAN_DILITHIUM3_AVX2_poly_uniform_eta_4x(&s2.vec[3], &s2.vec[4], &s2.vec[5], &t0, seed, 8, 9, 10, 11);
//
//    /* Pack secret vectors */
//    for (i = 0; i < L; i++) {
//        PQCLEAN_DILITHIUM3_AVX2_polyeta_pack(sk2 + 3 * SEEDBYTES + i * POLYETA_PACKEDBYTES, &s1.vec[i]);
//    }
//    for (i = 0; i < K; i++) {
//        PQCLEAN_DILITHIUM3_AVX2_polyeta_pack(sk2 + 3 * SEEDBYTES + (L + i)*POLYETA_PACKEDBYTES, &s2.vec[i]);
//    }
//
//    for (int j = 0; j < 256; ++j) {
//        printf("%d ", 4- s11.vec[0].coeffs[j]);
//    }
//    printf("\n");
//    for (int j = 0; j < 256; ++j) {
//        printf("%d ",4-  s1.vec[0].coeffs[j]);
//    }
//    printf("\n");
//    printf("\n");
//
//
//    for (int j =  3 * SEEDBYTES+ 4 *POLYETA_PACKEDBYTES; j <  3 * SEEDBYTES + 11 *POLYETA_PACKEDBYTES; ++j) {
//        printf("%#x ", sk[j]);
//    }
//    printf("\n");
//    for (int j =  3 * SEEDBYTES+ 4 *POLYETA_PACKEDBYTES; j <  3 * SEEDBYTES + 11 *POLYETA_PACKEDBYTES; ++j) {
//        printf("%#x ",  sk2[j]);
//    }

//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX512_crypto_sign_keypair(pk, sk);
//    }
//    print_results("ours  Keypair:", t, NTESTS);


//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX512_crypto_sign(sm, &smlen, sm, CRHBYTES, sk);
//    }
//    print_results("ours  Sign:", t, NTESTS);


//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX512_crypto_sign_keypair(pk, sk);
//    }
//    print_results("ours  Keypair:", t, NTESTS);


//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX512_crypto_sign(sm, &smlen, sm, CRHBYTES, sk);
//    }
//    print_results("ours  Sign:", t, NTESTS);

//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX512_crypto_sign_verify(sm, PQCLEAN_DILITHIUM3_AVX2_CRYPTO_BYTES, sm, CRHBYTES, pk);
//    }
//    print_results("ours  Verify:", t, NTESTS);

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
        for (int j  = 0; j < K; j++) {
            ExpandA_row(&row, rowbuf, seed, j);
        }
    }
    print_results("ours ExpandA:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        shake256(seedbuf, 2 * SEEDBYTES + CRHBYTES, seedbuf, SEEDBYTES);
    }
    print_results("ours shake256:", t, NTESTS);


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM3_AVX2_poly_uniform_gamma1_4x(&z.vec[0], &z.vec[1], &z.vec[2], &z.vec[3],
                                                       rhoprime, nonce, nonce + 1, nonce + 2, nonce + 3);
        PQCLEAN_DILITHIUM3_AVX2_poly_uniform_gamma1(&z.vec[4], rhoprime, nonce + 4);
        nonce += 5;
    }
    print_results("ours ExpandMask:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_poly_uniform_gamma1(&z, seed, nonce);
        nonce += 5;
    }
    print_results("ours ExpandMask:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        ExpandS_with_pack(&s11,&s22,sk+ 3 * SEEDBYTES, seed);
    }
    print_results("ours ExpandS_with_pack:", t, NTESTS);


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_AVX512_crypto_sign_keypair(pk, sk);
    }
    print_results("ours  Keypair:", t, NTESTS);


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_AVX512_crypto_sign(sm, &smlen, sm, CRHBYTES, sk);
    }
    print_results("ours  Sign:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_AVX512_crypto_sign_verify(sm, PQCLEAN_DILITHIUM3_AVX2_CRYPTO_BYTES, sm, CRHBYTES, pk);
    }
    print_results("ours  Verify:", t, NTESTS);




#endif

    return 0;
}
