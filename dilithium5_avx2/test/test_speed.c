#include <stdint.h>
#include <stdio.h>
#include "../sign.h"
#include "../api.h"
#include "../poly.h"
#include "../polyvec.h"
#include "../params.h"
#include "cpucycles.h"
#include "speed_print.h"
#include "../randombytes.h"
#include "../fips202x4.h"

#define NTESTS 20000

#define TEST 0

uint64_t t[NTESTS];






#define forward_shuffle(a,b) \
t = _mm256_permutevar8x32_epi32(a, idx); \
f = _mm256_permutevar8x32_epi32(b, idx2); \
a = _mm256_blend_epi32(t,f,0xf0); \
b = _mm256_blend_epi32(t,f,0x0f);        \
b = _mm256_permutevar8x32_epi32(b, idx3);\
\


#define LOAD(n,m) \
z0  = _mm256_load_si256((c + (n)     ));\
z1  = _mm256_load_si256((c + (n) + 8 ));\
z2  = _mm256_load_si256((c + (n) + 16));\
z3  = _mm256_load_si256((c + (n) + 24));\
z4  = _mm256_load_si256((c + (m)     ));\
z5  = _mm256_load_si256((c + (m) + 8 ));\
z6  = _mm256_load_si256((c + (m) + 16));\
z7  = _mm256_load_si256((c + (m) + 24));\
\



#define STORE(n,m) \
_mm256_store_si256((c + (n)     ), z0);\
_mm256_store_si256((c + (n) + 8 ), z1);\
_mm256_store_si256((c + (n) + 16), z2);\
_mm256_store_si256((c + (n) + 24), z3);\
_mm256_store_si256((c + (m)     ), z4);\
_mm256_store_si256((c + (m) + 8 ), z5);\
_mm256_store_si256((c + (m) + 16), z6);\
_mm256_store_si256((c + (m) + 24), z7);\
\





int main(void) {
    unsigned int i;
    size_t smlen;
    uint8_t pk[PQCLEAN_DILITHIUM5_AVX2_CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[PQCLEAN_DILITHIUM5_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t sm[PQCLEAN_DILITHIUM5_AVX2_CRYPTO_BYTES + CRHBYTES];
    __attribute__((aligned(32)))
    uint8_t seed[CRHBYTES] = {0};
    polyvecl mat[K],z;
    poly *a = &mat[0].vec[0];
    poly *b = &mat[0].vec[1];
    poly *c = &mat[0].vec[2];
    polyvecl s1;
    polyveck s2;
    uint32_t nonce = 0;

    randombytes(seed, SEEDBYTES);






    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_crypto_sign_verify(sm, PQCLEAN_DILITHIUM5_AVX2_CRYPTO_BYTES, sm, CRHBYTES, pk);
    }
    print_results("clean Verify:", t, NTESTS);


#if TEST == 0


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_polyvec_matrix_expand(mat, seed);
    }
    print_results("ExpandA clean:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_AVX2_polyvec_matrix_expand(mat, seed);
    }
    print_results("ExpandA our:", t, NTESTS);


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_poly_uniform_eta_4x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], seed, 0, 1, 2, 3);
        PQCLEAN_DILITHIUM5_AVX2_poly_uniform_eta_4x(&s1.vec[4], &s1.vec[5], &s1.vec[6], &s2.vec[0], seed, 4, 5, 6, 7);
        PQCLEAN_DILITHIUM5_AVX2_poly_uniform_eta_4x(&s2.vec[1], &s2.vec[2], &s2.vec[3], &s2.vec[4], seed, 8, 9, 10, 11);
        PQCLEAN_DILITHIUM5_AVX2_poly_uniform_eta_4x(&s2.vec[5], &s2.vec[6], &s2.vec[7], a, seed, 12, 13, 14, 15);

        /* Pack secret vectors */
        for (int j = 0; j < L; j++) {
            PQCLEAN_DILITHIUM5_AVX2_polyeta_pack(sk + 3 * SEEDBYTES + j * POLYETA_PACKEDBYTES, &s1.vec[j]);
        }
        for (int j = 0; j < K; j++) {
            PQCLEAN_DILITHIUM5_AVX2_polyeta_pack(sk + 3 * SEEDBYTES + (L + j) * POLYETA_PACKEDBYTES, &s2.vec[j]);
        }
    }
    print_results("clean sampling s1 & s2 uniform_eta with packing:", t, NTESTS);


//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX2_poly_uniform_eta_4x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], seed, 0, 1, 2, 3);
//        XURQ_AVX2_poly_uniform_eta_4x(&s1.vec[4], &s1.vec[5], &s1.vec[6], &s2.vec[0], seed, 4, 5, 6, 7);
//        XURQ_AVX2_poly_uniform_eta_4x(&s2.vec[1], &s2.vec[2], &s2.vec[3], &s2.vec[4], seed, 8, 9, 10, 11);
//        XURQ_AVX2_poly_uniform_eta_4x(&s2.vec[5], &s2.vec[6], &s2.vec[7], a, seed, 12, 13, 14, 15);
//
//        /* Pack secret vectors */
//        for (int j = 0; j < L; j++) {
//            PQCLEAN_DILITHIUM5_AVX2_polyeta_pack(sk + 3 * SEEDBYTES + j * POLYETA_PACKEDBYTES, &s1.vec[j]);
//        }
//        for (int j = 0; j < K; j++) {
//            PQCLEAN_DILITHIUM5_AVX2_polyeta_pack(sk + 3 * SEEDBYTES + (L + j) * POLYETA_PACKEDBYTES, &s2.vec[j]);
//        }
//    }
//    print_results("XURQ sampling s1 & s2 uniform_eta with packing:", t, NTESTS);


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        ExpandS_with_pack(&s1, &s2, sk + 3 * SEEDBYTES, seed);
    }
    print_results("ours sampling s1 & s2 uniform_eta with pack", t, NTESTS);


    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_AVX2_poly_uniform_gamma1_4x(&z.vec[0], &z.vec[1], &z.vec[2], &z.vec[3],
                                                       seed, nonce, nonce + 1, nonce + 2, nonce + 3);
        XURQ_AVX2_poly_uniform_gamma1_4x(&z.vec[4], &z.vec[5], &z.vec[6], a,
                                                       seed, nonce + 4, nonce + 5, nonce + 6, 0);
    }
    print_results("xurq expandMask uniform_gamma_4x:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_poly_uniform_gamma1_4x(&z.vec[0], &z.vec[1], &z.vec[2], &z.vec[3],
                                                       seed, nonce, nonce + 1, nonce + 2, nonce + 3);
        PQCLEAN_DILITHIUM5_AVX2_poly_uniform_gamma1_4x(&z.vec[4], &z.vec[5], &z.vec[6], a,
                                                       seed, nonce + 4, nonce + 5, nonce + 6, 0);
    }
    print_results("clean expandMask uniform_gamma_4x:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_poly_ntt(a);
    }
    print_results("poly_ntt:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_poly_invntt_tomont(a);
    }
    print_results("poly_invntt_tomont:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 7; ++k) {
                PQCLEAN_DILITHIUM5_AVX2_poly_nttunpack(&mat[j].vec[k]);
            }
        }
    }
    print_results("PQCLEAN shuffle A D5:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_poly_pointwise_montgomery(c, a, b);
    }
    print_results("poly_pointwise_montgomery:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_poly_challenge(c, seed);
    }
    print_results("poly_challenge:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_crypto_sign_keypair(pk, sk);
    }
    print_results("clean Keypair:", t, NTESTS);



    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_crypto_sign(sm, &smlen, sm, CRHBYTES, sk);
    }
    print_results("clean Sign:", t, NTESTS);



    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM5_AVX2_crypto_sign_verify(sm, PQCLEAN_DILITHIUM5_AVX2_CRYPTO_BYTES, sm, CRHBYTES, pk);
    }
    print_results("clean Verify:", t, NTESTS);




#endif

    return 0;
}
