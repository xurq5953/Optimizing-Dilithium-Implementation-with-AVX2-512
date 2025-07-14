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
#include "../fips202x8.h"
#include "../align.h"
#include "../fips202x4.h"
#include "../ntt.h"
#include "../rounding.h"
#include "../uniform_eta_pack.h"
#include "../rejsample.h"
#include "../keccak/KeccakP-1600-times8-SnP.h"
#include "../keccak/keccak8x.h"

#define NTESTS 10000
#define TEST  0

uint64_t t[NTESTS];


static void poly_naivemul(poly *c, const poly *a, const poly *b) {
    unsigned int i, j;
    int32_t r[2 * N] = {0};

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            r[i + j] = (r[i + j] + (int64_t) a->coeffs[i] * b->coeffs[j]) % Q;

    for (i = N; i < 2 * N; i++)
        r[i - N] = (r[i - N] - r[i]) % Q;

    for (i = 0; i < N; i++)
        c->coeffs[i] = r[i];
}


static const int32_t zetas[N] = {
        0,    25847, -2608894,  -518909,   237124,  -777960,  -876248,   466468,
        1826347,  2353451,  -359251, -2091905,  3119733, -2884855,  3111497,  2680103,
        2725464,  1024112, -1079900,  3585928,  -549488, -1119584,  2619752, -2108549,
        -2118186, -3859737, -1399561, -3277672,  1757237,   -19422,  4010497,   280005,
        2706023,    95776,  3077325,  3530437, -1661693, -3592148, -2537516,  3915439,
        -3861115, -3043716,  3574422, -2867647,  3539968,  -300467,  2348700,  -539299,
        -1699267, -1643818,  3505694, -3821735,  3507263, -2140649, -1600420,  3699596,
        811944,   531354,   954230,  3881043,  3900724, -2556880,  2071892, -2797779,
        -3930395, -1528703, -3677745, -3041255, -1452451,  3475950,  2176455, -1585221,
        -1257611,  1939314, -4083598, -1000202, -3190144, -3157330, -3632928,   126922,
        3412210,  -983419,  2147896,  2715295, -2967645, -3693493,  -411027, -2477047,
        -671102, -1228525,   -22981, -1308169,  -381987,  1349076,  1852771, -1430430,
        -3343383,   264944,   508951,  3097992,    44288, -1100098,   904516,  3958618,
        -3724342,    -8578,  1653064, -3249728,  2389356,  -210977,   759969, -1316856,
        189548, -3553272,  3159746, -1851402, -2409325,  -177440,  1315589,  1341330,
        1285669, -1584928,  -812732, -1439742, -3019102, -3881060, -3628969,  3839961,
        2091667,  3407706,  2316500,  3817976, -3342478,  2244091, -2446433, -3562462,
        266997,  2434439, -1235728,  3513181, -3520352, -3759364, -1197226, -3193378,
        900702,  1859098,   909542,   819034,   495491, -1613174,   -43260,  -522500,
        -655327, -3122442,  2031748,  3207046, -3556995,  -525098,  -768622, -3595838,
        342297,   286988, -2437823,  4108315,  3437287, -3342277,  1735879,   203044,
        2842341,  2691481, -2590150,  1265009,  4055324,  1247620,  2486353,  1595974,
        -3767016,  1250494,  2635921, -3548272, -2994039,  1869119,  1903435, -1050970,
        -1333058,  1237275, -3318210, -1430225,  -451100,  1312455,  3306115, -1962642,
        -1279661,  1917081, -2546312, -1374803,  1500165,   777191,  2235880,  3406031,
        -542412, -2831860, -1671176, -1846953, -2584293, -3724270,   594136, -3776993,
        -2013608,  2432395,  2454455,  -164721,  1957272,  3369112,   185531, -1207385,
        -3183426,   162844,  1616392,  3014001,   810149,  1652634, -3694233, -1799107,
        -3038916,  3523897,  3866901,   269760,  2213111,  -975884,  1717735,   472078,
        -426683,  1723600, -1803090,  1910376, -1667432, -1104333,  -260646, -3833893,
        -2939036, -2235985,  -420899, -2286327,   183443,  -976891,  1612842, -3545687,
        -554416,  3919660,   -48306, -1362209,  3937738,  1400424,  -846154,  1976782
};

int32_t montgomery_reduce(int64_t a) {
    int32_t t;

    t = (int32_t)a*QINV;
    t = (a - (int64_t)t*Q) >> 32;
    return t;
}

void poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
    unsigned int i;

    for(i = 0; i < N; ++i)
        c->coeffs[i] = montgomery_reduce((int64_t)a->coeffs[i] * b->coeffs[i]);
}


void ntt(int32_t a[N]) {
    unsigned int len, start, j, k;
    int32_t zeta, t;

    k = 0;
    for(len = 128; len > 0; len >>= 1) {
        for(start = 0; start < N; start = j + len) {
            zeta = zetas[++k];
            for(j = start; j < start + len; ++j) {
                t = montgomery_reduce((int64_t)zeta * a[j + len]);
                a[j + len] = a[j] - t;
                a[j] = a[j] + t;
            }
        }
    }
}

void invntt_tomont(int32_t a[N]) {
    unsigned int start, len, j, k;
    int32_t t, zeta;
    const int32_t f = 41978; // mont^2/256

    k = 256;
    for(len = 1; len < N; len <<= 1) {
        for(start = 0; start < N; start = j + len) {
            zeta = -zetas[--k];
            for(j = start; j < start + len; ++j) {
                t = a[j];
                a[j] = t + a[j + len];
                a[j + len] = t - a[j + len];
                a[j + len] = montgomery_reduce((int64_t)zeta * a[j + len]);
            }
        }
    }

    for(j = 0; j < N; ++j) {
        a[j] = montgomery_reduce((int64_t)f * a[j]);
    }
}


void test_mul() {
    unsigned int i, j;
    uint8_t seed[SEEDBYTES];
    uint16_t nonce = 0;
    poly a, b, c, d;
    poly t1, t2;

    randombytes(seed, sizeof(seed));


    for (i = 0; i < NTESTS; ++i) {
        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_4x(&a, &b, &t1, &t2, seed, nonce, nonce + 1, nonce + 2, nonce + 3);
        nonce += 4;
        c = a;
//        ntt_avx512_bo(c.coeffs);
//        for (j = 0; j < N; ++j)
//            c.coeffs[j] = (int64_t) c.coeffs[j] * -114592 % Q;
//        intt_avx512_bo(c.coeffs);
//        for (j = 0; j < N; ++j) {
//            if ((c.coeffs[j] - a.coeffs[j]) % Q)
//                fprintf(stderr, "ERROR in ntt/invntt: c[%d] = %d != %d\n",
//                        j, c.coeffs[j] % Q, a.coeffs[j]);
//        }

        poly_naivemul(&c, &a, &b);
        // poly_ntt(&a);
//        ntt_avx512_bo(a.coeffs);
        // poly_ntt(&b);
//        ntt_avx512_bo(b.coeffs);
        // poly_pointwise_montgomery(&d, &a, &b);
        pointwise_avx512(d.coeffs, a.coeffs, b.coeffs);
        // poly_invntt_tomont(&d);
//        intt_avx512_bo(d.coeffs);

        for (j = 0; j < N; ++j) {
            if ((d.coeffs[j] - c.coeffs[j]) % Q)
                fprintf(stderr, "ERROR in multiplication: d[%d] = %d != %d\n",
                        j, d.coeffs[j], c.coeffs[j]);
        }
    }
}

void test_mul_so() {
    unsigned int i, j;
    uint8_t seed[SEEDBYTES];
    uint16_t nonce = 0;
    poly a, b, c, d;
    poly t1, t2;

    randombytes(seed, sizeof(seed));
    for (i = 0; i < NTESTS; ++i) {
        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_4x(&a, &b, &t1, &t2, seed, nonce, nonce + 1, nonce + 2, nonce + 3);
        nonce += 4;
        c = a;
        // poly_ntt(&c);
        ntt_so_avx512(c.coeffs);
        for (j = 0; j < N; ++j)
            c.coeffs[j] = (int64_t) c.coeffs[j] * -114592 % Q;
        // poly_invntt_tomont(&c);
        intt_so_avx512(c.coeffs);
        for (j = 0; j < N; ++j) {
            if ((c.coeffs[j] - a.coeffs[j]) % Q)
                fprintf(stderr, "ERROR in ntt/invntt: c[%d] = %d != %d\n",
                        j, c.coeffs[j] % Q, a.coeffs[j]);
        }

        poly_naivemul(&c, &a, &b);
        // poly_ntt(&a);
        ntt_so_avx512(a.coeffs);
        // poly_ntt(&b);
        ntt_so_avx512(b.coeffs);
        // poly_pointwise_montgomery(&d, &a, &b);
        pointwise_avx512(d.coeffs, a.coeffs, b.coeffs);
        // poly_invntt_tomont(&d);
        intt_so_avx512(d.coeffs);

        for (j = 0; j < N; ++j) {
            if ((d.coeffs[j] - c.coeffs[j]) % Q)
                fprintf(stderr, "ERROR in multiplication: d[%d] = %d != %d\n",
                        j, d.coeffs[j], c.coeffs[j]);
        }
    }
}

int32_t decompose(int32_t *a0, int32_t a) {
    int32_t a1;

    a1 = (a + 127) >> 7;
    a1 = (a1 * 11275 + (1 << 23)) >> 24;
    a1 ^= ((43 - a1) >> 31) & a1;
    *a0 = a - a1 * 2 * GAMMA2;
    *a0 -= (((Q - 1) / 2 - *a0) >> 31) & Q;
    return a1;
}

extern int count;

int main(void) {
    unsigned int i;
    size_t smlen;
    uint8_t pk[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t sk2[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_SECRETKEYBYTES];
    uint8_t sm[PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES + CRHBYTES];
    ALIGN(64)
    uint8_t seed[CRHBYTES] = {0};
    polyvecl mat[K];
    polyvecl mat1[K];
    poly *a = &mat[0].vec[0];
    poly *b = &mat[0].vec[1];
    poly *c = &mat[0].vec[2];
    poly *d = &mat[0].vec[3];
    polyvecl s1, s11;
    polyveck s2;
    keccakx8_state state8x;
    keccakx4_state state4x;
    ALIGN(64) uint8_t buf[8][704];


    randombytes(seed, SEEDBYTES);



//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX512_crypto_sign_keypair(pk, sk);
//    }
//    print_results("Keypair:", t, NTESTS);
//
//
    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_AVX512_crypto_sign(sm, &smlen, sm, CRHBYTES, sk);
    }
    print_results("Sign:", t, NTESTS);

//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX512_crypto_sign_verify(sm, PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES, sm, CRHBYTES, pk);
//    }
//    print_results("Verify:", t, NTESTS);





#if TEST == 0

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        KeccakP1600times8_PermuteAll_24rounds(state8x.s);
    }
    print_results("CLEAN Keccak Permutation:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_keccak8x_function(state8x.s);
    }
    print_results("XURQ  Keccak Permutation:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();

    }
    print_results("XURQ  SHAKE128:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        state8x.s[0] = _mm512_set1_epi64(seed[0]);
        state8x.s[1] = _mm512_set1_epi64(seed[1]);
        state8x.s[2] = _mm512_set1_epi64(seed[2]);
        state8x.s[3] = _mm512_set1_epi64(seed[3]);
        state8x.s[4] = _mm512_set_epi64((0x1f << 16) ^ 7, (0x1f << 16) ^ 6, (0x1f << 16) ^ 5,
                                      (0x1f << 16) ^ 4,
                                      (0x1f << 16) ^ 3, (0x1f << 16) ^ 2, (0x1f << 16) ^ 1,
                                      (0x1f << 16) ^ 0);

        for (int j = 5; j < 25; ++j)
            state8x.s[j] = _mm512_setzero_si512();

        state8x.s[20] = _mm512_set1_epi64(0x1ULL << 63);

        XURQ_AVX512_shake128x8_squeezeblocks(&state8x, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], 1);
    }
    print_results("XURQ  SHAKE128:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_4x(&mat[0].vec[0], &mat[0].vec[1], &mat[0].vec[2], &mat[0].vec[3], seed, 0, 1, 2, 3);
        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_4x(&mat[1].vec[0], &mat[1].vec[1], &mat[1].vec[2], &mat[1].vec[3], seed, 256, 257, 258, 259);
        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_4x(&mat[2].vec[0], &mat[2].vec[1], &mat[2].vec[2], &mat[2].vec[3], seed, 512, 513, 514, 515);
        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_4x(&mat[3].vec[0], &mat[3].vec[1], &mat[3].vec[2], &mat[3].vec[3], seed, 768, 769, 770, 771);
    }
    print_results("clean matrix_expand:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_AVX512_poly_uniform_8x(&mat[0].vec[0],
                                    &mat[0].vec[1],
                                    &mat[0].vec[2],
                                    &mat[0].vec[3],
                                    &mat[1].vec[0],
                                    &mat[1].vec[1],
                                    &mat[1].vec[2],
                                    &mat[1].vec[3], seed,
                                    0, 1, 2, 3,
                                    256, 257, 258, 259);
        XURQ_AVX512_poly_uniform_8x(&mat[2].vec[0],
                                    &mat[2].vec[1],
                                    &mat[2].vec[2],
                                    &mat[2].vec[3],
                                    &mat[3].vec[0],
                                    &mat[3].vec[1],
                                    &mat[3].vec[2],
                                    &mat[3].vec[3], seed,
                                    0512, 513, 514, 515,
                                    768, 769, 770, 771);
    }
    print_results("xurq  matrix_expand:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_eta_4x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], seed, 0, 1, 2, 3);
        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_eta_4x(&s2.vec[0], &s2.vec[1], &s2.vec[2], &s2.vec[3], seed, 4, 5, 6, 7);
        for (int j = 0; j < L; j++) {
            PQCLEAN_DILITHIUM2_AVX2_polyeta_pack(sk + 3 * SEEDBYTES +j * POLYETA_PACKEDBYTES, &s1.vec[j]);
        }
        for (int j = 0; j < K; j++) {
            PQCLEAN_DILITHIUM2_AVX2_polyeta_pack(sk + 3 * SEEDBYTES + (L + j) * POLYETA_PACKEDBYTES, &s2.vec[j]);
        }
    }
    print_results("clean uniform_eta and pack sk:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_AVX512_poly_uniform_eta8x_with_pack(
                sk + 3 * SEEDBYTES + 0 * POLYETA_PACKEDBYTES,
                sk + 3 * SEEDBYTES + 1 * POLYETA_PACKEDBYTES,
                sk + 3 * SEEDBYTES + 2 * POLYETA_PACKEDBYTES,
                sk + 3 * SEEDBYTES + 3 * POLYETA_PACKEDBYTES,
                sk + 3 * SEEDBYTES + (L + 0) * POLYETA_PACKEDBYTES,
                sk + 3 * SEEDBYTES + (L + 1) * POLYETA_PACKEDBYTES,
                sk + 3 * SEEDBYTES + (L + 2) * POLYETA_PACKEDBYTES,
                sk + 3 * SEEDBYTES + (L + 3) * POLYETA_PACKEDBYTES,
                &s1, &s2, seed);
    }
    print_results("xurq uniform_eta and pack sk:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM2_AVX2_poly_uniform_gamma1_4x(&s11.vec[0], &s11.vec[1], &s11.vec[2], &s11.vec[3],
                                                       seed, 0, 1, 2, 3);
    }
    print_results("clean uniform_gamma1 y:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        XURQ_AVX512_precompute_gamma1_8x(buf, (uint64_t *) seed,
                                         0, 1, 2, 3,4, 5, 6, 7 );
        XURQ_AVX512_uniform_gamma1_8x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], buf, 0);
        XURQ_AVX512_uniform_gamma1_8x(&s2.vec[0], &s2.vec[1], &s2.vec[2], &s2.vec[3], buf, 1);
    }
    print_results("our  uniform_gamma1 y:", t, NTESTS);

//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        PQCLEAN_DILITHIUM2_AVX2_poly_ntt(a);
//    }
//    print_results("poly_ntt:", t, NTESTS);
//
//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        PQCLEAN_DILITHIUM2_AVX2_poly_invntt_tomont(a);
//    }
//    print_results("poly_invntt_tomont:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        ntt_bo_avx512(a->coeffs);
    }
    print_results("ntt avx512:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        intt_bo_avx512(a->coeffs);
    }
    print_results("inv_ntt avx512:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        ntt_so_avx512(a->coeffs);
    }
    print_results("ntt avx512 so:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        intt_so_avx512(a->coeffs);
    }
    print_results("inv_ntt avx512 so:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM2_AVX2_poly_pointwise_montgomery(c, a, b);
    }
    print_results("poly_pointwise_montgomery:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        pointwise_avx512(c->coeffs, a->coeffs, b->coeffs);
    }
    print_results("xurq poly_pointwise 512:", t, NTESTS);

    for (i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        PQCLEAN_DILITHIUM2_AVX2_poly_challenge(c, seed);
    }
    print_results("poly_challenge:", t, NTESTS);

//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX512_crypto_sign_keypair(pk, sk);
//    }
//    print_results("Keypair:", t, NTESTS);
//
//
//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX512_crypto_sign(sm, &smlen, sm, CRHBYTES, sk);
//    }
//    print_results("Sign:", t, NTESTS);
//    printf("average times: %f\n\n\n",(double )count / NTESTS);
//
//    for (i = 0; i < NTESTS; ++i) {
//        t[i] = cpucycles();
//        XURQ_AVX512_crypto_sign_verify(sm, PQCLEAN_DILITHIUM2_AVX2_CRYPTO_BYTES, sm, CRHBYTES, pk);
//    }
//    print_results("Verify:", t, NTESTS);

#endif

    return 0;
}
