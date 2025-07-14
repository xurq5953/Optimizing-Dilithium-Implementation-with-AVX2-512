//
// Created by xurq on 2022/10/19.
//
#include "../fips202x8.h"
#include "../fips202x4.h"
#include "../poly.h"
#include "../polyvec.h"
#include <stdio.h>
#include <time.h>
#include <x86intrin.h>


void test_shake128_vectors() {
    keccakx8_state state1;
    keccakx4_state state2;

    ALIGN(64) uint8_t in0[1024] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe};
    ALIGN(64) uint8_t in1[1024] = {0x69, 0x16, 0x66, 0x6c, 0x25, 0x2f, 0x24, 0x92};
    ALIGN(64) uint8_t in2[1024] = {0x41, 0x1a, 0xf2, 0x3e, 0xc7, 0xc7, 0xdc};
    ALIGN(64) uint8_t in3[1024] = {0x3c, 0xd9, 0x5a, 0x70, 0x37, 0x7b, 0x3};
    ALIGN(64) uint8_t in4[1024] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe};
    ALIGN(64) uint8_t in5[1024] = {0x69, 0x16, 0x66, 0x6c, 0x25, 0x2f, 0x24, 0x92};
    ALIGN(64) uint8_t in6[1024] = {0x41, 0x1a, 0xf2, 0x3e, 0xc7, 0xc7, 0xdc};
    ALIGN(64) uint8_t in7[1024] = {0x3c, 0xd9, 0x5a, 0x70, 0x37, 0x7b, 0x3};

    ALIGN(64) uint8_t out[8][1024] = {0};

    ALIGN(64) uint8_t out0[1024] = {0};
    ALIGN(64) uint8_t out1[1024] = {0};
    ALIGN(64) uint8_t out2[1024] = {0};
    ALIGN(64) uint8_t out3[1024] = {0};
    ALIGN(64) uint8_t out4[1024] = {0};
    ALIGN(64) uint8_t out5[1024] = {0};
    ALIGN(64) uint8_t out6[1024] = {0};
    ALIGN(64) uint8_t out7[1024] = {0};

    ALIGN(64) uint8_t ou0[1024] = {0};
    ALIGN(64) uint8_t ou1[1024] = {0};
    ALIGN(64) uint8_t ou2[1024] = {0};
    ALIGN(64) uint8_t ou3[1024] = {0};
    ALIGN(64) uint8_t ou4[1024] = {0};
    ALIGN(64) uint8_t ou5[1024] = {0};
    ALIGN(64) uint8_t ou6[1024] = {0};
    ALIGN(64) uint8_t ou7[1024] = {0};

    shake128x8_absorb(&state1, in0, in1, in2, in3, in4, in5, in6, in7, 8);
    shake128x8_squeezeblocks(&state1, out0, out1, out2, out3, out4, out5, out6, out7, 1);

    for (int j = 0; j < 32; ++j) {
        printf("%#x ", out0[j]);
    }
    printf("\n");
    for (int j = 0; j < 32; ++j) {
        printf("%#x ", out1[j]);
    }
    printf("\n");
    for (int j = 0; j < 32; ++j) {
        printf("%#x ", out2[j]);
    }
    printf("\n");
    for (int j = 0; j < 32; ++j) {
        printf("%#x ", out3[j]);
    }
    printf("\n");

    shake128x4_absorb(&state2,in0,in1,in2,in3,8);
    shake128x4_squeezeblocks(ou0,ou1,ou2,ou3,1,&state2);
    printf("\n");

    for (int j = 0; j < 32; ++j) {
        printf("%#x ", ou0[j]);
    }
    printf("\n");
    for (int j = 0; j < 32; ++j) {
        printf("%#x ", ou1[j]);
    }
    printf("\n");
    for (int j = 0; j < 32; ++j) {
        printf("%#x ", ou2[j]);
    }
    printf("\n");
    for (int j = 0; j < 32; ++j) {
        printf("%#x ", ou3[j]);
    }
    printf("\n");
}


void test_shake128_speed() {
    keccakx8_state state1;

    ALIGN(64) uint8_t in0[1024] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe};
    ALIGN(64) uint8_t in1[1024] = {0x69, 0x16, 0x66, 0x6c, 0x25, 0x2f, 0x24, 0x92};
    ALIGN(64) uint8_t in2[1024] = {0x41, 0x1a, 0xf2, 0x3e, 0xc7, 0xc7, 0xdc};
    ALIGN(64) uint8_t in3[1024] = {0x3c, 0xd9, 0x5a, 0x70, 0x37, 0x7b, 0x3};
    ALIGN(64) uint8_t in4[1024] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe};
    ALIGN(64) uint8_t in5[1024] = {0x69, 0x16, 0x66, 0x6c, 0x25, 0x2f, 0x24, 0x92};
    ALIGN(64) uint8_t in6[1024] = {0x41, 0x1a, 0xf2, 0x3e, 0xc7, 0xc7, 0xdc};
    ALIGN(64) uint8_t in7[1024] = {0x3c, 0xd9, 0x5a, 0x70, 0x37, 0x7b, 0x3};

    uint64_t time = 0x200000;
    uint64_t Mbits = ((time >> 17) * 1088) ;
    clock_t start, end;
    start = clock();
    for (long i = 0; i < time; ++i) {
        // shake128x8_absorb(&state1, in0, in1, in2, in3, in4, in5, in6, in7, 8);
        // shake128x8_squeezeblocks(&state1,  in0, in1, in2, in3, in4, in5, in6, in7, 1);
                keccak8x_function(state1.s);
    }
    end = clock();
    double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    double speed_time = Mbits / elapsed;
    printf("\n----------test_shake128x8_speed Elapsed Timing is %f\n \n", elapsed);


    printf("\n%u", in0[0]);

}

void test_shake128_uniform_eta() {
    ALIGN(64) uint8_t rhoprime[1024];
    polyvecl mat[K], s1;
    polyveck s2;

    uint64_t nonce = 0;
    keccakx8_state state;


    uint64_t time = 0x200000;
    uint64_t Mbits = ((time >> 17) * 1088) ;
    clock_t start, end;
    start = clock();
    for (int i = 0; i < time; ++i) {
        poly_uniform_eta_8x(&state, &s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3],
                            &s2.vec[0], &s2.vec[1], &s2.vec[2], &s2.vec[3],rhoprime,
                            rhoprime[i%1024], 1, 2, 3, 4, 5, 6, 7);

    }
    end = clock();
    double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    double speed_time = Mbits / elapsed;
    printf("\n----------test_shake128_uniform_eta Elapsed Timing is %f\n", elapsed);


    printf("\n%u", s1.vec[0].coeffs[0]);
}

void test_shake128_uniform() {
    ALIGN(64) uint8_t rho[1024];
    polyvecl mat[K], s1;
    polyveck s2;

    uint64_t nonce = 0;
    keccakx8_state state;


    uint64_t time = 0x200000;
    uint64_t Mbits = ((time >> 17) * 1088) ;
    clock_t start, end;
    start = clock();
    for (int i = 0; i < time; ++i) {
        poly_uniform_8x(&state,
                        &mat[0].vec[0],
                        &mat[0].vec[1],
                        &mat[0].vec[2],
                        &mat[0].vec[3],
                        &mat[1].vec[0],
                        &mat[1].vec[1],
                        &mat[1].vec[2],
                        &mat[1].vec[3],rho,
                        rho[i%1024], 1, 2, 3,256, 257, 258, 259);
        poly_uniform_8x(&state,
                        &mat[2].vec[0],
                        &mat[2].vec[1],
                        &mat[2].vec[2],
                        &mat[2].vec[3],
                        &mat[3].vec[0],
                        &mat[3].vec[1],
                        &mat[3].vec[2],
                        &mat[3].vec[3],rho,
                        rho[i%1024], 513, 514, 515,768, 769, 770, 771);
    }
    end = clock();
    double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    double speed_time = Mbits / elapsed;
    printf("\n----------test_shake128_uniform Elapsed Timing is %f\n", elapsed);


    printf("\n%u", s1.vec[0].coeffs[0]);
}

void test_shake() {
    uint8_t seedbuf[1024];

    uint64_t time = 0x200000;
    uint64_t Mbits = ((time >> 17) * 1088) ;
    clock_t start, end;
    start = clock();
    for (int i = 0; i < time; ++i) {
        shake256(seedbuf, 3 * SEEDBYTES, seedbuf, SEEDBYTES);
    }
    end = clock();
    double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    double speed_time = Mbits / elapsed;
    printf("\n----------test_shake reference C Elapsed Timing is %f\n", elapsed);


    printf("%d",seedbuf[21]);
}


int main() {
    test_shake128_speed();
    // test_shake128_uniform_eta();
    // test_shake128_uniform();
    test_shake();
    return 0;
}
