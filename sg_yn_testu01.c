/*
 * ============================================================
 * 2D Schaffer-Gompertz (2D-SG) Chaotic Map
 * TestU01 Wrapper — YN Sequence
 * ============================================================
 *
 * This file implements the 2D-SG chaotic map and feeds the
 * yn sequence to the TestU01 statistical test suite.
 *
 * MAP EQUATIONS:
 *   xn+1 = (4*pi*S(xn,yn) + G(xn,c1) + tanh(pi*(yn-0.5))) mod 1
 *   yn+1 = (4*pi*S(yn,xn) + G(yn,c2) + tanh(pi*(xn-0.5))) mod 1
 *
 * where:
 *   S(x,y) = Schaffer F6 function
 *   G(u,c) = Gompertz function = c * u * ln(1/u)
 *
 * COMPILE:
 *   gcc -O3 -o sg_yn_testu01 sg_yn_testu01.c \
 *       -lm -ltestu01 -lprobdist -lmylib
 *
 * USAGE:
 *   ./sg_yn_testu01 smallcrush 1e13
 *   ./sg_yn_testu01 crush      1e13
 *   ./sg_yn_testu01 bigcrush   1e13
 *
 * PARAMETERS (fixed):
 *   c1 = 3.7,  c2 = 6.3
 *   x0 = 0.123456789012345
 *   y0 = 0.987654321098765
 *   Warmup = 10000 iterations
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "unif01.h"
#include "bbattery.h"

/* ============================================================
 * SCHAFFER F6 FUNCTION
 *   S(x,y) = 0.5 + (sin^2(sqrt(x^2+y^2)) - 0.5)
 *                  / (1 + 0.001*(x^2+y^2))^2
 * Output range: [0, 1]
 * ============================================================ */
static double schaffer_f6(double x, double y) {
    double r2  = x*x + y*y;
    double num = sin(sqrt(r2));
    num = num*num - 0.5;
    double den = 1.0 + 0.001*r2;
    return 0.5 + num / (den*den);
}

/* ============================================================
 * GOMPERTZ FUNCTION
 *   G(u,c) = c * u * ln(1/u),  c in [2,10], u in (0,1)
 * ============================================================ */
static double gompertz(double u, double c) {
    if (u <= 1e-15)     u = 1e-15;
    if (u >= 1.0-1e-15) u = 1.0-1e-15;
    return c * u * log(1.0/u);
}

/* ============================================================
 * GLOBAL MAP STATE
 * ============================================================ */
static double gx, gy;
static double gc1 = 3.7, gc2 = 6.3;

/* Bit limit control */
static unsigned long long g_bit_limit = 0;
static unsigned long long g_bits_used = 0;

/* ============================================================
 * ONE MAP ITERATION
 *   Updates global state (gx, gy) -> (gx_new, gy_new)
 * ============================================================ */
static void map_iterate(void) {
    double x = gx, y = gy;

    double xn = 4.0*M_PI*schaffer_f6(x,y)
              + gompertz(x, gc1)
              + tanh(M_PI*(y-0.5));

    double yn = 4.0*M_PI*schaffer_f6(y,x)
              + gompertz(y, gc2)
              + tanh(M_PI*(x-0.5));

    /* mod 1 — keep in (0,1) */
    xn -= floor(xn); if(xn<=0) xn=1e-15; if(xn>=1) xn=1.0-1e-15;
    yn -= floor(yn); if(yn<=0) yn=1e-15; if(yn>=1) yn=1.0-1e-15;

    gx = xn;
    gy = yn;
}

/* ============================================================
 * STAFFORD 64-BIT AVALANCHE FINALIZER
 *   Eliminates residual linear correlation in raw bits.
 *   Constants: C1 = 0xbf58476d1ce4e5b9
 *              C2 = 0x94d049bb133111eb
 * ============================================================ */
static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

/* ============================================================
 * CROSS-DIMENSIONAL BIT INTERLEAVING
 *   Combines bits from two 64-bit words (x and y) to ensure
 *   full cross-coupling between both state variables.
 * ============================================================ */
static uint64_t interleave64(uint64_t x, uint64_t y) {
    uint64_t a = x ^ (y<<32) ^ (y>>32);
    uint64_t b = y ^ (x<<17) ^ (x>>47);
    return a ^ b ^ (a>>13) ^ (b<<41);
}

/* ============================================================
 * YN SEQUENCE — 32-BIT OUTPUT WORD
 *
 * Two consecutive map iterations are used per output word:
 *   Iter 1: ux = floor(xn+1 * (2^64-1))
 *           uy = floor(yn+1 * (2^64-1))
 *   Iter 2: vx = floor(xn+2 * (2^64-1))
 *           vy = floor(yn+2 * (2^64-1))
 *   w  = interleave64(uy XOR vx, ux XOR vy)
 *   w  = mix64(w)
 *   r  = upper 32 bits of w
 * ============================================================ */
static unsigned int yn_next_u32(void) {

    if (g_bit_limit > 0 && g_bits_used >= g_bit_limit)
        return 0;

    /* Iteration 1 */
    map_iterate();
    uint64_t ux = (uint64_t)(gx * (double)UINT64_MAX);
    uint64_t uy = (uint64_t)(gy * (double)UINT64_MAX);

    /* Iteration 2 */
    map_iterate();
    uint64_t vx = (uint64_t)(gx * (double)UINT64_MAX);
    uint64_t vy = (uint64_t)(gy * (double)UINT64_MAX);

    /* Cross-combine yn and xn values */
    uint64_t w = interleave64(uy ^ vx, ux ^ vy);

    /* Apply avalanche finalizer */
    w = mix64(w);

    g_bits_used += 32;
    return (unsigned int)(w >> 32);
}

/* ============================================================
 * MAIN
 * ============================================================ */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <test> <bits>\n"
            "  test : smallcrush | crush | bigcrush\n"
            "  bits : 1e13 | 1e14 | 1e15 | 1e16\n\n"
            "Examples:\n"
            "  ./sg_yn_testu01 smallcrush 1e13\n"
            "  ./sg_yn_testu01 crush      1e14\n"
            "  ./sg_yn_testu01 bigcrush   1e15\n",
            argv[0]);
        return 1;
    }

    const char *test = argv[1];
    double bits_d    = atof(argv[2]);
    g_bit_limit      = (unsigned long long)bits_d;

    /* Fixed initial conditions */
    gx  = 0.123456789012345;
    gy  = 0.987654321098765;
    gc1 = 3.7;
    gc2 = 6.3;
    g_bits_used = 0;

    /* Discard first 10000 iterations (warmup) */
    for (int i = 0; i < 10000; i++) map_iterate();

    printf("=== 2D-SG Chaotic Map — YN Sequence ===\n");
    printf("Test      : %s\n", test);
    printf("Bit limit : %.2e\n", bits_d);
    printf("c1=%.1f  c2=%.1f\n", gc1, gc2);
    printf("x0=%.15f\n", gx);
    printf("y0=%.15f\n\n", gy);

    /* Register with TestU01 */
    unif01_Gen *gen = unif01_CreateExternGenBits(
                        "2D-SG-YN", yn_next_u32);

    /* Run selected battery */
    if      (strcmp(test,"smallcrush") == 0) bbattery_SmallCrush(gen);
    else if (strcmp(test,"crush")      == 0) bbattery_Crush(gen);
    else if (strcmp(test,"bigcrush")   == 0) bbattery_BigCrush(gen);
    else {
        fprintf(stderr, "Unknown test: %s\n", test);
        unif01_DeleteExternGen01(gen);
        return 1;
    }

    printf("\nBits used : %.2e / %.2e\n",
           (double)g_bits_used, bits_d);

    unif01_DeleteExternGen01(gen);
    return 0;
}
