
#include <stdio.h>
#include <stdint.h>
#include <intrin.h>
#include <boost/preprocessor.hpp>

#include "avx_shift.h"

inline static void lf() { printf("\n"); }
inline static void idx(int i) { printf("%3d: ", i); }

static
void print_m256_floats(__m256 floats)
{
    for (int i = 7; i >= 0; --i) {
        printf("%f ", ((float *)&floats)[i]);
    }
    lf();
}

static
void print_m128i_bytes(__m128i bytes)
{
    for (int i = 15; i >= 0; --i) {
        printf("%2d ", ((unsigned char *)&bytes)[i]);
    }
    lf();
}

static
void print_m256i_bytes(__m256i bytes)
{
    for (int i = 31; i >= 0; --i) {
        printf("%2d ",((unsigned char *)&bytes)[i]);
    }
    lf();
}

// test __m256 constant shift
static
void test0()
{
    const __m256 reg =  _mm256_set_ps(
        8,7,6,5,4,3,2,1
    );
    printf("m256_shift_right\n");
    #define GEN_CALL(z, i, j) idx(i); print_m256_floats(m256_shift_right<i*4>(reg));
    BOOST_PP_REPEAT(9, GEN_CALL, 0)
    #undef GEN_CALL
    lf();

    printf("m256_shift_left\n");
    #define GEN_CALL(z, i, j) idx(i); print_m256_floats(m256_shift_left<i*4>(reg));
    BOOST_PP_REPEAT(9, GEN_CALL, 0)
    #undef GEN_CALL
    lf();

    lf();
}

// test __m256i constant shift
static
void test1()
{
    const __m256i a = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
        );
    printf("m256i_shift_right\n");
    #define GEN_CALL(z, i, j) idx(i); print_m256i_bytes(m256i_shift_right<i>(a));
    BOOST_PP_REPEAT(33, GEN_CALL, 0)
    #undef GEN_CALL
    lf();

    printf("m256i_shift_left\n");
    #define GEN_CALL(z, i, j) idx(i); print_m256i_bytes(m256i_shift_left<i>(a));
    BOOST_PP_REPEAT(33, GEN_CALL, 0)
    #undef GEN_CALL
    lf();

    lf();
}

// test __m256i constant funnel shift
static
void test2()
{
    const __m256i a = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
        );
    const __m256i b = _mm256_setr_epi8(
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63
        );

    printf("m256i_funnel_shift_right\n");
    #define GEN_CALL(z, i, j) idx(i); print_m256i_bytes(m256i_funnel_shift_right<i>(a, b));
    BOOST_PP_REPEAT(33, GEN_CALL, 0)
    #undef GEN_CALL
    lf();

    printf("m256i_funnel_shift_left\n");
    #define GEN_CALL(z, i, j) idx(i); print_m256i_bytes(m256i_funnel_shift_left<i>(b, a));
    BOOST_PP_REPEAT(33, GEN_CALL, 0)
    #undef GEN_CALL
    lf();

    lf();
}

// test __m128i variable shift
static
void test3()
{
    const __m128i a = _mm_setr_epi8(
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16
        );
    const __m128i b = _mm_setr_epi8(
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32
        );
    
    printf("m128i_shift\n");
    for (int i = -16; i <= 16; ++i) {
        __m128i c = m128i_shift(a, i);
        idx(i);
        print_m128i_bytes(c);
    }
    lf();

    printf("m128i_funnel_shift_right\n");
    for (int i = 0; i <= 16; ++i) {
        __m128i c = m128i_funnel_shift_right(a, b, i);
        idx(i);
        print_m128i_bytes(c);
    }
    lf();

    printf("m128i_funnel_shift_left\n");
    for (int i = 0; i <= 16; ++i) {
        __m128i c = m128i_funnel_shift_left(b, a, i);
        idx(i);
        print_m128i_bytes(c);
    }
    lf();

    lf();
}

// test __m256i variable shift
static
void test4()
{
    const __m256i a = _mm256_setr_epi8(
         1,  2,  3,  4,  5,  6,  7,  8,
         9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32
        );
    const __m256i b = _mm256_setr_epi8(
        33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64
        );

    printf("m256i_shift_right\n");
    for (int i = 0; i <= 32; ++i) {
        __m256i c = m256i_shift_right(a, i);
        idx(i);
        print_m256i_bytes(c);
    }
    lf();

    printf("m256i_shift_left\n");
    for (int i = 0; i <= 32; ++i) {
        __m256i c = m256i_shift_left(a, i);
        idx(i);
        print_m256i_bytes(c);
    }
    lf();

    printf("m256i_funnel_shift_right\n");
    for (int i = 0; i <= 32; ++i) {
        __m256i c = m256i_funnel_shift_right(a, b, i);
        idx(i);
        print_m256i_bytes(c);
    }
    lf();

    printf("m256i_funnel_shift_left\n");
    for (int i = 0; i <= 32; ++i) {
        __m256i c = m256i_funnel_shift_left(b, a, i);
        idx(i);
        print_m256i_bytes(c);
    }
    lf();

    lf();
}

int main(int argc, char* argv[])
{
    printf("constant shift routines\n");
    lf();
    test0();
    test1();
    test2();

    printf("variable shift routines\n");
    lf();
    test3();
    test4();

    return 0;
}

