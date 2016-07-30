
#include <stdio.h>
#include <intrin.h>
#include <immintrin.h>
#include <stdint.h>
#include <boost/preprocessor.hpp>

template<bool> struct Range;

// shift right

template<unsigned int N, typename = Range<true>>
struct mm256_shift_right_impl
{};

template<unsigned int N>
struct mm256_shift_right_impl<N, Range<N == 0>> {
    static __m256i doit(__m256i a) {
        return a;
    }
};

template<unsigned int N>
struct mm256_shift_right_impl<N, Range<(0 < N && N < 16)>> {
    static __m256i doit(__m256i a) {
        __m256i mask = _mm256_permute2x128_si256(a, a, 0x81);
        return _mm256_alignr_epi8(mask, a, N);
    }
};

template<unsigned int N>
struct mm256_shift_right_impl<N, Range<N == 16>> {
    static __m256i doit(__m256i a) {
        return _mm256_permute2x128_si256(a, a, 0x81);
    }
};

template<unsigned int N>
struct mm256_shift_right_impl<N, Range<(16 < N && N < 32)>> {
    static __m256i doit(__m256i a) {
        __m256i y1 = _mm256_srli_si256(a, N - 16);
        return _mm256_permute2x128_si256(y1, y1, 0x81);
    }
};

template<unsigned int N>
struct mm256_shift_right_impl<N, Range<N == 32>> {
    static __m256i doit(__m256i a) {
        return _mm256_setzero_si256();
    }
};

template <unsigned int N>
__m256i mm256_shift_right(__m256i a) {
    return mm256_shift_right_impl<N>::doit(a);
}

// shift left

template<unsigned int N, typename = Range<true>>
struct mm256_shift_left_impl
{};

template<unsigned int N>
struct mm256_shift_left_impl<N, Range<N == 0>> {
    static __m256i doit(__m256i a) {
        return a;
    }
};

template<unsigned int N>
struct mm256_shift_left_impl<N, Range<(0 < N && N < 16)>> {
    static __m256i doit(__m256i a) {
        __m256i mask = _mm256_permute2x128_si256(a, a, 0x08);
        return _mm256_alignr_epi8(a, mask, 16-N);
    }
};

template<unsigned int N>
struct mm256_shift_left_impl<N, Range<N == 16>> {
    static __m256i doit(__m256i a) {
        return _mm256_permute2x128_si256(a, a, 0x08);
    }
};

template<unsigned int N>
struct mm256_shift_left_impl<N, Range<(16 < N && N < 32)>> {
    static __m256i doit(__m256i a) {
        __m256i y1 = _mm256_slli_si256(a, N - 16);
        return _mm256_permute2x128_si256(y1, y1, 0x08);
    }
};

template<unsigned int N>
struct mm256_shift_left_impl<N, Range<N == 32>> {
    static __m256i doit(__m256i a) {
        return _mm256_setzero_si256();
    }
};

template <unsigned int N>
__m256i mm256_shift_left(__m256i a) {
    return mm256_shift_left_impl<N>::doit(a);
}

// funnel shift right

template<unsigned int N, typename = Range<true>>
struct mm256_funnel_shift_right_impl
{};

template<unsigned int N>
struct mm256_funnel_shift_right_impl<N, Range<N == 0>> {
    static __m256i doit(__m256i a, __m256i b) {
        return a;
    }
};

template<unsigned int N>
struct mm256_funnel_shift_right_impl<N, Range<(0 < N && N < 16)>> {
    static __m256i doit(__m256i a, __m256i b) {
        __m256i mask = _mm256_permute2x128_si256(a, b, 0x21);
        return _mm256_alignr_epi8(mask, a, N);
    }
};

template<unsigned int N>
struct mm256_funnel_shift_right_impl<N, Range<N == 16>> {
    static __m256i doit(__m256i a, __m256i b) {
        return _mm256_permute2x128_si256(a, b, 0x21);
    }
};

template<unsigned int N>
struct mm256_funnel_shift_right_impl<N, Range<(16 < N && N < 32)>> {
    static __m256i doit(__m256i a, __m256i b) {
        __m256i mask = _mm256_permute2x128_si256(a, b, 0x21);
        return _mm256_alignr_epi8(b, mask, N-16);
    }
};

template<unsigned int N>
struct mm256_funnel_shift_right_impl<N, Range<N == 32>> {
    static __m256i doit(__m256i a, __m256i b) {
        return b;
    }
};

template <unsigned int N>
inline __m256i mm256_funnel_shift_right(__m256i a, __m256i b) {
    return mm256_funnel_shift_right_impl<N>::doit(a, b);
}

// funnel shift left

template<unsigned int N, typename = Range<true>>
struct mm256_funnel_shift_left_impl
{};

template<unsigned int N>
struct mm256_funnel_shift_left_impl<N, Range<N == 0>> {
    static __m256i doit(__m256i a, __m256i b) {
        return a;
    }
};

template<unsigned int N>
struct mm256_funnel_shift_left_impl<N, Range<(0 < N && N < 16)>> {
    static __m256i doit(__m256i a, __m256i b) {
        __m256i mask = _mm256_permute2x128_si256(b, a, 0x21);
        return _mm256_alignr_epi8(a,mask,16-N);
    }
};

template<unsigned int N>
struct mm256_funnel_shift_left_impl<N, Range<N == 16>> {
    static __m256i doit(__m256i a, __m256i b) {
        return _mm256_permute2x128_si256(b, a, 0x21);
    }
};

template<unsigned int N>
struct mm256_funnel_shift_left_impl<N, Range<(16 < N && N < 32)>> {
    static __m256i doit(__m256i a, __m256i b) {
        __m256i mask = _mm256_permute2x128_si256(b, a, 0x21);
        return _mm256_alignr_epi8(mask,b,32-N);
    }
};

template<unsigned int N>
struct mm256_funnel_shift_left_impl<N, Range<N == 32>> {
    static __m256i doit(__m256i a, __m256i b) {
        return b;
    }
};

template <unsigned int N>
inline __m256i mm256_funnel_shift_left(__m256i a, __m256i b) {
    return mm256_funnel_shift_left_impl<N>::doit(a, b);
}

#ifdef _MSC_VER
#define MIE_ALIGN(x) __declspec(align(x))
#else
#define MIE_ALIGN(x) __attribute__((aligned(x)))
#endif

// original : http://homepage1.nifty.com/herumi/diary/1411.html

static const unsigned char MIE_ALIGN(16) shiftPattern[] = {
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
};

inline __m128i shift_bytes(__m128i v, int shift)
{
    __m128i mask = _mm_loadu_si128((const __m128i*)(shiftPattern + 16 + shift));
    return _mm_shuffle_epi8(v, mask);
}

inline __m128i funnel_shift_bytes(__m128i a, __m128i b, int shift)
{
    __m128i maskA = _mm_loadu_si128((const __m128i*)(shiftPattern + 16 + shift));
    __m128i maskB = _mm_loadu_si128((const __m128i*)(shiftPattern + shift));
    __m128i a0 = _mm_shuffle_epi8(a, maskA);
    __m128i b0 = _mm_shuffle_epi8(b, maskB);
    return _mm_or_si128(a0, b0);
}

inline __m256i shift_bytes(__m256i v, int shift)
{
    const __m128i* addr = (const __m128i*)(shiftPattern + 16 + shift);
    __m256i mask = _mm256_loadu2_m128i(addr, addr);
    return _mm256_shuffle_epi8(v, mask);
}

static
void print_m256i_bytes(__m256i bytes)
{
    for (int i = 31; i >= 0; --i) {
        printf("%2d ",((unsigned char *)&bytes)[i]);
    }
    printf("\n");
}

static
void print_m128i_bytes(__m128i bytes)
{
    for (int i = 15; i >= 0; --i) {
        printf("%2d ", ((unsigned char *)&bytes)[i]);
    }
    printf("\n");
}

void test()
{
    __m256i reg =  _mm256_set_epi8(
        32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,
        16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1
    );
    
    printf("shift right\n");
    #define GEN_CALL(z, i) printf("[%2d] ", i); print_m256i_bytes(mm256_shift_right<i>(reg));
    BOOST_PP_REPEAT(33,GEN_CALL)
    #undef GEN_CALL
    printf("\n");

    printf("shift left\n");
    #define GEN_CALL(z, i) printf("[%2d] ", i); print_m256i_bytes(mm256_shift_left<i>(reg));
    BOOST_PP_REPEAT(33,GEN_CALL)
    #undef GEN_CALL
    printf("\n");
}

void test2()
{
    __m256i a, b;
    a = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
        );
    b = _mm256_setr_epi8(
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63
        );

    printf("funnel shift right\n");
    #define GEN_CALL(z, i) printf("[%2d] ", i); print_m256i_bytes(mm256_funnel_shift_right<i>(a, b));
    BOOST_PP_REPEAT(33,GEN_CALL)
    #undef GEN_CALL
    printf("\n");

    printf("funnel shift left\n");
    #define GEN_CALL(z, i) printf("[%2d] ", i); print_m256i_bytes(mm256_funnel_shift_left<i>(b, a));
    BOOST_PP_REPEAT(33,GEN_CALL)
    #undef GEN_CALL
    printf("\n");
}

int main(int argc, char* argv[])
{
    test();
    test2();

#if 0
    {
        __m128i a, b;
        a = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15
            );
        b = _mm_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
            );
        //for (int i = -16; i <= 16; ++i) {
        //  __m128i c = shift_bytes(a, i);
        //  print_m128i_bytes(c);
        //}
        //printf("\n");
        for (int i = -16; i <= 16; ++i) {
            __m128i c = funnel_shift_bytes(a, b, i);
            print_m128i_bytes(c);
        }
    }
#endif

    //printf("\n");
    //{
    //  __m256i a;
    //  a = _mm256_setr_epi8(
    //      0, 1, 2, 3, 4, 5, 6, 7,
    //      8, 9, 10, 11, 12, 13, 14, 15,
    //      16, 17, 18, 19, 20, 21, 22, 23,
    //      24, 25, 26, 27, 28, 29, 30, 31
    //      );
    //  for (int i = -32; i <= 32; ++i) {
    //      __m256i c = shift_bytes(a, i);
    //      print_mm256_bytes(c);
    //  }
    //}

    //extern void test_gather();
    //test_gather();
    return 0;
}

