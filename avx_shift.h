
#include <intrin.h>
#include <type_traits>
#include <cassert>

#ifdef _MSC_VER
#define MIE_ALIGN(x) __declspec(align(x))
#else
#define MIE_ALIGN(x) __attribute__((aligned(x)))
#endif

template<bool> struct Range;

template<int n>
struct foobar : std::false_type
{ };

// constant shift amount

// AVX
// Byte Shift YMM Register Across 128-bit Lanes
// limitation : shift amount is immediate and is multiples of 4

template <int n>
inline __m256 m256_shift_left(__m256 a) {
    static_assert(foobar<n>::value, "unsupported shift amount");
    return a;
}

template <>
inline __m256 m256_shift_left<0>(__m256 x) {
    return x;
}

// http://stackoverflow.com/q/19516585
template <>
inline __m256 m256_shift_left<4>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x6, x5, x4, x7, x2, x1, x0, x3)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
    // t1 = (x2, x1, x0, x3, 0, 0, 0, 0)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
    // y  = (x6, x5, x4, x3, x2, x1, x0, 0)
    __m256 y = _mm256_blend_ps(t0, t1, 0x11);
    return y;
}

// http://stackoverflow.com/q/19516585
template <>
inline __m256 m256_shift_left<8>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x5, x4, x7, x6, x1, x0, x3, x2)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
    // t1 = (x1, x0, x3, x2, 0, 0, 0, 0)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
    // y  = (x5, x4, x3, x2, x1, x0, 0, 0)
    __m256 y = _mm256_blend_ps(t0, t1, 0x33 /* 0b00110011 */ );
    return y;
}

template <>
inline __m256 m256_shift_left<12>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x4, x7, x6, x5, x0, x3, x2, x1)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(0, 3, 2, 1));
    // t1 = (x0, x3, x2, x1, 0, 0, 0, 0)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
    // y  = (x4, x3, x2, x1, x0, 0, 0, 0)
    __m256 y = _mm256_blend_ps(t0, t1, 0x77 /* 0b01110111 */ );
    return y;
}

template <>
inline __m256 m256_shift_left<16>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // y  = (x3, x2, x1, x0, 0, 0, 0, 0)
    __m256 y = _mm256_permute2f128_ps(x, x, 0x08);
    return y;
}

template <>
inline __m256 m256_shift_left<20>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x6, x5, x4, x7, x2, x1, x0, x3)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
    // t1 = (x2, x1, x0, x3, 0, 0, 0, 0)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
    // y  = (x2, x1, x0, 0, 0, 0, 0, 0)
    __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0x10 /* 0b00010000 */ );
    return y;
}

template <>
inline __m256 m256_shift_left<24>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x5, x4, x7, x6, x1, x0, x3, x2)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
    // t1 = (x1, x0, x3, x2, 0, 0, 0, 0)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
    // y  = (x1, x0, 0, 0, 0, 0, 0, 0)
    __m256 y = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xC0 /* 0b11000000 */ );
    return y;
}

template <>
inline __m256 m256_shift_left<28>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x4, x7, x6, x5, x0, x3, x2, x1)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(0, 3, 2, 1));
    // t1 = (x0, x3, x2, x1, 0, 0, 0, 0)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
    // y  = (x0, 0, 0, 0, 0, 0, 0, 0)
    __m256 y = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0x80 /* 0b10000000 */ );
    return y;
}

template <>
inline __m256 m256_shift_left<32>(__m256 x) {
    return _mm256_setzero_ps();
}

template <int n>
inline __m256 m256_shift_right(__m256 a)
{
    static_assert(foobar<n>::value, "unsupported shift amount");
    return a;
}

template <>
inline __m256 m256_shift_right<0>(__m256 x) {
    return x;
}

// http://stackoverflow.com/a/19532415/4699324
template <>
inline __m256 m256_shift_right<4>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x4, x7, x6, x5, x0, x3, x2, x1)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(0, 3, 2, 1));
    // t1 = (0, 0, 0, 0, x4, x7, x6, x5)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

    //      ( -, x7, x6, x5,  -, x3, x2, x1)
    //      ( 0,  -,  -,  -, x4,  -,  -,  -)
    // y  = ( 0, x7, x6, x5, x4, x3, x2, x1)
    __m256 y = _mm256_blend_ps(t0, t1, 0x88 /* 0b10001000 */ );
    return y;
}

template <>
inline __m256 m256_shift_right<8>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x5, x4, x7, x6, x1, x0, x3, x2)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
    // t1 = (0, 0, 0, 0, x5, x4, x7, x6)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

    //      ( -,  -, x7, x6,  -,  -, x3, x2)
    //      ( 0,  0,  -,  -, x5, x4,  -,  -)
    // y  = ( 0,  0, x7, x6, x5, x4, x3, x2)
    __m256 y = _mm256_blend_ps(t0, t1, 0xCC /* 0b11001100 */ );
    return y;
}

template <>
inline __m256 m256_shift_right<12>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x6, x5, x4, x7, x2, x1, x0, x3)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
    // t1 = ( 0,  0,  0,  0, x6, x5, x4, x7)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

    //      ( -,  -,  -, x7,  -,  -,  -, x3)
    //      ( 0,  0,  0,  -, x6, x5, x4,  -)
    // y  = ( 0,  0,  0, x7, x6, x5, x4, x3)
    __m256 y = _mm256_blend_ps(t0, t1, 0xEE /* 0b11101110 */ );
    return y;
}

template <>
inline __m256 m256_shift_right<16>(__m256 x)
{
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // y  = ( 0,  0,  0,  0, x7, x6, x5, x4)
    __m256 y = _mm256_permute2f128_ps(x, x, 0x81);
    return y;
}

template <>
inline __m256 m256_shift_right<20>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x4, x7, x6, x5, x0, x3, x2, x1)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(0, 3, 2, 1));
    // t1 = ( 0,  0,  0,  0, x4, x7, x6, x5)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

    //      ( -,  -,  -,  -,  -, x7, x6, x5)
    //      ( 0,  0,  0,  0,  0,  -,  -,  -)
    // y  = ( 0,  0,  0,  0,  0, x7, x6, x5)
    __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0xF8 /* 0b11111000 */ );
    return y;
}

template <>
inline __m256 m256_shift_right<24>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x5, x4, x7, x6, x1, x0, x3, x2)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
    // t1 = ( 0,  0,  0,  0, x5, x4, x7, x6)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

    //      ( -,  -,  -,  -,  -,  -, x7, x6)
    //      ( 0,  0,  0,  0,  0,  0,  -,  -)
    // y  = ( 0,  0,  0,  0,  0,  0, x7, x6)
    __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0xFC /* 0b11111100 */ );
    return y;
}

template <>
inline __m256 m256_shift_right<28>(__m256 x) {
    // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

    // t0 = (x6, x5, x4, x7, x2, x1, x0, x3)
    __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
    // t1 = ( 0,  0,  0,  0, x6, x5, x4, x7)
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

    //      ( -,  -,  -,  -,  -,  -,  -, x7)
    //      ( 0,  0,  0,  0,  0,  0,  0,  -)
    // y  = ( 0,  0,  0,  0,  0,  0,  0, x7)
    __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0xFE /* 0b11111110 */ );
    return y;
}

template <>
inline __m256 m256_shift_right<32>(__m256 x) {
    return _mm256_setzero_ps();
}

// AVX2
// Byte Shift YMM Register Across 128-bit Lanes
// limitation : shift amount is immediate

namespace {

// shift right impl

template<unsigned int N, typename = Range<true>>
struct m256i_shift_right_impl
{};

template<unsigned int N>
struct m256i_shift_right_impl<N, Range<N == 0>> {
    static __m256i doit(__m256i a) {
        return a;
    }
};

template<unsigned int N>
struct m256i_shift_right_impl<N, Range<(0 < N && N < 16)>> {
    static __m256i doit(__m256i a) {
        __m256i mask = _mm256_permute2x128_si256(a, a, 0x81);
        return _mm256_alignr_epi8(mask, a, N);
    }
};

template<unsigned int N>
struct m256i_shift_right_impl<N, Range<N == 16>> {
    static __m256i doit(__m256i a) {
        return _mm256_permute2x128_si256(a, a, 0x81);
    }
};

template<unsigned int N>
struct m256i_shift_right_impl<N, Range<(16 < N && N < 32)>> {
    static __m256i doit(__m256i a) {
        __m256i tmp = _mm256_srli_si256(a, N - 16);
        return _mm256_permute2x128_si256(tmp, tmp, 0x81);
    }
};

template<unsigned int N>
struct m256i_shift_right_impl<N, Range<N >= 32>> {
    static __m256i doit(__m256i a) {
        return _mm256_setzero_si256();
    }
};

// shift left impl

template<unsigned int N, typename = Range<true>>
struct m256i_shift_left_impl
{};

template<unsigned int N>
struct m256i_shift_left_impl<N, Range<N == 0>> {
    static __m256i doit(__m256i a) {
        return a;
    }
};

template<unsigned int N>
struct m256i_shift_left_impl<N, Range<(0 < N && N < 16)>> {
    static __m256i doit(__m256i a) {
        __m256i mask = _mm256_permute2x128_si256(a, a, 0x08);
        return _mm256_alignr_epi8(a, mask, 16-N);
    }
};

template<unsigned int N>
struct m256i_shift_left_impl<N, Range<N == 16>> {
    static __m256i doit(__m256i a) {
        return _mm256_permute2x128_si256(a, a, 0x08);
    }
};

template<unsigned int N>
struct m256i_shift_left_impl<N, Range<(16 < N && N < 32)>> {
    static __m256i doit(__m256i a) {
        __m256i tmp = _mm256_slli_si256(a, N - 16);
        return _mm256_permute2x128_si256(tmp, tmp, 0x08);
    }
};

template<unsigned int N>
struct m256i_shift_left_impl<N, Range<N >= 32>> {
    static __m256i doit(__m256i a) {
        return _mm256_setzero_si256();
    }
};

// funnel shift right impl

template<unsigned int N, typename = Range<true>>
struct m256i_funnel_shift_right_impl
{};

template<unsigned int N>
struct m256i_funnel_shift_right_impl<N, Range<N == 0>> {
    static __m256i doit(__m256i a, __m256i b) {
        return a;
    }
};

template<unsigned int N>
struct m256i_funnel_shift_right_impl<N, Range<(0 < N && N < 16)>> {
    static __m256i doit(__m256i a, __m256i b) {
        __m256i mask = _mm256_permute2x128_si256(a, b, 0x21);
        return _mm256_alignr_epi8(mask, a, N);
    }
};

template<unsigned int N>
struct m256i_funnel_shift_right_impl<N, Range<N == 16>> {
    static __m256i doit(__m256i a, __m256i b) {
        return _mm256_permute2x128_si256(a, b, 0x21);
    }
};

template<unsigned int N>
struct m256i_funnel_shift_right_impl<N, Range<(16 < N && N < 32)>> {
    static __m256i doit(__m256i a, __m256i b) {
        __m256i mask = _mm256_permute2x128_si256(a, b, 0x21);
        return _mm256_alignr_epi8(b, mask, N-16);
    }
};

template<unsigned int N>
struct m256i_funnel_shift_right_impl<N, Range<N == 32>> {
    static __m256i doit(__m256i a, __m256i b) {
        return b;
    }
};

// funnel shift left impl

template<unsigned int N, typename = Range<true>>
struct m256i_funnel_shift_left_impl
{};

template<unsigned int N>
struct m256i_funnel_shift_left_impl<N, Range<N == 0>> {
    static __m256i doit(__m256i a, __m256i b) {
        return a;
    }
};

template<unsigned int N>
struct m256i_funnel_shift_left_impl<N, Range<(0 < N && N < 16)>> {
    static __m256i doit(__m256i a, __m256i b) {
        __m256i mask = _mm256_permute2x128_si256(b, a, 0x21);
        return _mm256_alignr_epi8(a,mask,16-N);
    }
};

template<unsigned int N>
struct m256i_funnel_shift_left_impl<N, Range<N == 16>> {
    static __m256i doit(__m256i a, __m256i b) {
        return _mm256_permute2x128_si256(b, a, 0x21);
    }
};

template<unsigned int N>
struct m256i_funnel_shift_left_impl<N, Range<(16 < N && N < 32)>> {
    static __m256i doit(__m256i a, __m256i b) {
        __m256i mask = _mm256_permute2x128_si256(b, a, 0x21);
        return _mm256_alignr_epi8(mask,b,32-N);
    }
};

template<unsigned int N>
struct m256i_funnel_shift_left_impl<N, Range<N == 32>> {
    static __m256i doit(__m256i a, __m256i b) {
        return b;
    }
};

} // anonymous namespace

// interface

template <unsigned int N>
__m256i m256i_shift_right(__m256i a) {
    return m256i_shift_right_impl<N>::doit(a);
}

template <unsigned int N>
__m256i m256i_shift_left(__m256i a) {
    return m256i_shift_left_impl<N>::doit(a);
}

template <unsigned int N>
inline __m256i m256i_funnel_shift_right(__m256i a, __m256i b) {
    return m256i_funnel_shift_right_impl<N>::doit(a, b);
}

template <unsigned int N>
inline __m256i m256i_funnel_shift_left(__m256i a, __m256i b) {
    return m256i_funnel_shift_left_impl<N>::doit(a, b);
}

// variable shift amount

static const unsigned char MIE_ALIGN(16) shiftPattern[] = {
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
};

// original : http://homepage1.nifty.com/herumi/diary/1411.html
inline __m128i m128i_shift(__m128i v, int shift)
{
    assert(-16 <= shift && shift <= 16);
    return _mm_shuffle_epi8(v, _mm_loadu_si128((const __m128i*)(shiftPattern + 32 + shift)));
}

inline __m128i m128i_funnel_shift_right(__m128i a, __m128i b, int shift)
{
    assert(0 <= shift && shift <= 16);
    __m128i b0 = _mm_shuffle_epi8(b, _mm_loadu_si128((const __m128i*)(shiftPattern + 16 +shift)));
    __m128i a0 = _mm_shuffle_epi8(a, _mm_loadu_si128((const __m128i*)(shiftPattern + 32 + shift)));
    return _mm_or_si128(a0, b0);
}

inline __m128i m128i_funnel_shift_left(__m128i a, __m128i b, int shift)
{
    assert(0 <= shift && shift <= 16);
    __m128i a0 = _mm_shuffle_epi8(a, _mm_loadu_si128((const __m128i*)(shiftPattern + 32 - shift)));
    __m128i b0 = _mm_shuffle_epi8(b, _mm_loadu_si128((const __m128i*)(shiftPattern + 48 - shift)));
    return _mm_or_si128(a0, b0);
}

inline __m256i m256i_shift_right(__m256i v, int shift)
{
    assert(0 <= shift && shift <= 32);
    __m256i mask = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 32 + shift)));
    __m256i a0 = _mm256_shuffle_epi8(v, mask);

    __m256i a1 = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm256_extracti128_si256(v, 1), _mm_loadu_si128((const __m128i*)(shiftPattern + 16 + shift))));
    __m256i a2 = _mm256_permute2x128_si256(a1, a1, 0x80);

    return _mm256_or_si256(a0, a2);
}

inline __m256i m256i_shift_left(__m256i v, int shift)
{
    assert(0 <= shift && shift <= 32);
    __m256i mask = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 32 - shift)));
    __m256i a0 = _mm256_shuffle_epi8(v, mask);

    __m256i a1 = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm256_castsi256_si128(v), _mm_loadu_si128((const __m128i*)(shiftPattern + 48 - shift))));
    __m256i a2 = _mm256_permute2x128_si256(a1, a1, 0x08);

    return _mm256_or_si256(a0, a2);
}

inline __m256i m256i_funnel_shift_right(__m256i a, __m256i b, int shift) {
    assert(0 <= shift && shift <= 32);

    __m256i ma = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 32 + shift)));
    __m256i a0 = _mm256_shuffle_epi8(a, ma);

    __m256i mb = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + shift)));
    __m256i b0 = _mm256_shuffle_epi8(b, mb);
    
    const __m256i pat = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 16 + shift)));
    __m256i ab = _mm256_permute2x128_si256(a, b, 0x21);
    __m256i ab0 = _mm256_shuffle_epi8(ab, pat);

    return _mm256_or_si256(ab0, _mm256_or_si256(a0, b0));
}

inline __m256i m256i_funnel_shift_left(__m256i a, __m256i b, int shift) {
    assert(0 <= shift && shift <= 32);

    __m256i mb = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 64 - shift)));
    __m256i b0 = _mm256_shuffle_epi8(b, mb);

    __m256i ma = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 32 - shift)));
    __m256i a0 = _mm256_shuffle_epi8(a, ma);
    
    const __m256i pat = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 48 - shift)));
    __m256i ab = _mm256_permute2x128_si256(a, b, 0x03);
    __m256i ab0 = _mm256_shuffle_epi8(ab, pat);

    return _mm256_or_si256(ab0, _mm256_or_si256(a0, b0));
}

