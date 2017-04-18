

//#include <intrin.h>
#include <x86intrin.h>

// in  : ( x3, x2, x1, x0 )
// out : (  -,  -,  -, x3+x2+x1+x0 )
inline __m128 hsum128_ps_naive(__m128 x)
{
    // a = ( x3+x2, x1+x0, x3+x2, x1+x0 )
    __m128 a = _mm_hadd_ps(x, x);
    // a = ( x3+x2+x1+x0, x3+x2+x1+x0, x3+x2+x1+x0, x3+x2+x1+x0 )
    a = _mm_hadd_ps(a, a);
    return a;
}

// Horizontal sum routines
// '-' lanes of returned value are not meant to be used by caller
// (      lane4,     lane3,     lane2,    lane1 )
// ( bits96-127, bits64-95, bits32-63, bits0-31 )

// in  : ( x3, x2, x1, x0 )
// out : (  -,  -,  -, x3+x2+x1+x0 )
inline __m128 hsum128_ps(__m128 x)
{
    // loDual = ( -, -, x1, x0 )
    const __m128 loDual = x;
    // hiDual = ( -, -, x3, x2 )
    const __m128 hiDual = _mm_movehl_ps(x, x);
    // sumDual = ( -, -, x1+x3, x0+x2 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0+x2 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1+x3 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0+x1+x2+x3 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return sum;
}

// in  : ( x7, x6, x5, x4, x3, x2, x1, x0 )
// out : (  -,  -,  -,  -,  -,  -,  -, x7+x6+x5+x4+x3+x2+x1+x0 )
inline __m128 hsum256_ps(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3+x7, x2+x6, x1+x5, x0+x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1+x5, x0+x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3+x7, x2+x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1+x3 + x5+x7, x0+x2 + x4+x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0+x2 + x4+x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1+x3 + x5+x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0+x1+x2+x3 + x4+x5+x6+x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return sum;
}

// Horizontally add elements of each __m256 type arguments at once
// in a : ( a7, a6, a5, a4, a3, a2, a1, a0 )
// in b : ( b7, b6, b5, b4, b3, b2, b1, b0 )
// out  : (  -,  -,  -,  -,  -,  -, b1+b5+b3+b7+b0+b4+b2+b6, a1+a5+a3+a7+a0+a4+a2+a6 )
inline __m128 hsum2x256_ps(__m256 a, __m256 b) {
    // (b3, b2, b1, b0, a3, a2, a1, a0)
    __m256 x = _mm256_permute2f128_ps(a, b, 0x20);
    // (b7, b6, b5, b4, a7, a6, a5, a4)
    __m256 y = _mm256_permute2f128_ps(a, b, 0x31);
    // (b3+b7, b2+b6, b1+b5, b0+b4, a3+a7, a2+a6, a1+a5, a0+a4)
    x = _mm256_add_ps(x, y);
    // (-, -, b3+b7, b2+b6, -, -, a3+a7, a2+a6)
    y = _mm256_permute_ps(x, _MM_SHUFFLE(3, 2, 3, 2));
    // (-, -, b1+b5+b3+b7, b0+b4+b2+b6, -, -, a1+a5+a3+a7, a0+a4+a2+a6)
    x = _mm256_add_ps(x, y);
    // (-, -, -, b1+b5+b3+b7, -, -, -, a1+a5+a3+a7)
    y = _mm256_permute_ps(x, _MM_SHUFFLE(1, 1, 1, 1));
    // (-, -, -, b1+b5+b3+b7+b0+b4+b2+b6, -, -, -, a1+a5+a3+a7+a0+a4+a2+a6)
    x = _mm256_add_ps(x, y);
    // (-, -, -, b1+b5+b3+b7+b0+b4+b2+b6)
    __m128 upper = _mm256_extractf128_ps(x, 1);
    // (-, -, -, -, -, -, b1+b5+b3+b7+b0+b4+b2+b6, a1+a5+a3+a7+a0+a4+a2+a6)
    __m128 ret = _mm_unpacklo_ps(_mm256_castps256_ps128(x), upper);
    return ret;
}

// Horizontally add elements of each __m256 type arguments at once
// in a : ( a7, a6, a5, a4, a3, a2, a1, a0 )
// in b : ( b7, b6, b5, b4, b3, b2, b1, b0 )
// in c : ( c7, c6, c5, c4, c3, c2, c1, c0 )
// in d : ( d7, d6, d5, d4, d3, d2, d1, d0 )
// out  : ( dsum, csum, bsum, asum )
inline __m128 hsum4x256_ps(__m256 a, __m256 b, __m256 c, __m256 d) {

    // (b3,b2,b1,b0, a3,a2,a1,a0)
    __m256 w = _mm256_permute2f128_ps(a, b, 0x20);
    // (b7,b6,b5,b4, a7,a6,a5,a4)
    __m256 x = _mm256_permute2f128_ps(a, b, 0x31);
    // (d3,d2,d1,d0, c3,c2,c1,c0)
    __m256 y = _mm256_permute2f128_ps(c, d, 0x20);
    // (d7,d6,d5,d4, c7,c6,c5,c4)
    __m256 z = _mm256_permute2f128_ps(c, d, 0x31);

    // (b3,b2,b1,b0, a3,a2,a1,a0)
    // (b7,b6,b5,b4, a7,a6,a5,a4)
    w = _mm256_add_ps(w, x);
    // (-,-,b3,b2, -,-,a3,a2)
    // (-,-,b7,b6, -,-,a7,a6)
    x = _mm256_permute_ps(w, _MM_SHUFFLE(3, 2, 3, 2));
    // (-,-,b1,b0, -,-,a1,a0)
    // (-,-,b5,b4, -,-,a5,a4)
    // (-,-,b3,b2, -,-,a3,a2)
    // (-,-,b7,b6, -,-,a7,a6)
    w = _mm256_add_ps(w, x);

    // (d3,d2,d1,d0, c3,c2,c1,c0)
    // (d7,d6,d5,d4, c7,c6,c5,c4)
    y = _mm256_add_ps(y, z);
    // (-,-,d3,d2, -,-,c3,c2)
    // (-,-,d7,d6, -,-,c7,c6)
    z = _mm256_permute_ps(y, _MM_SHUFFLE(3, 2, 3, 2));
    // (-,-,d1,d0, -,-,c1,c0)
    // (-,-,d5,d4, -,-,c5,c4)
    // (-,-,d3,d2, -,-,c3,c2)
    // (-,-,d7,d6, -,-,c7,c6)
    z = _mm256_add_ps(y, z);

    // d1,d0,b1,b0, c1,c0,a1,a0)
    // d5,d4,b5,b4, c5,c4,a5,a4)
    // d3,d2,b3,b2, c3,c2,a3,a2)
    // d7,d6,b7,b6, c7,c6,a7,a6)
    w = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(w), _mm256_castps_pd(z)));

    // (d0,d1,b0,b1, c0,c1,a0,a1)
    // (d4,d5,b4,b5, c4,c5,a4,a5)
    // (d2,d3,b2,b3, c2,c3,a2,a3)
    // (d6,d7,b6,b7, c6,c7,a6,a7)
    x = _mm256_permute_ps(w, _MM_SHUFFLE(2, 3, 0, 1));

    // (d1,d1,b1,b1, c1,c1,a1,a1)
    // (d5,d5,b5,b5, c5,c5,a5,a5)
    // (d3,d3,b3,b3, c3,c3,a3,a3)
    // (d7,d7,b7,b7, c7,c7,a7,a7)
    // (d0,d0,b0,b0, c0,c0,a0,a0)
    // (d4,d4,b4,b4, c4,c4,a4,a4)
    // (d2,d2,b2,b2, c2,c2,a2,a2)
    // (d6,d6,b6,b6, c6,c6,a6,a6)
    w = _mm256_add_ps(w, x);

    // (d1,d1,b1,b1)
    // (d5,d5,b5,b5)
    // (d3,d3,b3,b3)
    // (d7,d7,b7,b7)
    // (d0,d0,b0,b0)
    // (d4,d4,b4,b4)
    // (d2,d2,b2,b2)
    // (d6,d6,b6,b6)
    __m128 upper = _mm256_extractf128_ps(w, 1);

    // (d1,c1,b1,a1)
    // (d5,c5,b5,a5)
    // (d3,c3,b3,a3)
    // (d7,c7,b7,a7)
    // (d0,c0,b0,a0)
    // (d4,c4,b4,a4)
    // (d2,c2,b2,a2)
    // (d6,c6,b6,a6)
    __m128 ret = _mm_blend_ps(_mm256_castps256_ps128(w), upper, 0x0A /* 0b1010 */);

    return ret;
}

inline __m256 hsum8x256_ps(const __m256 &a,
                           const __m256 &b,
                           const __m256 &c,
                           const __m256 &d,
                           const __m256 &e,
                           const __m256 &f,
                           const __m256 &g,
                           const __m256 &h) {
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    __m256 tt0, tt1, tt2, tt3;
    t0 = _mm256_unpacklo_ps(a, b);    // b5,a5,b4,a4, b1,a1,b0,a0
    t1 = _mm256_unpackhi_ps(a, b);    // b7,a7,b6,a6, b3,a3,b2,a2
    t2 = _mm256_unpacklo_ps(c, d);    // d5,c5,d4,c4, d1,c1,d0,c0
    t3 = _mm256_unpackhi_ps(c, d);    // d7,c7,d6,c6, d3,c3,d2,c2
    t4 = _mm256_unpacklo_ps(e, f);    // f5,e5,f4,e4, f1,e1,f0,e0
    t5 = _mm256_unpackhi_ps(e, f);    // f7,e7,f6,e6, f3,e3,f2,e2
    t6 = _mm256_unpacklo_ps(g, h);    // h5,g5,h4,g4, h1,g1,h0,g0
    t7 = _mm256_unpackhi_ps(g, h);    // h7,g7,h6,g6, h3,g3,h2,g2

    tt0 = _mm256_add_ps(t0, t1);      // b57,a57,b46,a46, b13,a13,b02,a02
    tt1 = _mm256_add_ps(t2, t3);      // d57,c57,d46,c46, d13,c13,d02,c02
    tt2 = _mm256_add_ps(t4, t5);      // f57,e57,f46,e46, f13,e13,f02,e02
    tt3 = _mm256_add_ps(t6, t7);      // h57,g57,h46,g46, h13,g13,h02,g02

    t0 = _mm256_shuffle_ps(tt0, tt1,_MM_SHUFFLE(1,0,1,0)); // d46,c46,b46,a46, d02,c02,b02,a02
    t1 = _mm256_shuffle_ps(tt0, tt1,_MM_SHUFFLE(3,2,3,2)); // d57,c57,b57,a57, d13,c13,b13,a13
    t2 = _mm256_shuffle_ps(tt2, tt3,_MM_SHUFFLE(1,0,1,0)); // h46,g46,f46,e46, h02,g02,f02,e02
    t3 = _mm256_shuffle_ps(tt2, tt3,_MM_SHUFFLE(3,2,3,2)); // h57,g57,f57,e57, h13,g13,f13,e13

    tt0 = _mm256_add_ps(t0, t1);      // d4567,c4567,b4567,a4567, d0123,c0123,b0123,a0123
    tt1 = _mm256_add_ps(t2, t3);      // h4567,g4567,f4567,e4567, h0123,g0123,f0123,e0123

    t0 = _mm256_permute2f128_ps(tt0, tt1, 0x20); // h0123,g0123,f0123,e0123, d0123,c0123,b0123,a0123
    t1 = _mm256_permute2f128_ps(tt0, tt1, 0x31); // h4567,g4567,f4567,e4567, d4567,c4567,b4567,a4567

    return _mm256_add_ps(t0, t1);
}

// in  : ( x3, x2, x1, x0 )
// out : (  -,  -,  -, x3+x2+x1+x0 )
inline __m128d hsum256_pd(__m256d x) {
    // hiDual = ( x3, x2 )
    const __m128d hiDual = _mm256_extractf128_pd(x, 1);
    // loDual = ( x1, x0 )
    const __m128d loDual = _mm256_castpd256_pd128(x);
    // sumQuad = ( x2+x3, x0+x1 )
    const __m128d sumDual = _mm_add_pd(loDual, hiDual);
    // sum = ( 0, x0+x1+x2+x3 );
    const __m128d sum = _mm_hadd_pd(sumDual, _mm_setzero_pd());
    return sum;
}

