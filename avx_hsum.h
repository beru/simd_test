
#include <intrin.h>

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

