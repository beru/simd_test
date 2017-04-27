
#include <stdio.h>
#include <limits>
#include <type_traits>

#include "avx_hsum.h"

template < typename T, size_t N >
size_t countof( T ( & arr )[ N ] )
{
    return std::extent< T[ N ] >::value;
}

void float_test()
{
    float fs[8*8];
    for (size_t i=0; i<countof(fs); ++i) {
        fs[i] = (float)(i + 1);
    }

    __m128 ret;
    float results[8];
    float a, b, diff;
    bool pass;
    
    {
        ret = hsum128_ps_naive(_mm_loadu_ps(&fs[0]));
        _mm_storeu_ps(results, ret);
        a = results[0];
        b = fs[0] + fs[1] + fs[2] + fs[3];
        diff = a - b;
        pass = abs(diff) <= std::numeric_limits<float>::epsilon();
        printf("%s hsum128_ps_naive\n", pass ? "PASS" : "FAIL");
    }

    {
        ret = hsum128_ps(_mm_loadu_ps(&fs[0]));
        _mm_storeu_ps(results, ret);
        a = results[0];
        b = fs[0] + fs[1] + fs[2] + fs[3];
        diff = a - b;
        pass = abs(diff) <= std::numeric_limits<float>::epsilon();
        printf("%s hsum128_ps\n", pass ? "PASS" : "FAIL");
    }

    {
        ret = hsum256_ps(_mm256_loadu_ps(&fs[0]));
        _mm_storeu_ps(results, ret);
        a = results[0];
        b = fs[0] + fs[1] + fs[2] + fs[3] + fs[4] + fs[5] + fs[6] + fs[7];
        diff = a - b;
        pass = abs(diff) <= std::numeric_limits<float>::epsilon();
        printf("%s hsum256_ps\n", pass ? "PASS" : "FAIL");
    }

    {
        ret = hsum2x256_ps(_mm256_loadu_ps(&fs[0]), _mm256_loadu_ps(&fs[010]));
        _mm_storeu_ps(results, ret);
        a = results[0];
        b = fs[0] + fs[1] + fs[2] + fs[3] + fs[4] + fs[5] + fs[6] + fs[7];
        diff = a - b;
        pass = abs(diff) <= std::numeric_limits<float>::epsilon();

        a = results[1];
        b = fs[010] + fs[011] + fs[012] + fs[013] + fs[014] + fs[015] + fs[016] + fs[017];
        diff = a - b;
        pass = pass && (abs(diff) <= std::numeric_limits<float>::epsilon());

        printf("%s hsum2x256_ps\n", pass ? "PASS" : "FAIL");
    }

    {
        ret = hsum4x256_ps(
            _mm256_loadu_ps(&fs[0]),
            _mm256_loadu_ps(&fs[010]),
            _mm256_loadu_ps(&fs[020]),
            _mm256_loadu_ps(&fs[030])
        );
        _mm_storeu_ps(results, ret);
        pass = true;
        for (size_t i=0; i<4; ++i) {
            a = results[i];
            int base = 8 * i;
            b = fs[base+0] + fs[base+1] + fs[base+2] + fs[base+3] + fs[base+4] + fs[base+5] + fs[base+6] + fs[base+7];
            diff = a - b;
            pass &= abs(diff) <= std::numeric_limits<float>::epsilon();
        }
        printf("%s hsum4x256_ps\n", pass ? "PASS" : "FAIL");
    }

    {
        __m256 ret;
        ret = hsum6x256_ps(
            _mm256_loadu_ps(&fs[0]),
            _mm256_loadu_ps(&fs[010]),
            _mm256_loadu_ps(&fs[020]),
            _mm256_loadu_ps(&fs[030]),
            _mm256_loadu_ps(&fs[040]),
            _mm256_loadu_ps(&fs[050])
        );
        _mm256_storeu_ps(results, ret);
        pass = true;
        for (size_t i=0; i<6; ++i) {
            a = results[i];
            int base = 8 * i;
            b = fs[base+0] + fs[base+1] + fs[base+2] + fs[base+3] + fs[base+4] + fs[base+5] + fs[base+6] + fs[base+7];
            diff = a - b;
            pass &= abs(diff) <= std::numeric_limits<float>::epsilon();
        }
        printf("%s hsum6x256_ps\n", pass ? "PASS" : "FAIL");
    }

    {
        __m256 ret;
        ret = hsum8x256_ps(
            _mm256_loadu_ps(&fs[0]),
            _mm256_loadu_ps(&fs[010]),
            _mm256_loadu_ps(&fs[020]),
            _mm256_loadu_ps(&fs[030]),
            _mm256_loadu_ps(&fs[040]),
            _mm256_loadu_ps(&fs[050]),
            _mm256_loadu_ps(&fs[060]),
            _mm256_loadu_ps(&fs[070])
        );
        _mm256_storeu_ps(results, ret);
        pass = true;
        for (size_t i=0; i<8; ++i) {
            a = results[i];
            int base = 8 * i;
            b = fs[base+0] + fs[base+1] + fs[base+2] + fs[base+3] + fs[base+4] + fs[base+5] + fs[base+6] + fs[base+7];
            diff = a - b;
            pass &= abs(diff) <= std::numeric_limits<float>::epsilon();
        }
        printf("%s hsum8x256_ps\n", pass ? "PASS" : "FAIL");
    }


}


void double_test()
{
    double fs[4*4];
    for (size_t i=0; i<countof(fs); ++i) {
        fs[i] = (double)(i + 1);
    }

    __m128d ret;
    double results[4];
    double a, b, diff;
    bool pass;

    {
        ret = hsum256_pd(_mm256_loadu_pd(&fs[0]));
        _mm_storeu_pd(results, ret);
        a = results[0];
        b = fs[0] + fs[1] + fs[2] + fs[3];
        diff = a - b;
        pass = abs(diff) <= std::numeric_limits<double>::epsilon();
        printf("%s hsum256_pd\n", pass ? "PASS" : "FAIL");
    }

}

int main(int argc, char* argv[])
{
    float_test();
    double_test();

    return 0;
}

