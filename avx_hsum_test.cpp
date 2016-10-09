
#include <stdio.h>
#include <limits>
#include "avx_hsum.h"

void float_test()
{
    float fs[8*4];
    for (size_t i=0; i<_countof(fs); ++i) {
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
        a = results[0];
        b = fs[0] + fs[1] + fs[2] + fs[3] + fs[4] + fs[5] + fs[6] + fs[7];
        diff = a - b;
        pass = abs(diff) <= std::numeric_limits<float>::epsilon();

        a = results[1];
        b = fs[010] + fs[011] + fs[012] + fs[013] + fs[014] + fs[015] + fs[016] + fs[017];
        diff = a - b;
        pass = pass && (abs(diff) <= std::numeric_limits<float>::epsilon());

        a = results[2];
        b = fs[020] + fs[021] + fs[022] + fs[023] + fs[024] + fs[025] + fs[026] + fs[027];
        diff = a - b;
        pass = pass && (abs(diff) <= std::numeric_limits<float>::epsilon());

        a = results[3];
        b = fs[030] + fs[031] + fs[032] + fs[033] + fs[034] + fs[035] + fs[036] + fs[037];
        diff = a - b;
        pass = pass && (abs(diff) <= std::numeric_limits<float>::epsilon());

        printf("%s hsum4x256_ps\n", pass ? "PASS" : "FAIL");
    }
}

void double_test()
{
    double fs[4*4];
    for (size_t i=0; i<_countof(fs); ++i) {
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

