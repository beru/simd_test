
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>

#include <vector>
#include "timer.h"

unsigned long xor128() {
	static unsigned long x=123456789, y=362436069, z=521288629, w=88675123;
	unsigned long t=(x^(x<<11));
	x=y; y=z; z=w;
	return ( w=(w^(w>>19))^(t^(t>>8)) );
}

static
void test(int tableSize, int numElements)
{
	std::vector<uint8_t> table(tableSize);
	const int* pTable = (int*)&table[0];
	for (size_t i=0; i<table.size(); ++i) {
		table[i] = (256 * i) / tableSize;
	}
	std::vector<uint16_t> input(numElements);
	for (size_t i=0; i<numElements; ++i) {
		input[i] = xor128() % tableSize;
	}

	std::vector<uint8_t> output(numElements);

	Timer t;

	{
		t.Start();
		for (size_t i=0; i<numElements; ++i) {
			output[i] = table[input[i]];
			//output[i] = input[i];
		}
		printf("%f\n", t.ElapsedSecond());
	}

	{
		t.Start();

		const __m256i* pInput = (const __m256i*)&input[0];
		__m256i* pOutput = (__m256i*)&output[0];
#if 1
		const __m256i mask0 = _mm256_set1_epi32(0xFF);
		for (size_t i=0; i<numElements; i+=32) {
			__m256i in0 = _mm256_lddqu_si256(pInput++);
			__m256i in1 = _mm256_lddqu_si256(pInput++);

			__m256i ints00 = _mm256_unpacklo_epi16(in0, _mm256_setzero_si256());
			__m256i ints01 = _mm256_unpackhi_epi16(in0, _mm256_setzero_si256());
			__m256i ints10 = _mm256_unpacklo_epi16(in1, _mm256_setzero_si256());
			__m256i ints11 = _mm256_unpackhi_epi16(in1, _mm256_setzero_si256());
			__m256i gathered0 = _mm256_i32gather_epi32(pTable, ints00, 1);
			__m256i gathered1 = _mm256_i32gather_epi32(pTable, ints01, 1);
			__m256i gathered2 = _mm256_i32gather_epi32(pTable, ints10, 1);
			__m256i gathered3 = _mm256_i32gather_epi32(pTable, ints11, 1);
			__m256i packed0 = _mm256_packus_epi32(gathered0, gathered1);
			__m256i packed1 = _mm256_packus_epi32(gathered2, gathered3);
			__m256i packed = _mm256_packs_epi16(packed0, packed1);

			_mm256_stream_si256(pOutput++, packed);
		}
#else
		for (size_t i=0; i<numElements; i+=32) {
			__m256i in0 = _mm256_lddqu_si256(pInput++);
			__m256i in1 = _mm256_lddqu_si256(pInput++);
			_mm256_extractf128_pd
			table[(in0, 0)];
			_mm256_stream_si256(pOutput++, packed);
		}
#endif
		printf("%f\n", t.ElapsedSecond());
	}

#if 1
	int64_t sum = 0;
	for (size_t i=0; i<numElements; ++i) {
		sum += output[i];
	}
	printf("%lld\n", sum);
#endif

}

void test_gather()
{
	test(4096, 4096*4096);
//	test(4096, 1024*1024);

	int hoge = 0;
}


