
#include <stdio.h>
#include <intrin.h>
#include <immintrin.h>

template<bool> struct Range;

template<unsigned int N, typename = Range<true> >
struct mm256_shift_left_impl
{};

template<unsigned int N>
struct mm256_shift_left_impl<N, Range<(0 <= N && N <= 16)> >
{
	static __m256i doit(__m256i a)
	{
		__m256i mask = _mm256_permute2x128_si256(a, a, _MM_SHUFFLE(0,0,3,0) );
		return _mm256_alignr_epi8(a,mask,16-N);
	}
};

template<unsigned int N>
struct mm256_shift_left_impl<N, Range<(16 < N && N <= 32)> >
{
	static __m256i doit(__m256i a)
	{
		__m256i y1 = _mm256_slli_si256(a, N - 16);
		return _mm256_permute2x128_si256(y1, y1, _MM_SHUFFLE(0,0,3,0) );
	}
};

template <unsigned int N>
__m256i mm256_shift_left(__m256i a)
{
	return mm256_shift_left_impl<N>::doit(a);
}

static
void print_mm256_bytes(__m256i bytes)
{
	for (int i = 0; i < 32; i++) {
		printf("%2d ",((unsigned char *)&bytes)[i]);
	}
	printf("\n");
}

template <unsigned int N>
void test(__m256i reg)
{
	print_mm256_bytes(mm256_shift_left<N>(reg));
}


void test()
{
	__m256i reg =  _mm256_set_epi8(
		32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,
		16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1
	);
	
	test<0>(reg);
	test<1>(reg);
	test<2>(reg);
	test<3>(reg);
	test<4>(reg);
	test<5>(reg);
	test<6>(reg);
	test<7>(reg);
	test<8>(reg);
	test<9>(reg);
	test<10>(reg);
	test<11>(reg);
	test<12>(reg);
	test<13>(reg);
	test<14>(reg);
	test<15>(reg);
	test<16>(reg);
	test<17>(reg);
	test<18>(reg);
	test<19>(reg);
	test<20>(reg);
	test<21>(reg);
	test<22>(reg);
	test<23>(reg);
	test<24>(reg);
	test<25>(reg);
	test<26>(reg);
	test<27>(reg);
	test<28>(reg);
	test<29>(reg);
	test<30>(reg);
	test<31>(reg);
	test<32>(reg);
}

int main(int argc, char* argv[])
{
	test();
	return 0;
}

