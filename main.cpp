
#include <stdio.h>
#include <intrin.h>
#include <immintrin.h>
#include <stdint.h>

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
void print_mm256_bytes(__m256i bytes)
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
//	test();

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
		//	__m128i c = shift_bytes(a, i);
		//	print_m128i_bytes(c);
		//}
		//printf("\n");
		for (int i = -16; i <= 16; ++i) {
			__m128i c = funnel_shift_bytes(a, b, i);
			print_m128i_bytes(c);
		}
	}

	//printf("\n");
	//{
	//	__m256i a;
	//	a = _mm256_setr_epi8(
	//		0, 1, 2, 3, 4, 5, 6, 7,
	//		8, 9, 10, 11, 12, 13, 14, 15,
	//		16, 17, 18, 19, 20, 21, 22, 23,
	//		24, 25, 26, 27, 28, 29, 30, 31
	//		);
	//	for (int i = -32; i <= 32; ++i) {
	//		__m256i c = shift_bytes(a, i);
	//		print_mm256_bytes(c);
	//	}
	//}

//	getchar();
	return 0;
}

