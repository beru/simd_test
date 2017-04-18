cmake_minimum_required (VERSION 2.8.11)
project (simd_test LANGUAGES CXX)

add_executable (simd_test
    avx_hsum_test.cpp
)

set(CMAKE_CXX_FLAGS "-std=c++11 -mavx")


