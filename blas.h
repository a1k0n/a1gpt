#pragma once

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

#include <cassert>

// optimized blas routines needed for our little implementation

static float sdot(float *a, float *b, int n) {
#ifdef __APPLE__
  return cblas_sdot(n, a, 1, b, 1);
#elif defined(__AVX__)
  int i = 0;
  float sum = 0;
  if (n > 7) {
    __m256 sum8 = _mm256_setzero_ps(); // accumulate in a vector
    int n8 = n&(~7);
    for (; i < n8; i += 8) {
      __m256 a8 = _mm256_loadu_ps(a + i);
      __m256 b8 = _mm256_loadu_ps(b + i);
      __m256 prod = _mm256_mul_ps(a8, b8);
      sum8 = _mm256_add_ps(sum8, prod);
    }
    // sum up the vector
    __m128 low128 = _mm256_extractf128_ps(sum8, 0);
    __m128 high128 = _mm256_extractf128_ps(sum8, 1);
    low128 = _mm_add_ps(low128, high128);
    low128 = _mm_hadd_ps(low128, low128);
    low128 = _mm_hadd_ps(low128, low128);
    sum = _mm_cvtss_f32(low128);
  }
  for (; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
#else
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
#endif
}

static void saxpy(int n, float a, float * const x, float *y) {
#ifdef __APPLE__
  cblas_saxpy(n, a, x, 1, y, 1);
#elif defined(__AVX__)
  // n is assumed to be a multiple of 8 and y is assumed to be aligned
  __m256 a_vec = _mm256_set1_ps(a);
  int i = 0;
  for (; i < n; i += 8) {
      __m256 src_vec = _mm256_load_ps(x + i);
      __m256 dest_vec = _mm256_load_ps(y + i);
      __m256 result_vec = _mm256_add_ps(dest_vec, _mm256_mul_ps(a_vec, src_vec));
      _mm256_store_ps(y + i, result_vec);
  }
  assert(i == n && "axpy: n is not a multiple of 8");
#else
  for (int i = 0; i < n; i++) {
    y[i] += a * x[i];
  }
#endif
}

