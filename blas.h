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
  float *aend = a + n;
  if (n > 15) {
    __m256 sum8 = _mm256_setzero_ps(); // accumulate in a vector
    __m256 sum81 = _mm256_setzero_ps(); // accumulate in a vector
    int n8 = n&(~15);
    float *aend = a + n8;
    for (; a < aend; a += 16, b += 16) {
      __m256 a80 = _mm256_load_ps(a);
      __m256 b80 = _mm256_load_ps(b);
      __m256 a81 = _mm256_load_ps(a + 8);
      __m256 b81 = _mm256_load_ps(b + 8);
      __m256 prod0 = _mm256_mul_ps(a80, b80);
      __m256 prod1 = _mm256_mul_ps(a81, b81);
      sum8 = _mm256_add_ps(sum8, prod0);
      sum81 = _mm256_add_ps(sum81, prod1);
    }
    // sum up the vector
    sum8 = _mm256_add_ps(sum8, sum81);
    __m128 low128 = _mm256_extractf128_ps(sum8, 0);
    __m128 high128 = _mm256_extractf128_ps(sum8, 1);
    low128 = _mm_add_ps(low128, high128);
    low128 = _mm_hadd_ps(low128, low128);
    low128 = _mm_hadd_ps(low128, low128);
    sum = _mm_cvtss_f32(low128);
  }
  while (a < aend) {
    sum += (*a++) * (*b++);
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

// single-precision vector addition: y = a*x + y
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


// variant on saxpy: y = x + b*y
static void sxpby(int n, float *const x, float b, float *y) {
#ifdef __APPLE__
  cblas_sscal(n, b, y, 1);
  cblas_saxpy(n, 1.0, x, 1, y, 1);
#elif defined(__AVX__)
  // n is assumed to be a multiple of 8 and y is assumed to be aligned
  __m256 b_vec = _mm256_set1_ps(b);
  int i = 0;
  for (; i < n; i += 8) {
      __m256 src_vec = _mm256_load_ps(x + i);
      __m256 dest_vec = _mm256_load_ps(y + i);
      __m256 result_vec = _mm256_add_ps(src_vec, _mm256_mul_ps(b_vec, dest_vec));
      _mm256_store_ps(y + i, result_vec);
  }
  assert(i == n && "xpby: n is not a multiple of 8");
#else
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + b*y[i];
  }
#endif
}

static void sscal(int n, float a, float *x) {
#ifdef __APPLE__
  cblas_sscal(n, a, x, 1);
#elif defined(__AVX__)
  // n is assumed to be a multiple of 8 and x is assumed to be aligned
  __m256 a_vec = _mm256_set1_ps(a);
  int i = 0;
  for (; i < n; i += 8) {
      __m256 src_vec = _mm256_load_ps(x + i);
      __m256 result_vec = _mm256_mul_ps(a_vec, src_vec);
      _mm256_store_ps(x + i, result_vec);
  }
  assert(i == n && "scal: n is not a multiple of 8");
#else
  for (int i = 0; i < n; i++) {
    x[i] *= a;
  }
#endif
}