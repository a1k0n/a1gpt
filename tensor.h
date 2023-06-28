#pragma once

#include <unistd.h>

template <int N> struct Tensorf {
  int shape[N];
  float *data;
  float *alloc;

  Tensorf() {
    alloc = NULL;
  }

  Tensorf(int i) {
    shape[0] = i;
    // allocate aligned float array
    alloc = new float[i + 7];
    data = (float *)(((uintptr_t)alloc + 31) & ~31);
  }

  Tensorf(int i, int j) {
    shape[0] = i;
    shape[1] = j;
    // allocate aligned float array
    alloc = new float[i * j + 7];
    data = (float *)(((uintptr_t)alloc + 31) & ~31);
  }

  Tensorf(int i, int j, int k) {
    shape[0] = i;
    shape[1] = j;
    shape[2] = k;
    // allocate aligned float array
    alloc = new float[i * j * k + 7];
    data = (float *)(((uintptr_t)alloc + 31) & ~31);
  }

  Tensorf(const Tensorf<N> &other) {
    for (int i = 0; i < N; i++) {
      shape[i] = other.shape[i];
    }
    data = other.data;
    alloc = other.alloc;
  }

  ~Tensorf() {
    if (alloc) {
      delete[] alloc;
    }
  }

  void show() const {
    if (N == 1) {
      int k = 10;
      if (shape[0] < k) {
        k = shape[0];
      }
      for (int i = 0; i < k; i++) {
        printf("%f ", data[i]);
      }
      printf("\n");
    } else if (N == 2) {
      int ki = 10;
      int kj = 10;
      if (shape[0] < ki) {
        ki = shape[0];
      }
      if (shape[1] < kj) {
        kj = shape[1];
      }
      for (int i = 0; i < ki; i++) {
        for (int j = 0; j < kj; j++) {
          printf("%f ", data[i * shape[1] + j]);
        }
        printf("\n");
      }
    }
  }

  float& operator[](int i) const {
    if (N != 1) {
      fprintf(stderr, "Tensorf: operator[]: expected 1 dimension, got %d\n", N);
      abort();
    }
    if (i >= shape[0]) {
      fprintf(stderr, "Tensorf: out of bounds: %d >= %d\n", i, shape[N-1]);
      abort();
    }
    return data[i];
  }

  Tensorf<N-1> slice(int i) const {
    if (N <= 1) {
      fprintf(stderr, "Tensorf: row: expected >1 dimensions, got %d\n", N);
      abort();
    }
    if (i >= shape[0]) {
      fprintf(stderr, "Tensorf: out of bounds: %d >= %d\n", i, shape[0]);
      abort();
    }
    // return new tensor with no alloc, so it won't destroy the underlying array
    // when it goes out of scope
    Tensorf<N-1> out;
    int stride = 1;
    for (int j = 0; j < N-1; j++) {
      out.shape[j] = shape[j+1];
      stride *= shape[j+1];
    }
    out.data = data + i * stride;
    out.alloc = NULL;
    return out;
  }

  float& operator()(int i, int j) const {
    if (N != 2) {
      fprintf(stderr, "Tensorf: operator[]: expected 2 dimensions, got %d\n", N);
      abort();
    }
    if (i >= shape[0]) {
      fprintf(stderr, "Tensorf: out of bounds: %d >= %d\n", i, shape[N-2]);
      abort();
    }
    if (j >= shape[1]) {
      fprintf(stderr, "Tensorf: out of bounds: %d >= %d\n", j, shape[N-1]);
      abort();
    }
    return data[i * shape[1] + j];
  }

  Tensorf<N>& operator+=(const Tensorf<N> &other) {
    int size = 1;
    for (int i = 0; i < N; i++) {
      if (shape[i] != other.shape[i]) {
        fprintf(stderr, "Tensorf: operator+: shape mismatch\n");
        abort();
      }
      size *= shape[i];
    }
    for (int i = 0; i < size; i++) {
      data[i] += other.data[i];
    }
    return *this;
  }

  size_t size() const {
    size_t size = 1;
    for (int i = 0; i < N; i++) {
      size *= shape[i];
    }
    return size;
  }

  void zero() {
    memset(data, 0, size() * sizeof(float));
  }

  void destroy() {
    delete[] data;
  }

  Tensorf<2> *TransposedCopy() {
    int m = shape[1], n = shape[0];
    Tensorf<2> *out = new Tensorf<2>(m, n);
    float *dout = out->data;
    for (int j = 0; j < m; j++) {
      float *din = data + j;
      for (int i = 0; i < n; i++) {
        *dout++ = *din;
        din += m;
      }
    }
    return out;
  }
};
