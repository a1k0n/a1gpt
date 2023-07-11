#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef CUDA
#include <cuda_runtime.h>
#endif

#undef ALLOC_DEBUG

template <int N> struct Tensorf {
  int shape[N];
  float *data;
  float *alloc;
#ifdef CUDA
  float *gpu_data;
  float *gpu_alloc;
#endif

  Tensorf() {
    data = NULL;
    alloc = NULL;
#ifdef CUDA
    gpu_data = NULL;
    gpu_alloc = NULL;
#endif
  }

  Tensorf(float *_data, int i) {
    assert(N == 1);
    shape[0] = i;
    data = _data;
    alloc = NULL;
#ifdef CUDA
    gpu_data = NULL;
    gpu_alloc = NULL;
#endif
  }

  Tensorf(float *_data, int i, int j) {
    assert(N == 2);
    shape[0] = i;
    shape[1] = j;
    data = _data;
    alloc = NULL;
#ifdef CUDA
    gpu_data = NULL;
    gpu_alloc = NULL;
#endif
  }

  void _alloc(size_t nfloats) {
#ifdef CUDA
    _alloc_device(nfloats);
    alloc = NULL;
    data = NULL;
#else
    _alloc_local(nfloats);
#endif
  }

  void _alloc_device(size_t nfloats) {
#ifdef CUDA
    if (cudaMalloc(&gpu_alloc, nfloats * sizeof(float)) != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaGetLastError()));
      abort();
    }
    gpu_data = gpu_alloc;
#else
    _alloc_local(nfloats);
#endif
  }

  void _alloc_local(size_t nfloats) {
    // allocate aligned float array
    alloc = new float[nfloats + 7];
    data = (float *)(((uintptr_t)alloc + 31) & ~31);
#ifdef ALLOC_DEBUG
    printf("allocating (%d) %p -> %p\n", nfloats, alloc, data);
#endif
  }

  bool copyToDevice() {
#ifdef CUDA
    if (gpu_data == NULL) {
      _alloc_device(size());
    }
    if (cudaMemcpy(gpu_data, data, size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaGetLastError()));
      abort();
    }
#endif
    return true;
  }

  bool copyToCpu() {
#ifdef CUDA
    if (data == NULL) {
      _alloc_local(size());
    }
    if (cudaMemcpy(data, gpu_data, size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaGetLastError()));
      abort();
    }
#endif
    return true;
  }

  Tensorf(int i) {
    assert(N == 1);
    shape[0] = i;
    _alloc(i);
  }

  Tensorf(int i, int j) {
    assert(N == 2);
    shape[0] = i;
    shape[1] = j;
    _alloc(i*j);
  }

  Tensorf(int i, int j, int k) {
    assert(N == 3);
    shape[0] = i;
    shape[1] = j;
    shape[2] = k;
    _alloc(i*j*k);
  }

  Tensorf(const Tensorf<N> &other) {
    for (int i = 0; i < N; i++) {
      shape[i] = other.shape[i];
    }
    data = other.data;
    alloc = other.alloc;
#ifdef CUDA
    gpu_data = other.gpu_data;
    gpu_alloc = other.gpu_alloc;
#endif
  }

  ~Tensorf() {
    if (alloc) {
#ifdef ALLOC_DEBUG
      printf("freeing %p\n", alloc);
#endif
#ifdef CUDA
      if (alloc == gpu_data) {
        cudaFree(gpu_data);
        return;
      }
#endif
      delete[] alloc;
    }
  }

  void show() const {
    if (data == NULL) {
      printf("Tensorf: NULL\n");
      return;
    }
    if (N == 1) {
      int k = 128;
      if (shape[0] < k) {
        k = shape[0];
      }
      for (int i = 0; i < k; i++) {
        printf("%7.4f ", data[i]);
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
          printf("%7.4f ", data[i * shape[1] + j]);
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
    if (data != NULL) {
      out.data = data + i * stride;
    }
    out.alloc = NULL;
#ifdef CUDA
    if (gpu_data != NULL) {
      out.gpu_data = gpu_data + i * stride;
    }
    out.gpu_alloc = NULL;
#endif
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
};
