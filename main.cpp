#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

#include <string>

#ifdef __AVX__
#include <immintrin.h>
#endif

#include <nlohmann/json.hpp>

/*
extern float* gpuTransferFloats(float *data, int size);
extern void gpuDumpMemoryInfo();
*/

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

  void show() {
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

struct CausalSelfAttention {
  int num_heads;
  Tensorf<1> c_attn_bias;
  Tensorf<2> c_attn_weight;
  Tensorf<1> c_proj_bias;
  Tensorf<2> c_proj_weight;
};

struct LayerNorm {
  Tensorf<1> bias;
  Tensorf<1> weight;

  void apply(const Tensorf<1> &out, const Tensorf<1> &in) {
    float sum1 = 0;
    float sum2 = 0;
    for (int i = 0; i < in.shape[0]; i++) {
      sum1 += in[i];
      sum2 += in[i] * in[i];
    }
    // compute mean and variance
    float mean = sum1 / in.shape[0];
    float variance = sum2 / in.shape[0] - mean * mean;
    const float eps = 1e-5; // layernorm default
    float invstddev = 1.0/sqrt(variance + eps);
    for (int i = 0; i < in.shape[0]; i++) {
      out[i] = (in[i] - mean) * invstddev * weight[i] + bias[i];
    }
  }
};

struct TransformerBlock {
  // combined key, query, value
  CausalSelfAttention attn;
  LayerNorm ln_1, ln_2;
  Tensorf<1> mlp_c_fc_bias;
  Tensorf<2> mlp_c_fc_weight;
  Tensorf<1> mlp_c_proj_bias;
  Tensorf<2> mlp_c_proj_weight;
};

struct Model {
  int embedding_dim;
  int num_tokens;
  int context_len;
  int ntokens;

  Tensorf<2> wte_weight;
  Tensorf<2> wpe_weight;
  LayerNorm ln_f;

  TransformerBlock *h;
};

// TODO: load onto GPU
template <int N>
Tensorf<N> fetch_weights(const nlohmann::json &j, void *addr, const char *name) {
  Tensorf<N> out;

  int offset = j[name]["data_offsets"][0];
  auto shape = j[name]["shape"];
  if (N != shape.size()) {
    int shape_size = shape.size();
    fprintf(stderr, "fetch_weights: %s: expected %d dimensions, got %d\n", name, N, shape_size);
    exit(1);
  }
  for (int i = 0; i < N; i++) {
    out.shape[i] = shape[i];
  }
  
  float *weights = (float *)((char *)addr + offset);
  out.data = weights;

  return out;
}

template <int N>
Tensorf<N> fetch_layer_weights(const nlohmann::json &j, void *addr, int layer, const char *name) {
  char buf[256];
  sprintf(buf, "h.%d.%s", layer, name);
  return fetch_weights<N>(j, addr, buf);
}

static float dot(float *a, float *b, int n) {
#ifndef __AVX__
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
#else // vectorized dot product
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
#endif
}

static void saxpy(int n, float a, float * const x, float *y) {
#ifndef __AVX__
  for (int i = 0; i < n; i++) {
    y[i] += a * x[i];
  }
#else
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
#endif
}


int main() {
  // mmap "model.safetensors" into memory
  int fd = open("model.safetensors", O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Failed to open model.safetensors\n");
    exit(1);
  }
  struct stat sb;
  fstat(fd, &sb);
  char *addr = (char*) mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  uint64_t header_len = *((uint64_t *)addr);

  std::string header(addr + sizeof(uint64_t), header_len);
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(header);
  } catch (nlohmann::json::parse_error &e) {
    fprintf(stderr,
            "Failed to parse header: %s exception id: %d position %lu\n",
            e.what(), e.id, e.byte);
  }
  addr += header_len + sizeof(uint64_t);

  Model m;
  m.h = new TransformerBlock[12];
  m.wte_weight = fetch_weights<2>(j, addr, "wte.weight");
  m.wpe_weight = fetch_weights<2>(j, addr, "wpe.weight");
  m.ln_f.bias = fetch_weights<1>(j, addr, "ln_f.bias");
  m.ln_f.weight = fetch_weights<1>(j, addr, "ln_f.weight");
  m.embedding_dim = m.wte_weight.shape[1];
  m.context_len = m.wpe_weight.shape[0];
  m.ntokens = m.wte_weight.shape[0];
  printf("embedding_dim: %d\n", m.embedding_dim);
  printf("context_len: %d\n", m.context_len);
  printf("ntokens: %d\n", m.ntokens);

  for (int i = 0; i < 12; i++) {
    m.h[i].attn.num_heads = 12;
    m.h[i].attn.c_attn_bias = fetch_layer_weights<1>(j, addr, i, "attn.c_attn.bias");
    m.h[i].attn.c_attn_weight = *fetch_layer_weights<2>(j, addr, i, "attn.c_attn.weight").TransposedCopy();
    m.h[i].attn.c_proj_bias = fetch_layer_weights<1>(j, addr, i, "attn.c_proj.bias");
    m.h[i].attn.c_proj_weight = *fetch_layer_weights<2>(j, addr, i, "attn.c_proj.weight").TransposedCopy();
    m.h[i].ln_1.bias = fetch_layer_weights<1>(j, addr, i, "ln_1.bias");
    m.h[i].ln_1.weight = fetch_layer_weights<1>(j, addr, i, "ln_1.weight");
    m.h[i].ln_2.bias = fetch_layer_weights<1>(j, addr, i, "ln_2.bias");
    m.h[i].ln_2.weight = fetch_layer_weights<1>(j, addr, i, "ln_2.weight");
    m.h[i].mlp_c_fc_bias = fetch_layer_weights<1>(j, addr, i, "mlp.c_fc.bias");
    m.h[i].mlp_c_fc_weight = *fetch_layer_weights<2>(j, addr, i, "mlp.c_fc.weight").TransposedCopy();
    m.h[i].mlp_c_proj_bias = fetch_layer_weights<1>(j, addr, i, "mlp.c_proj.bias");
    m.h[i].mlp_c_proj_weight = *fetch_layer_weights<2>(j, addr, i, "mlp.c_proj.weight").TransposedCopy();
  }

  // tokenize("The rain in spain falls mainly on the")
  // benchmark 100 iterations
  // get t0 first
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  {
    const int N = 9;
    int test_vector[N] = {464, 6290,  287,  599,  391, 8953, 8384,  319,  262};
    Tensorf<2> input(N, m.embedding_dim);
    for (int i = 0; i < N; i++) {
      float sum1 = 0;
      float sum2 = 0;
      for (int j = 0; j < m.embedding_dim; j++) {
        input(i, j) = m.wte_weight(test_vector[i], j) + m.wpe_weight(i, j);
      }
    }
    // query buf (reused for each layer)
    Tensorf<2> qbuf(N, m.embedding_dim);
    // per-layer key-value buf (retained between subsequent tokens)
    Tensorf<3> kvbuf(12, N, 2*m.embedding_dim);
    // TODO: transpose attnbuf
    Tensorf<2> attnbuf(N, m.h[0].attn.num_heads);
    Tensorf<2> xbuf(N, m.embedding_dim);
    Tensorf<2> hbuf(N, 4*m.embedding_dim);
    Tensorf<2> ybuf(N, m.embedding_dim);
    Tensorf<1> logits(m.ntokens);

    for (int l = 0; l < 12; l++) {
      // transformer blocks
      ybuf.zero();
      Tensorf<2> lkvbuf = kvbuf.slice(l);
      for (int i = 0; i < N; i++) {
        // layernorm
        m.h[l].ln_1.apply(xbuf.slice(i), input.slice(i));
        // self-attention
        // matmul into qkvbuf; noting that weights are transposed
        // so we sum each input entry into the qkv buf
        float *w = m.h[l].attn.c_attn_weight.data;
        float *x = xbuf.data + i*m.embedding_dim;
        float *b = m.h[l].attn.c_attn_bias.data;
        for (int k = 0; k < m.embedding_dim; k++) {
          qbuf(i, k) = (*b++) + dot(x, w, m.embedding_dim);
          w += m.embedding_dim;
        }
        for (int k = 0; k < 2*m.embedding_dim; k++) {
          lkvbuf(i, k) = (*b++) + dot(x, w, m.embedding_dim);
          w += m.embedding_dim;
        }
      }
      /*
        printf("qkvbuf(%d):\n", l);
        qkvbuf.show();
      */

      // at this point, illustrating with 3 attention heads, qkvbuf looks like
      //        h1 h2 h3 h1 h2 h3 h1 h2 h3
      // token0 q1 q2 q3 k1 k2 k3 v1 v2 v3
      // token1 q2 q2 q3 k1 k2 k3 v1 v2 v3

      int num_heads = m.h[l].attn.num_heads;
      int head_siz = m.embedding_dim / num_heads;
      float attn_scale = 1.0 / sqrt(head_siz);
      int qoff = 0;
      //attnbuf.zero();
      for (int i = 0; i < N; i++) {
        // for generation, we don't need to compute the full attention matrix
        // for the last block, but it uses information from all previous tokens
        // & blocks.
        if (l == 11 && i < N-1) continue;

        for (int j = 0; j <= i; j++) {
          float *qk = qbuf.data + i*m.embedding_dim;
          float *kk = lkvbuf.data + j*2*m.embedding_dim;
          for (int h = 0; h < num_heads; h++) {
            attnbuf(j, h) = dot(qk, kk, head_siz) * attn_scale;
            qk += head_siz;
            kk += head_siz;
          }
        }
        // softmax
        for (int h = 0; h < num_heads; h++) {
          float max = -1e20;
          float denom = 0;
          for (int j = 0; j <= i; j++) {
            float a = attnbuf(j, h);
            if (a > max) {
              max = a;
            }
          }
          for (int j = 0; j <= i; j++) {
            float a = exp(attnbuf(j, h) - max);
            denom += a;
            attnbuf(j, h) = a;
          }
          float scale = 1.0 / denom;
          for (int j = 0; j <= i; j++) {
            attnbuf(j, h) *= scale;
          }
        }
        // finally accumulate attention @ values -> ybuf
        for (int j = 0; j <= i; j++) {
          float *y = ybuf.data + i*m.embedding_dim;
          int voff = m.embedding_dim;
          float *v = lkvbuf.data + j*2*m.embedding_dim + voff;
          float *a = attnbuf.data + j*num_heads;
          for (int h = 0; h < num_heads; h++) {
            saxpy(head_siz, *a++, v, y);
            v += head_siz;
            y += head_siz;
          }
        }
        // matmul the projection and sum into input
        for (int j = 0; j < m.embedding_dim; j++) {
          float *y = ybuf.data + i*m.embedding_dim;
          float *w = m.h[l].attn.c_proj_weight.data + j*m.embedding_dim;
          float sum = m.h[l].attn.c_proj_bias[j] + dot(y, w, m.embedding_dim);
          input(i, j) += sum;
        }

        // fc part of block
        m.h[l].ln_2.apply(xbuf.slice(i), input.slice(i));
        int hidden_dim = 4*m.embedding_dim;
        float *fc_w = m.h[l].mlp_c_fc_weight.data;
        for (int j = 0; j < hidden_dim; j++) {
          float *x = xbuf.data + i*m.embedding_dim;
          float sum = m.h[l].mlp_c_fc_bias[j] + dot(x, fc_w, m.embedding_dim);
          fc_w += m.embedding_dim;
          float gelu = sum * 0.5 * (1.0 + tanh(0.7978845608028654 * (sum + 0.044715 * sum * sum * sum)));
          hbuf(i, j) = gelu;
        }
        // matmul the projection and sum into input
        float *proj_w = m.h[l].mlp_c_proj_weight.data;
        for (int j = 0; j < m.embedding_dim; j++) {
          float *h = hbuf.data + i*hidden_dim;
          float sum = m.h[l].mlp_c_proj_bias[j] + dot(h, proj_w, hidden_dim);
          proj_w += hidden_dim;
          input(i, j) += sum;
        }
      }
      printf("x(%d):\n", l);
      input.slice(N-1).show();
    }

    // finally, layernorm and dot with embedding matrix
    for (int i = N-1; i < N; i++) {
      m.ln_f.apply(ybuf.slice(i), input.slice(i));
      float *w = m.wte_weight.data;
      int largmax = 0;
      for (int j = 0; j < m.ntokens; j++) {
        logits[j] = dot(ybuf.slice(i).data, w, m.embedding_dim);
        w += m.embedding_dim;
        if (logits[j] > logits[largmax]) {
          largmax = j;
        }
      }
      printf("logits: "); logits.show();
      printf("argmax: %d (%f)\n", largmax, logits[largmax]);
      printf("logits[198]: %f\n", logits[198]);
    }
  }
  printf("expected result: x tensor([-3.4188, -1.0318, -3.5397,  5.2943,  3.9251, -2.0164, -2.1934, -2.5857, 0.5539, -4.0938])\n");

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  printf("elapsed: %f\n", elapsed);
}
