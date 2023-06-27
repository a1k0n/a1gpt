#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

#include <string>

#include <nlohmann/json.hpp>

/*
extern float* gpuTransferFloats(float *data, int size);
extern void gpuDumpMemoryInfo();
*/

template <int N> struct Tensorf {
  int shape[N];
  float *data;

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

  Tensorf<N-1> row(int i) const {
    if (N != 2) {
      fprintf(stderr, "Tensorf: row: expected 2 dimensions, got %d\n", N);
      abort();
    }
    if (i >= shape[0]) {
      fprintf(stderr, "Tensorf: out of bounds: %d >= %d\n", i, shape[N-2]);
      abort();
    }
    Tensorf<N-1> out;
    out.shape[0] = shape[1];
    out.data = data + i * shape[1];
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
    Tensorf<N> out;
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
};

Tensorf<2> NewMatrix(int i, int j) {
  Tensorf<2> out;
  out.shape[0] = i;
  out.shape[1] = j;
  out.data = new float[i * j];
  return out;
}

Tensorf<1> NewVector(int n) {
  Tensorf<1> out;
  out.shape[0] = n;
  out.data = new float[n];
  return out;
}

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

  Tensorf<2> wte_weight;
  Tensorf<2> wpe_weight;
  Tensorf<1> ln_f_weight;
  Tensorf<1> ln_f_bias;

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

int main() {
  // mmap "model.safetensors" into memory
  int fd = open("model.safetensors", O_RDONLY);
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
  m.ln_f_bias = fetch_weights<1>(j, addr, "ln_f.bias");
  m.ln_f_weight = fetch_weights<1>(j, addr, "ln_f.weight");
  m.embedding_dim = m.wte_weight.shape[1];
  m.context_len = m.wpe_weight.shape[0];
  printf("embedding_dim: %d\n", m.embedding_dim);
  printf("context_len: %d\n", m.context_len);

  for (int i = 0; i < 12; i++) {
    m.h[i].attn.num_heads = 12;
    m.h[i].attn.c_attn_bias = fetch_layer_weights<1>(j, addr, i, "attn.c_attn.bias");
    m.h[i].attn.c_attn_weight = fetch_layer_weights<2>(j, addr, i, "attn.c_attn.weight");
    m.h[i].attn.c_proj_bias = fetch_layer_weights<1>(j, addr, i, "attn.c_proj.bias");
    m.h[i].attn.c_proj_weight = fetch_layer_weights<2>(j, addr, i, "attn.c_proj.weight");
    m.h[i].ln_1.bias = fetch_layer_weights<1>(j, addr, i, "ln_1.bias");
    m.h[i].ln_1.weight = fetch_layer_weights<1>(j, addr, i, "ln_1.weight");
    m.h[i].ln_2.bias = fetch_layer_weights<1>(j, addr, i, "ln_2.bias");
    m.h[i].ln_2.weight = fetch_layer_weights<1>(j, addr, i, "ln_2.weight");
    m.h[i].mlp_c_fc_bias = fetch_layer_weights<1>(j, addr, i, "mlp.c_fc.bias");
    m.h[i].mlp_c_fc_weight = fetch_layer_weights<2>(j, addr, i, "mlp.c_fc.weight");
    m.h[i].mlp_c_proj_bias = fetch_layer_weights<1>(j, addr, i, "mlp.c_proj.bias");
    m.h[i].mlp_c_proj_weight = fetch_layer_weights<2>(j, addr, i, "mlp.c_proj.weight");
  }

  // tokenize("The rain in spain falls mainly on the")
  {
    const int N = 9;
    int test_vector[N] = {464, 6290,  287,  599,  391, 8953, 8384,  319,  262};
    auto input = NewMatrix(N, m.embedding_dim);
    for (int i = 0; i < N; i++) {
      float sum1 = 0;
      float sum2 = 0;
      for (int j = 0; j < m.embedding_dim; j++) {
        input(i, j) = m.wte_weight(test_vector[i], j) + m.wpe_weight(i, j);
      }
    }
    auto qkvbuf = NewMatrix(N, 3*m.embedding_dim);
    auto attnbuf = NewMatrix(N, m.h[0].attn.num_heads);
    auto xbuf = NewMatrix(N, m.embedding_dim);
    auto hbuf = NewMatrix(N, 4*m.embedding_dim);
    auto ybuf = NewMatrix(N, m.embedding_dim);

    for (int l = 0; l < 12; l++) {
      // transformer blocks
      ybuf.zero();
      for (int i = 0; i < N; i++) {
        // layernorm
        m.h[l].ln_1.apply(xbuf.row(i), input.row(i));
        // self-attention
        // matmul into qkvbuf; noting that weights are transposed
        // so we sum each input entry into the qkv buf
        for (int k = 0; k < 3*m.embedding_dim; k++) {
          qkvbuf(i, k) = m.h[l].attn.c_attn_bias[k];
        }
        for (int j = 0; j < m.embedding_dim; j++) {
          float x = xbuf(i, j);
          for (int k = 0; k < 3*m.embedding_dim; k++) {
            qkvbuf(i, k) += x * m.h[l].attn.c_attn_weight(j, k);
          }
        }
      }
      printf("qkvbuf(%d):\n", l);
      qkvbuf.show();

      // at this point, illustrating with 3 attention heads, qkvbuf looks like
      //        h1 h2 h3 h1 h2 h3 h1 h2 h3
      // token0 q1 q2 q3 k1 k2 k3 v1 v2 v3
      // token1 q2 q2 q3 k1 k2 k3 v1 v2 v3

      int num_heads = m.h[l].attn.num_heads;
      int head_siz = m.embedding_dim / num_heads;
      float attn_scale = 1.0 / sqrt(head_siz);
      int qoff = 0;
      attnbuf.zero();
      for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
          int qoff = 0;
          int koff = m.embedding_dim;
          for (int h = 0; h < num_heads; h++) {
            float sum = 0;
            for (int k = 0; k < head_siz; k++) {
              float qk = qkvbuf(i, qoff++);
              float vk = qkvbuf(j, koff++);
              sum += qk * vk;
            }
            attnbuf(j, h) = exp(sum * attn_scale);
          }
        }
        // softmax
        for (int h = 0; h < num_heads; h++) {
          float denom = 0;
          for (int j = 0; j <= i; j++) {
            denom += attnbuf(j, h);
          }
          denom = 1.0 / denom;
          for (int j = 0; j <= i; j++) {
            attnbuf(j, h) *= denom;
          }
        }
        // finally accumulate attention @ values -> ybuf
        for (int j = 0; j <= i; j++) {
          int voff = m.embedding_dim*2;
          int yoff = 0;
          for (int h = 0; h < num_heads; h++) {
            float a = attnbuf(j, h);
            for (int k = 0; k < head_siz; k++) {
              float vk = qkvbuf(j, voff++);
              ybuf(i, yoff++) += a * vk;
            }
          }
        }
        // matmul the projection and sum into input
        for (int j = 0; j < m.embedding_dim; j++) {
          float sum = m.h[l].attn.c_proj_bias[j];
          for (int k = 0; k < m.embedding_dim; k++) {
            // would be better to transpose the weight matrix
            sum += ybuf(i, k) * m.h[l].attn.c_proj_weight(k, j);
          }
          input(i, j) += sum;
        }

        // fc part of block
        m.h[l].ln_2.apply(xbuf.row(i), input.row(i));
        int hidden_dim = 4*m.embedding_dim;
        for (int j = 0; j < hidden_dim; j++) {
          float sum = m.h[l].mlp_c_fc_bias[j];
          for (int k = 0; k < m.embedding_dim; k++) {
            sum += xbuf(i, k) * m.h[l].mlp_c_fc_weight(k, j);
          }
          float gelu = sum * 0.5 * (1.0 + tanh(0.7978845608028654 * (sum + 0.044715 * sum * sum * sum)));
          hbuf(i, j) = gelu;
        }
        // matmul the projection and sum into input
        for (int j = 0; j < m.embedding_dim; j++) {
          float sum = m.h[l].mlp_c_proj_bias[j];
          for (int k = 0; k < hidden_dim; k++) {
            sum += hbuf(i, k) * m.h[l].mlp_c_proj_weight(k, j);
          }
          input(i, j) += sum;
        }
      }
      printf("x(%d):\n", l);
      input.show();
    }

    qkvbuf.destroy();
    input.destroy();
  }
}
