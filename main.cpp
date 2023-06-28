#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

#include <string>

#include <nlohmann/json.hpp>

#include "bpe.h"
#include "blas.h"
#include "tensor.h"

/*
extern float* gpuTransferFloats(float *data, int size);
extern void gpuDumpMemoryInfo();
*/

struct CausalSelfAttention {
  int num_heads;
  Tensorf<1> c_attn_bias;
  Tensorf<2> c_attn_weight;
  Tensorf<1> c_proj_bias;
  Tensorf<2> c_proj_weight;

  // adds self-attention(x, kvbuf) to x at token index i
  // kvbuf is a buffer of shape <tokens, 2*embedding_dim>
  // (modifies kvbuf[i], reads kvbuf[:i-1])
  void apply(Tensorf<1> &x, Tensorf<2> &kvbuf, int i);
};

struct LayerNorm {
  Tensorf<1> bias;
  Tensorf<1> weight;

  void apply(const Tensorf<1> &out, const Tensorf<1> &in) {
    float sum1 = 0;
    float sum2 = 0;
    float *i = in.data;
    int n = in.shape[0];
    for (int j = 0; j < n; j++) {
      float ij = *i++;
      sum1 += ij;
      sum2 += ij*ij;
    }
    // compute mean and variance
    float mean = sum1 / in.shape[0];
    float variance = sum2 / in.shape[0] - mean * mean;
    const float eps = 1e-5; // layernorm default
    float invstddev = 1.0/sqrt(variance + eps);
    float *w = weight.data;
    float *b = bias.data;
    float *o = out.data;
    i = in.data;
    // could vectorize, but not as performance critical as dot/matmuls
    for (int j = 0; j < n; j++) {
      *o++ = ((*i++) - mean) * invstddev * (*w++) + (*b++);
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

  // x += attn(ln_1(x), kvbuf, i)
  // x += proj(gelu(fc(ln_2(x))))
  void apply(Tensorf<1> &x, Tensorf<2> &kvbuf, int i);
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

struct ModelState {
  Tensorf<3> x;      // <layers, tokens up to layers, embedding_dim>
  Tensorf<3> kvbuf;  // <layers, tokens, 2*embedding_dim>
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
  snprintf(buf, sizeof(buf), "h.%d.%s", layer, name);
  return fetch_weights<N>(j, addr, buf);
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

  BPEDecoder decoder;
  if (!decoder.Init("vocab.bin")) {
    fprintf(stderr, "Failed to init decoder from vocab.bin\n");
    exit(1);
  }

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
    {
      char buf[256];
      decoder.Decode(test_vector, N, buf, 256);
      printf("decoded test vector: %s\n", buf);
    }
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
          qbuf(i, k) = (*b++) + sdot(x, w, m.embedding_dim);
          w += m.embedding_dim;
        }
        for (int k = 0; k < 2*m.embedding_dim; k++) {
          lkvbuf(i, k) = (*b++) + sdot(x, w, m.embedding_dim);
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

        {
          float *att = attnbuf.data;
          for (int j = 0; j <= i; j++) {
            float *qk = qbuf.data + i*m.embedding_dim;
            float *kk = lkvbuf.data + j*2*m.embedding_dim;
            for (int h = 0; h < num_heads; h++) {
              *att++ = sdot(qk, kk, head_siz) * attn_scale;
              qk += head_siz;
              kk += head_siz;
            }
          }
        }
        // softmax
        for (int h = 0; h < num_heads; h++) {
          float max = -1e20;
          float denom = 0;
          float *att = attnbuf.data + h;
          for (int j = 0; j <= i; j++) {
            float a = *att;
            att += num_heads;
            if (a > max) {
              max = a;
            }
          }
          att = attnbuf.data + h;
          for (int j = 0; j <= i; j++) {
            float a = exp(*att - max);
            denom += a;
            *att = a;
            att += num_heads;
          }
          float scale = 1.0 / denom;
          att = attnbuf.data + h;
          for (int j = 0; j <= i; j++) {
            *att *= scale;
            att += num_heads;
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
        // input += c_proj_weight @ ybuf + c_proj_bias
        {
          float *w = m.h[l].attn.c_proj_weight.data;
          float *y = ybuf.data + i*m.embedding_dim;
          float *inp = input.data + i*m.embedding_dim;
          for (int j = 0; j < m.embedding_dim; j++) {
            *inp++ += m.h[l].attn.c_proj_bias[j] + sdot(y, w, m.embedding_dim);
            w += m.embedding_dim;
          }
        }

        // xbuf = layernorm(input)
        m.h[l].ln_2.apply(xbuf.slice(i), input.slice(i));
        // fc part of block
        // input += mlp_c_proj_weight @ gelu(mlp_c_fc_weight @ xbuf + mlp_c_fc_bias) + mlp_c_proj_bias
        int hidden_dim = 4*m.embedding_dim;
        {
          float *fc_w = m.h[l].mlp_c_fc_weight.data;
          float *x = xbuf.data + i*m.embedding_dim;
          float *h = hbuf.data + i*hidden_dim;
          for (int j = 0; j < hidden_dim; j++) {
            float sum = m.h[l].mlp_c_fc_bias[j] + sdot(x, fc_w, m.embedding_dim);
            float gelu = sum * 0.5 * (1.0 + tanh(0.7978845608028654 * (sum + 0.044715 * sum * sum * sum)));
            *h++ = gelu;
            fc_w += m.embedding_dim;
          }
        }
        // matmul the projection and sum into input
        {
          float *proj_w = m.h[l].mlp_c_proj_weight.data;
          float *inp = input.data + i*m.embedding_dim;
          float *h = hbuf.data + i*hidden_dim;
          for (int j = 0; j < m.embedding_dim; j++) {
            float sum = m.h[l].mlp_c_proj_bias[j] + sdot(h, proj_w, hidden_dim);
            *inp++ += sum;
            proj_w += hidden_dim;
          }
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
        logits[j] = sdot(ybuf.slice(i).data, w, m.embedding_dim);
        w += m.embedding_dim;
        if (logits[j] > logits[largmax]) {
          largmax = j;
        }
      }
      printf("logits: "); logits.show();
      printf("argmax: %d (%s) = %f\n", largmax, decoder.vocab_[largmax].c_str(), logits[largmax]);
    }
  }
  printf("expected result: x tensor([-3.4188, -1.0318, -3.5397,  5.2943,  3.9251, -2.0164, -2.1934, -2.5857, 0.5539, -4.0938])\n");

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  printf("elapsed: %f\n", elapsed);
}
