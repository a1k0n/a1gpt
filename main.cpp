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
  void apply(const Tensorf<1> &out, const Tensorf<1> &xbuf, const Tensorf<2> &kvbuf, int i) {
    int emb_siz = xbuf.shape[0];
    Tensorf<2> attnbuf(i+1, num_heads);
    Tensorf<1> qbuf(emb_siz);
    Tensorf<1> ybuf(emb_siz);

    int head_siz = emb_siz / num_heads;
    float attn_scale = 1.0 / sqrt(head_siz);
    // first compute q, kv[i]

    {
      // matmul into q/kv; kv is cached for future invocations so write our
      // entry in there
      float *w = c_attn_weight.data;
      float *x = xbuf.data;
      float *b = c_attn_bias.data;
      float *q = qbuf.data;
      // matmul q = Qx
      for (int k = 0; k < emb_siz; k++) {
        *q++ = (*b++) + sdot(x, w, emb_siz);
        w += emb_siz;
      }
      // kv[i] = KVx
      float *kv = &kvbuf(i, 0);
      for (int k = 0; k < 2 * emb_siz; k++) {
        *kv++ = (*b++) + sdot(x, w, emb_siz);
        w += emb_siz;
      }
    }

    {
      float *att = attnbuf.data;
      for (int j = 0; j <= i; j++) {
        float *qk = qbuf.data;
        float *kk = &kvbuf(j, 0);
        for (int h = 0; h < num_heads; h++) {
          *att++ = sdot(qk, kk, head_siz) * attn_scale;
          qk += head_siz;
          kk += head_siz;
        }
      }
    }

    // att = softmax(att)
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
    {
      ybuf.zero();
      float *att = attnbuf.data;
      for (int j = 0; j <= i; j++) {
        float *y = ybuf.data;
        float *v = &kvbuf(j, emb_siz); // pick out the value vector from the key-value buf
        for (int h = 0; h < num_heads; h++) {
          saxpy(head_siz, *att++, v, y);
          v += head_siz;
          y += head_siz;
        }
      }
    }

    // matmul the projection and sum into input
    // input += c_proj_weight @ ybuf + c_proj_bias
    {
      float *w = c_proj_weight.data;
      float *y = ybuf.data;
      float *o = out.data;
      for (int j = 0; j < emb_siz; j++) {
        *o++ += c_proj_bias[j] + sdot(y, w, emb_siz);
        w += emb_siz;
      }
    }
  }
};

struct LayerNorm {
  Tensorf<1> bias;
  Tensorf<1> weight;

  void apply(Tensorf<1> &out, const Tensorf<1> &in) {
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

struct MLPBlock {
  // two-layer MLP
  Tensorf<1> c_fc_bias;
  Tensorf<2> c_fc_weight;
  Tensorf<1> c_proj_bias;
  Tensorf<2> c_proj_weight;

  // x += proj(gelu(fc(x)))
  void apply(const Tensorf<1> &out, const Tensorf<1> &in) {
    int hidden_dim = c_fc_bias.shape[0];
    Tensorf<1> hbuf(hidden_dim);
    // fc part of block
    // input += mlp_c_proj_weight @ gelu(mlp_c_fc_weight @ xbuf + mlp_c_fc_bias) + mlp_c_proj_bias
    {
      float *fc_w = c_fc_weight.data;
      float *x = in.data;
      float *h = hbuf.data;
      for (int j = 0; j < hidden_dim; j++) {
        float sum = c_fc_bias[j] + sdot(x, fc_w, in.shape[0]);
        float gelu = sum * 0.5 * (1.0 + tanh(0.7978845608028654 * (sum + 0.044715 * sum * sum * sum)));
        *h++ = gelu;
        fc_w += in.shape[0];
      }
    }
    // matmul the projection and sum into input
    {
      float *proj_w = c_proj_weight.data;
      float *o = out.data;
      float *h = hbuf.data;
      for (int j = 0; j < in.shape[0]; j++) {
        float sum = c_proj_bias[j] + sdot(h, proj_w, hidden_dim);
        *o++ += sum;
        proj_w += hidden_dim;
      }
    }
  }
};

struct TransformerBlock {
  // combined key, query, value
  CausalSelfAttention attn;
  LayerNorm ln_1, ln_2;
  MLPBlock mlp;

  void apply(const Tensorf<1> &x, const Tensorf<2> &kvbuf, int i) {
    Tensorf<1> xbuf(x.shape[0]);
    // x += attn(ln_1(x), kvbuf, i)
    ln_1.apply(xbuf, x);
    attn.apply(x, xbuf, kvbuf, i);
    // x += proj(gelu(fc(ln_2(x))))
    ln_2.apply(xbuf, x);
    mlp.apply(x, xbuf);
  }
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
    m.h[i].mlp.c_fc_bias = fetch_layer_weights<1>(j, addr, i, "mlp.c_fc.bias");
    m.h[i].mlp.c_fc_weight = *fetch_layer_weights<2>(j, addr, i, "mlp.c_fc.weight").TransposedCopy();
    m.h[i].mlp.c_proj_bias = fetch_layer_weights<1>(j, addr, i, "mlp.c_proj.bias");
    m.h[i].mlp.c_proj_weight = *fetch_layer_weights<2>(j, addr, i, "mlp.c_proj.weight").TransposedCopy();
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
    Tensorf<3> kvbuf(12, N, 2*m.embedding_dim);

    for (int i = 0; i < N; i++) {
      float sum1 = 0;
      float sum2 = 0;
      for (int j = 0; j < m.embedding_dim; j++) {
        input(i, j) = m.wte_weight(test_vector[i], j) + m.wpe_weight(i, j);
      }
    }

    Tensorf<1> logits(m.ntokens);

    for (int j = 0; j < N; j++) {
      auto input_j = input.slice(j);
      for (int l = 0; l < 12; l++) {
        printf("--- layer %d ---\n", l);
        m.h[l].apply(input_j, kvbuf.slice(l), j);
        printf("x(token=%d, layer=%d):\n", j, l); input_j.show();
      }
      // at this point we could apply lm_head but we only really need it for prediction
    }

    {
      Tensorf<1> ybuf(m.embedding_dim);
      // finally, layernorm and dot with embedding matrix
      for (int i = N-1; i < N; i++) {
        m.ln_f.apply(ybuf, input.slice(i));
        float *w = m.wte_weight.data;
        int largmax = 0;
        for (int j = 0; j < m.ntokens; j++) {
          logits[j] = sdot(ybuf.data, w, m.embedding_dim);
          w += m.embedding_dim;
          if (logits[j] > logits[largmax]) {
            largmax = j;
          }
        }
        printf("logits: "); logits.show();
        printf("argmax: %d (%s) = %f\n", largmax, decoder.vocab_[largmax].c_str(), logits[largmax]);
      }
    }
  }
  printf("expected result: x tensor([-3.4188, -1.0318, -3.5397,  5.2943,  3.9251, -2.0164, -2.1934, -2.5857, 0.5539, -4.0938])\n");

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  printf("elapsed: %f\n", elapsed);
}
