#pragma once

#include "tensor.h"

extern void normalize_mat_layernorm(Tensorf<2> &mat, Tensorf<1> &ln_bias,
                                    Tensorf<1> &ln_weight);

struct CausalSelfAttention {
  int num_heads;
  Tensorf<1> c_attn_bias;
  Tensorf<2> c_attn_weight;
  Tensorf<1> c_proj_bias;
  Tensorf<2> c_proj_weight;

  // adds self-attention(x, kvbuf) to x at token index i
  // kvbuf is a buffer of shape <tokens, 2*embedding_dim>
  // (modifies kvbuf[i], reads kvbuf[:i-1])
  void apply(const Tensorf<1> &out, const Tensorf<1> &xbuf, int i, const Tensorf<2> &kvbuf);
};

struct LayerNorm {
  Tensorf<1> bias;
  Tensorf<1> weight;

  void apply(Tensorf<1> &out, const Tensorf<1> &in);
};

struct MLPBlock {
  // two-layer MLP
  Tensorf<1> c_fc_bias;
  Tensorf<2> c_fc_weight;
  Tensorf<1> c_proj_bias;
  Tensorf<2> c_proj_weight;

  // x += proj(gelu(fc(x)))
  void apply(const Tensorf<1> &out, const Tensorf<1> &in);
};

struct TransformerBlock {
  // combined key, query, value
  CausalSelfAttention attn;
  LayerNorm ln_1, ln_2;
  MLPBlock mlp;

  void normalize() {
    // normalize weight matrices to -1..1, fusing with the preceding layernorm

    // this can only be done on c_attn_weight and c_fc_weight without
    // introducing two extra scaling vectors but let's start with those.
  }

  void apply(const Tensorf<1> &x, int i, const Tensorf<2> &kvbuf) {
    Tensorf<1> xbuf(x.shape[0]);
    // x += attn(ln_1(x), kvbuf, i)
    ln_1.apply(xbuf, x);
    attn.apply(x, xbuf, i, kvbuf);
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

  char *mmap_data;
  size_t mmap_siz;

  Tensorf<2> wte_weight;
  Tensorf<2> wpe_weight;
  LayerNorm ln_f;

  TransformerBlock *h;

  Model() {
    h = NULL;
    mmap_data = NULL;
  }

  ~Model();

  void to_device() {
    wte_weight.copyToDevice();
    wpe_weight.copyToDevice();
    ln_f.bias.copyToDevice();
    ln_f.weight.copyToDevice();
    for (int i = 0; i < 12; i++) {
      h[i].attn.c_attn_bias.copyToDevice();
      h[i].attn.c_attn_weight.copyToDevice();
      h[i].attn.c_proj_bias.copyToDevice();
      h[i].attn.c_proj_weight.copyToDevice();
      h[i].ln_1.bias.copyToDevice();
      h[i].ln_1.weight.copyToDevice();
      h[i].ln_2.bias.copyToDevice();
      h[i].ln_2.weight.copyToDevice();
      h[i].mlp.c_fc_bias.copyToDevice();
      h[i].mlp.c_fc_weight.copyToDevice();
      h[i].mlp.c_proj_bias.copyToDevice();
      h[i].mlp.c_proj_weight.copyToDevice();
    }
  }

  void apply_transformer(int token_id, int input_pos, const Tensorf<3> &kvbuf,
             const Tensorf<1> &emb_out);

  void apply_lm_head(Tensorf<1> &emb_in, Tensorf<1> &logits);
};

int sample_logits(float sampling_temperature, float uniform_sample, Tensorf<1> &logits);
float cross_entropy(const Tensorf<1> &logits, int index);
