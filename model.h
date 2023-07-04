#pragma once

#include "tensor.h"

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

  Tensorf<2> wte_weight;
  Tensorf<2> wpe_weight;
  LayerNorm ln_f;

  TransformerBlock *h;

  void apply(int token_id, int input_pos, const Tensorf<3> &kvbuf,
             const Tensorf<1> &emb_out) {
    for (int k = 0; k < embedding_dim; k++) {
      emb_out[k] = wte_weight(token_id, k) + wpe_weight(input_pos, k);
    }
    for (int layer = 0; layer < 12; layer++) {
      h[layer].apply(emb_out, input_pos, kvbuf.slice(layer));
    }
  }

  int sample_head(Tensorf<1> &emb_in, float sampling_temperature,
                  float uniform_sample, Tensorf<1> &logits);
};
