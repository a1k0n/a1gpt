#include <math.h>
#include <sys/mman.h>

#include "kernel.h"
#include "model.h"

Model::~Model() {
  delete[] h;
  if (mmap_data) {
    munmap(mmap_data, mmap_siz);
  }
}

void CausalSelfAttention::apply(const Tensorf<1> &out, const Tensorf<1> &xbuf,
                                int i, const Tensorf<2> &kvbuf) {
  const int emb_siz = 768;
  const int num_heads = 12;
  const int head_siz = 64;

  assert(xbuf.shape[0] == emb_siz);
  assert(emb_siz / num_heads == head_siz);

  Tensorf<1> ybuf(emb_siz);
  Tensorf<1> qbuf(emb_siz);

  // qbuf, kvbuf[i] = qkv(QKV, xbuf)
  qkv(i, xbuf.gpu_data, qbuf.gpu_data, kvbuf.gpu_data, c_attn_weight.gpu_data, c_attn_bias.gpu_data, emb_siz);
  // y = attn(qbuf, kvbuf[i])
  attn(i, ybuf.gpu_data, qbuf.gpu_data, kvbuf.gpu_data, emb_siz, num_heads);
  // x += proj(y)
  gemvSum(out.gpu_data, c_proj_weight.gpu_data,
          ybuf.gpu_data, c_proj_bias.gpu_data, emb_siz,
          emb_siz);
}

void LayerNorm::apply(Tensorf<1> &out, const Tensorf<1> &in) {
  layerNorm(out.gpu_data, out.shape[0], weight.gpu_data,
            bias.gpu_data, in.gpu_data);
}

void MLPBlock::apply(const Tensorf<1> &out, const Tensorf<1> &in) {
  int hidden_dim = 4*768; 
  int emb_dim = 768;

  assert(in.shape[0] == emb_dim);
  assert(c_fc_bias.shape[0] == hidden_dim);

  Tensorf<1> hbuf(hidden_dim);

  // h = gelu(mlp.fc(y))
  gemvGelu(hbuf.gpu_data, c_fc_weight.gpu_data, in.gpu_data, c_fc_bias.gpu_data,
           emb_dim * 4, emb_dim);

  // x += mlp.proj(h)
  gemvSum(out.gpu_data, c_proj_weight.gpu_data, hbuf.gpu_data,
          c_proj_bias.gpu_data, emb_dim, emb_dim * 4);
}

void Model::apply_transformer(int token_id, int input_pos,
                              const Tensorf<3> &kvbuf,
                              const Tensorf<1> &emb_out) {
  loadEmbedding(emb_out.gpu_data, token_id, input_pos, embedding_dim,
                wte_weight.gpu_data, wpe_weight.gpu_data);
  for (int layer = 0; layer < 12; layer++) {
    h[layer].apply(emb_out, input_pos, kvbuf.slice(layer));
  }
}

void Model::apply_lm_head(Tensorf<1> &emb_in, Tensorf<1> &logits) {
  assert(emb_in.shape[0] == embedding_dim);
  // layernorm and dot with embedding matrix
  ln_f.apply(emb_in, emb_in);
  const int ntokens = logits.shape[0];
  float m = -INFINITY;
  gemv(logits.gpu_data, wte_weight.gpu_data, emb_in.gpu_data, NULL, ntokens, embedding_dim);
  logits.copyToCpu();
  for (int j = 0; j < ntokens; j++) {
    if (logits[j] > m) {
      m = logits[j];
    }
  }
  // subtract max for numerical stability
  for (int j = 0; j < ntokens; j++) {
    logits[j] -= m;
  }
}

int sample_logits(float sampling_temperature, float uniform_sample, Tensorf<1> &logits) {
  // sample from logits (also normalizes logits to probabilities)
  int ntokens = logits.shape[0];
  float sum = 0;
  for (int j = 0; j < ntokens; j++) {
    logits[j] = expf(logits[j] / sampling_temperature);
    sum += logits[j];
  }
  for (int j = 0; j < ntokens; j++) {
    logits[j] /= sum;
  }
  float acc = 0;
  for (int j = 0; j < ntokens; j++) {
    acc += logits[j];
    if (acc >= uniform_sample) {
      return j;
    }
  }
  fprintf(stderr, "[sampling error? r=%f, acc=%f]\n", uniform_sample, acc);
  return 0;
}

float cross_entropy(const Tensorf<1> &logits, int index) {
  float sum = 0;
  // max has already been subtracted, so we just need the log sum
  for (int j = 0; j < logits.shape[0]; j++) {
    sum += expf(logits[j]);
  }
  return logits[index] - logf(sum);
}