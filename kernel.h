#pragma once

extern void loadEmbedding(float *output, int token, int pos, int embeddingSize, float* wte, float* wpe);
extern void layerNorm(float* output, int embedding_dim, float* gamma, float* beta, float* input);

extern void qkv(int kv_idx, float* xbuf, float *qbuf, float* kvbuf,
         float* attn_weight, float* attn_bias, int embedding_dim);

extern void gemv(float *y, float *A, float *x, float *b, int m, int k);
extern void gemvSum(float *y, float *A, float *x, float *b, int m, int k);
extern void gemvGelu(float *y, float *A, float *x, float *b, int m, int k);

extern void attn(int kv_idx, float *xbuf, float *qbuf, float *kvbuf, int emb_siz, int num_heads);
