#include <stdio.h>

extern __shared__ float shared_buf[];

__device__ void sumSharedMem(float *shared, int index, int siz) {
    shared += index;
    __syncthreads();
    for (int i = 1; i < siz; i <<= 1) {
        if ((index & i) == 0 && index + i < siz) {
            shared[0] += shared[i];
        }
        __syncthreads();
    }
}

__device__ void sumSharedMem2(float* shared, int index, int siz) {
    shared += index;
    __syncthreads();
    for (int i = 1; i < siz; i <<= 1) {
        if ((index & i) == 0 && index + i < siz) {
            shared[0] += shared[i];
            shared[siz] += shared[siz + i];
        }
        __syncthreads();
    }
}

// aggregate two partial softmax values
// m(x) = m([x1 x2]) = max(m(x1), m(x2))
// l(x) = l([x1 x2]) = e^(m(x1) - m(x)) * l(x1) + e^(m(x2) - m(x)) * l(x2)
// v(x) = v([x1 x2]) = e^(m(x1) - m(x)) * v(x1) + e^(m(x2) - m(x)) * v(x2)
__device__ inline void aggregateSoftmax4(float *m, float m2, float *l, float l2, float *v, const float *v2) {
    float newm = m2 > *m ? m2 : *m;
    float e1 = __expf(*m - newm);
    float e2 = __expf(m2 - newm);
    *l = e1* *l + e2*l2;
    *m = newm;
    v[0] = e1 * v[0] + e2 * v2[0];
    v[1] = e1 * v[1] + e2 * v2[1];
    v[2] = e1 * v[2] + e2 * v2[2];
    v[3] = e1 * v[3] + e2 * v2[3];
}

__device__ void aggregateSharedSoftmax4(float *mlv, int i, int stride, int index, int siz) {
    mlv += index*stride;
    __syncthreads();
    for (int j = 1; j < siz; j <<= 1, stride <<= 1) {
        if ((index & j) == 0 && index + j < siz) {
            aggregateSoftmax4(mlv, mlv[stride],                    // *m, m2
                              mlv + 1, mlv[stride + 1],            // *l, l2
                              mlv + 2 + i, mlv + stride + 2 + i);  // v[], v2[]
        }
        __syncthreads();
    }
}

__global__ void loadEmbeddingKernel(float *output, int token, int pos, int embeddingSize, float* wte, float *wpe) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < embeddingSize) {
        output[index] = wte[token * embeddingSize + index] + wpe[pos * embeddingSize + index];
    }
}

void loadEmbedding(float *output, int token, int pos, int embeddingSize, float* wte, float* wpe) {
    loadEmbeddingKernel<<<1, embeddingSize>>>(output, token, pos, embeddingSize, wte, wpe);
}

__global__ void layerNormKernel8(float* output, float* gamma, float* beta, float* input) {
    const int index = threadIdx.x;
    const int siz = blockDim.x;
    const int k = index*8;
    float *x = input + k;

    // shared memory reduction of mean and variance (requires 10 iterations for any embedding >512, <=1024)
    float *shared_mean = shared_buf;
    float *shared_var = shared_buf + siz;

    shared_mean[index] = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
    shared_var[index] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] +
                        x[4]*x[4] + x[5]*x[5] + x[6]*x[6] + x[7]*x[7];

    // sums both mean and variance assuming they are contiguous in shared
    // memory, which they are
    sumSharedMem2(shared_buf, index, siz);

    float mean = shared_mean[0] / (8*siz);
    float variance = shared_var[0] / (8*siz) - mean * mean;
    const float eps = 1e-5f;
    float stddev = sqrt(variance + eps);  // Small constant for numerical stability

    // Normalize input and store in output
    for (int i = 0; i < 8; i++) {
        output[k+i] = (x[i] - mean) / stddev * gamma[k+i] + beta[k+i];
    }
}

void layerNorm(float* output, int embedding_dim, float* gamma, float* beta, float* input) {
    size_t shared_siz = embedding_dim * 2 * sizeof(float) / 8;
    layerNormKernel8<<<1, embedding_dim / 8, shared_siz>>>(output, gamma, beta, input);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in layerNormKernel\n");
        abort();
    }
}

// compute q and kv by applying weight matrix / bias vector
__global__ void qkvKernel4(float* xbuf, float *qbuf, float* kvbuf,
                           float* attn_weight, float* attn_bias) {
    const int k = threadIdx.x;
    const int j = blockIdx.x;
    const int siz = blockDim.x*4;

    float *A = attn_weight + (j * siz) + k * 4;
    float *x = xbuf + k*4;

    // TODO: do sum within warp before accumulating across shared memory
    // using __shfl_down_sync (but we need to know our warp mask)

    // compute q[j] and kv[j] in parallel
    shared_buf[k] = A[0]*x[0] + A[1]*x[1] + A[2]*x[2] + A[3]*x[3];
    sumSharedMem(shared_buf, k, blockDim.x);
    // accumulate and sum
    if (j < siz) {
        qbuf[j] = shared_buf[0] + attn_bias[j];
    } else {
        kvbuf[j - siz] = shared_buf[0] + attn_bias[j];
    }
}

void qkv(int kv_idx, float* xbuf, float *qbuf, float* kvbuf,
         float* attn_weight, float* attn_bias, int embedding_dim) {

    size_t sharedbuf_siz = embedding_dim * sizeof(float) / 4;
    qkvKernel4<<<embedding_dim * 3, embedding_dim / 4, sharedbuf_siz>>>(
        xbuf, qbuf, kvbuf + kv_idx * 2*embedding_dim, attn_weight, attn_bias);

    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in qkvKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}

// <<<num_heads, head_size, head_size * sizeof(float)>>>
__global__ void attnKernel(int kv_idx, float *ybuf, float *qbuf, float *kvbuf, int emb_siz) {
    int k = threadIdx.x;
    // which attention head are we in?
    int h = blockIdx.x;
    int head_siz = blockDim.x;

    // offset inputs/outputs by our attention head position
    qbuf += h * head_siz;
    ybuf += h * head_siz;
    kvbuf += h * head_siz;

    float *shared_a = shared_buf;
    float attn_scale = 1.0f / sqrtf(head_siz);

    // initially, only one value to pick from, so that's our output value
    ybuf[k] = kvbuf[k + emb_siz];

    // compute q*k for first kv within our own attention head
    shared_a[k] = qbuf[k] * kvbuf[k];
    sumSharedMem(shared_a, k, head_siz);
    float a = shared_a[0] * attn_scale;
    float m = a;  // maximum softmax value for our attention head
    float l = 1;  // denominator sum for our attention head

    for (int i = 1; i <= kv_idx; i++) {
        // move on to next kv
        kvbuf += emb_siz*2;
        // compute q*k for the others and aggregate
        shared_a[k] = qbuf[k] * kvbuf[k];
        sumSharedMem(shared_a, k, head_siz);
        float a = shared_a[0] * attn_scale;
        if (a > m) {  // we won't have branch divergence here
            float e = expf(m - a);  // < 1.0
            ybuf[k] = kvbuf[k + emb_siz] + e * ybuf[k];
            l = 1 + e * l;
            m = a;  // new maximum
        } else {
            float e = expf(a - m); // < 1.0
            ybuf[k] += e * kvbuf[k+emb_siz];
            l += e;
            // m is still the maximum
        }
    }
    // rescale y by 1/l
    ybuf[k] /= l;
}

void attn(int kv_idx, float *xbuf, float *qbuf, float *kvbuf, int emb_siz, int num_heads) {
    int head_siz = emb_siz / num_heads;
    size_t sharedbuf_siz = head_siz * sizeof(float);
    attnKernel<<<num_heads, head_siz, sharedbuf_siz>>>(kv_idx, xbuf, qbuf, kvbuf, emb_siz);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in attnKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}

__global__ void attn2Kernel(int kv_idx, float *out_mlv, float *qbuf, float *kvbuf, int emb_siz) {
    int i = threadIdx.x*4;
    int j = threadIdx.y;
    int h = blockIdx.x;
    int b = blockIdx.y;
    int head_siz = blockDim.x*4;
    int nblocks = gridDim.y;
    int nkvs = kv_idx+1;

    // which key/value is this thread looking at?
    int kvs_per_block = blockDim.y;
    int kv_offset = b*kvs_per_block + j;
    if (kv_offset >= nkvs) {
        return;
    }
    int kvs_this_block = kvs_per_block;
    // ideally (b+1) * kvs_per_block == nkvs on the last block, but
    // we might have some leftovers on the last block
    if ((b+1)*kvs_per_block > nkvs) {
        kvs_this_block = nkvs - b*kvs_per_block;
    }

    // we are computing attention for 0..kv_idx but in blocks

    // offset inputs/outputs by our attention head position
    qbuf += h * head_siz;
    kvbuf += h * head_siz + kv_offset * emb_siz * 2;
    // we are going to compute q*k[j, i:i+4] in this kernel,
    // then sum up a = q*k among our thread group j
    float *shared_a = shared_buf + j*(head_siz+2);
    float attn_scale = 1.0f / sqrtf(head_siz);
    {
        float *q = qbuf + i;
        float *k = kvbuf + i;
        float z = q[0]*k[0] + q[1]*k[1] + q[2]*k[2] + q[3]*k[3];
        shared_a[threadIdx.x] = z;
        sumSharedMem(shared_a, threadIdx.x, blockDim.x);
    }
    float a = shared_a[0] * attn_scale;

    // we've computed a, so re-use the shared buf for m, l, v aggregation
    float *shared_m = shared_a;
    float *shared_l = shared_a+1;
    float *shared_v = shared_a+2;

    // init our own m, l, v and aggregate them between all thread groups in the block
    *shared_m = a;
    *shared_l = 1;
    shared_v[i] = kvbuf[i + emb_siz];
    shared_v[i+1] = kvbuf[i + emb_siz + 1];
    shared_v[i+2] = kvbuf[i + emb_siz + 2];
    shared_v[i+3] = kvbuf[i + emb_siz + 3];

    // now aggregate all mlvs
    aggregateSharedSoftmax4(shared_buf,       // mlv
                            i,                // inner index
                            head_siz + 2,     // stride
                            j,                // outer index
                            kvs_this_block);  // siz

    // once we've aggregated all mlv for this block, the first kv threads can write the output
    if (j == 0) {
        // mlv layout:
        // head 0: <<block 0: l, m, v>, <block 1: l, m, v>, ...>>
        // a future reduction step will combine softmax across blocks
        out_mlv += (h * nblocks + b) * (2 + head_siz);
        float *out_m = out_mlv;
        float *out_l = out_mlv + 1;
        float *out_v = out_mlv + 2;

        *out_m = *shared_m;
        *out_l = *shared_l;
        out_v[i] = shared_v[i];
        out_v[i+1] = shared_v[i+1];
        out_v[i+2] = shared_v[i+2];
        out_v[i+3] = shared_v[i+3];
    }
}

__global__ void attn2AggregateKernel(int nblocks, float *xbuf, float *mlv) {
    // aggregate across blocks, solving final value for our attention head, and
    // place the result in xbuf
    // one block per head
    int i = threadIdx.x*4;
    int j = threadIdx.y;
    int h = blockIdx.x;
    int head_siz = blockDim.x*4;

    // mlv array for this head
    mlv += (head_siz + 2) * (h * nblocks + j*2);

    float *shared_mlv = shared_buf;
    int shr_off = j * (head_siz + 2);
    if (i == 0) {
        shared_mlv[shr_off] = mlv[0];
        shared_mlv[shr_off + 1] = mlv[1];
    }
    shared_mlv[shr_off + 2 + i] = mlv[2 + i];
    shared_mlv[shr_off + 2 + i + 1] = mlv[2 + i + 1];
    shared_mlv[shr_off + 2 + i + 2] = mlv[2 + i + 2];
    shared_mlv[shr_off + 2 + i + 3] = mlv[2 + i + 3];
    if (j*2+1 < nblocks) {
        // get the next odd mlv value which we will immediately merge into our even one
        float *mlv2 = mlv + (head_siz + 2);
        aggregateSoftmax4(shared_mlv + shr_off, mlv2[0],      // *m, m2
                          shared_mlv + shr_off + 1, mlv2[1],  // *l, l2
                          shared_mlv + shr_off + 2 + i,       // v[]
                          mlv2 + 2 + i);                      // v2[]
    }
    aggregateSharedSoftmax4(shared_mlv, i, head_siz + 2, j, blockDim.y);

    // copy shared_mlv+2 into our xbuf head
    if (j == 0) {
        float scale = 1.0 / shared_mlv[1];
        xbuf[h * head_siz + i] = shared_mlv[2 + i] * scale;
        xbuf[h * head_siz + i + 1] = shared_mlv[2 + i + 1] * scale;
        xbuf[h * head_siz + i + 2] = shared_mlv[2 + i + 2] * scale;
        xbuf[h * head_siz + i + 3] = shared_mlv[2 + i + 3] * scale;
    }
}

void attn2(int kv_idx, float *xbuf, float *qbuf, float *kvbuf, int emb_siz, int num_heads) {
    int head_siz = emb_siz / num_heads;
    // calc how many kvs we can do in one block based on the shared memory available
    int max_sharedbuf_siz;
    if (cudaDeviceGetAttribute(&max_sharedbuf_siz, cudaDevAttrMaxSharedMemoryPerBlock, 0) != cudaSuccess) {
        fprintf(stderr, "Error getting max shared memory per block\n");
        abort();
    }
    int nblocks = 8;
    if ((kv_idx+1) < nblocks) {
        nblocks = kv_idx+1;
    }
    int threads_per_kv = head_siz / 4;
    int max_kvs_per_block = 512 / threads_per_kv;
    int kvs_per_block = 1 + kv_idx / nblocks;
    if (kvs_per_block > max_kvs_per_block) {
        kvs_per_block = max_kvs_per_block;
    }
    nblocks = 1 + kv_idx / kvs_per_block;
    int threads_per_block = threads_per_kv * kvs_per_block;
    if (threads_per_block > 1024) {
        // this could be handled by adding more blocks, but in GPT-2 we don't
        // have enough context length to justify it
        fprintf(stderr, "Error: too many threads per block\n");
        abort();
    }
    int sharedbuf_siz = (2 + head_siz) * kvs_per_block * sizeof(float);

    float *tmpBuf;
    // tmpbuf is laid out like
    // head 0: <<block 0: l, m, v>, <block 1: l, m, v>, ...>>
    if (cudaMalloc(&tmpBuf, nblocks * num_heads * (2 + head_siz) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error allocating temporary buffer\n");
        abort();
    }
    cudaMemset(tmpBuf, 0, nblocks * num_heads * (2 + head_siz) * sizeof(float));

    attn2Kernel<<<dim3(num_heads, nblocks), dim3(threads_per_kv, kvs_per_block), sharedbuf_siz>>>(
        kv_idx, tmpBuf, qbuf, kvbuf, emb_siz);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in attnKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }

    int naggr_groups = (1 + nblocks) / 2;
    int shared2_siz = (2 + head_siz) * naggr_groups * sizeof(float);
    // this should be impossible as the largest possible number of blocks is 16 with 1024 context
    if (shared2_siz > max_sharedbuf_siz) {
        fprintf(stderr, "Error: too much shared memory required for attn2AggregateKernel\n");
        abort();
    }
    attn2AggregateKernel<<<num_heads, dim3(threads_per_kv, naggr_groups), shared2_siz>>>(nblocks, xbuf, tmpBuf);

    cudaFree(tmpBuf);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in attnKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}

__global__ void gemvKernel4(float *y, float *A, float *x, float *b) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockDim.x;

    A += (j*k + i) * 4;
    x += i*4;

    shared_buf[i] = A[0]*x[0] + A[1]*x[1] + A[2]*x[2] + A[3]*x[3];
    sumSharedMem(shared_buf, i, k);
    if (i == 0) {
        float z = shared_buf[0];
        if (b) {
            z += b[j];
        }
        y[j] = z;
    }
}

__global__ void gemvSumKernel4(float *y, float *A, float *x, float *b) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockDim.x;

    A += (j*k + i) * 4;
    x += i*4;

    shared_buf[i] = A[0]*x[0] + A[1]*x[1] + A[2]*x[2] + A[3]*x[3];
    sumSharedMem(shared_buf, i, k);
    if (i == 0) {
        y[j] += shared_buf[0] + b[j];
    }
}

__global__ void gemvGeluKernel4(float *y, float *A, float *x, float *b) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockDim.x;

    A += 4*(j*k + i);
    x += 4*i;

    shared_buf[i] = A[0]*x[0] + A[1]*x[1] + A[2]*x[2] + A[3]*x[3];
    sumSharedMem(shared_buf, i, k);
    if (i == 0) {
        float z = shared_buf[0] + b[j];
        //y[j] = 0.5f * z * (1.0f + tanhf(0.7978845608028654f * (z + 0.044715f * z * z * z)));
        y[j] = z / (1 + expf(-1.702*z));
    }
}

// y = A * x + b
// A is m x k
// x is k x 1
// b is m x 1
void gemv(float *y, float *A, float *x, float *b, int m, int k) {
    size_t shared_siz = k * sizeof(float) / 4;
    gemvKernel4<<<m, k / 4, shared_siz>>>(y, A, x, b);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in gemvKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}

void gemvSum(float *y, float *A, float *x, float *b, int m, int k) {
    size_t shared_siz = k * sizeof(float) / 4;
    gemvSumKernel4<<<m, k / 4, shared_siz>>>(y, A, x, b);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in gemvKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}

void gemvGelu(float *y, float *A, float *x, float *b, int m, int k) {
    size_t shared_siz = k * sizeof(float) / 4;
    gemvGeluKernel4<<<m, k/4, shared_siz>>>(y, A, x, b);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in gemvKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}