#include <stdio.h>

extern __shared__ float shared_buf[];

__device__ void sumSharedMem(float *shared, int index, int siz) {
    __syncthreads();
    for (int i = 1; i < siz; i <<= 1) {
        if ((index & i) == 0 && index + i < siz) {
            shared[index] += shared[index + i];
        }
        __syncthreads();
    }
}

__device__ void sumSharedMem2(float* shared, int index, int siz) {
    __syncthreads();
    for (int i = 1; i < siz; i <<= 1) {
        if ((index & i) == 0 && index + i < siz) {
            shared[index] += shared[index + i];
            shared[index + siz] += shared[index + i + siz];
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

__global__ void layerNormKernel(float* output, float* gamma, float* beta, float* input) {
    const int index = threadIdx.x;
    const int siz = blockDim.x;

    // shared memory reduction of mean and variance (requires 10 iterations for any embedding >512, <=1024)
    float *shared_mean = shared_buf;
    float *shared_var = shared_buf + siz;

    shared_mean[index] = input[index];
    shared_var[index] = input[index] * input[index];

    // sums both mean and variance assuming they are contiguous in shared
    // memory, which they are
    sumSharedMem2(shared_buf, index, siz);

    float mean = shared_mean[0] / siz;
    float variance = shared_var[0] / siz - mean * mean;
    const float eps = 1e-5f;
    float stddev = sqrt(variance + eps);  // Small constant for numerical stability

    // Normalize input and store in output
    output[index] = (input[index] - mean) / stddev * gamma[index] + beta[index];
}

void layerNorm(float* output, int embedding_dim, float* gamma, float* beta, float* input) {
    size_t shared_siz = embedding_dim * 2 * sizeof(float);
    layerNormKernel<<<1, embedding_dim, shared_siz>>>(output, gamma, beta, input);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in layerNormKernel\n");
        abort();
    }
}

// compute q and kv by applying weight matrix / bias vector
__global__ void qkvKernel(float* xbuf, float *qbuf, float* kvbuf,
                          float* attn_weight, float* attn_bias) {
    const int k = threadIdx.x;
    const int j = blockIdx.x;
    const int siz = blockDim.x;

    // compute q[j] and kv[j] in parallel
    shared_buf[k] = xbuf[k] * attn_weight[j * siz + k];
    sumSharedMem(shared_buf, k, siz);
    // accumulate and sum
    if (j < siz) {
        qbuf[j] = shared_buf[0] + attn_bias[j];
    } else {
        kvbuf[j - siz] = shared_buf[0] + attn_bias[j];
    }
}

void qkv(int kv_idx, float* xbuf, float *qbuf, float* kvbuf,
         float* attn_weight, float* attn_bias, int embedding_dim) {

    size_t sharedbuf_siz = embedding_dim * sizeof(float);
    qkvKernel<<<embedding_dim * 3, embedding_dim, sharedbuf_siz>>>(
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

__global__ void gemvKernel(float *y, float *A, float *x, float *b) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockDim.x;

    shared_buf[i] = A[j * k + i] * x[i];
    sumSharedMem(shared_buf, i, k);
    if (i == 0) {
        float z = shared_buf[0];
        if (b) {
            z += b[j];
        }
        y[j] = z;
    }
}

__global__ void gemvSumKernel(float *y, float *A, float *x, float *b) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockDim.x;

    shared_buf[i] = A[j * k + i] * x[i];
    sumSharedMem(shared_buf, i, k);
    if (i == 0) {
        y[j] += shared_buf[0] + b[j];
    }
}

__global__ void gemvSumKernelN(float *y, float *A, float *x, float *b, int n) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockDim.x;

    A += (j*k + i) * n;
    x += i*n;
    float sum = 0;
    while (n--) {
        sum += *A++ * *x++;
    }
    shared_buf[i] = sum;
    sumSharedMem(shared_buf, i, k);
    if (i == 0) {
        y[j] += shared_buf[0] + b[j];
    }
}

__global__ void gemvGeluKernel(float *y, float *A, float *x, float *b) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockDim.x;

    shared_buf[i] = A[j * k + i] * x[i];
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
    size_t shared_siz = k * sizeof(float);
    gemvKernel<<<m, k, shared_siz>>>(y, A, x, b);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in gemvKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}

void gemvSum(float *y, float *A, float *x, float *b, int m, int k) {
    if (k > 1024) {
        int stride = 1 + (k >> 10);
        int sub_k = k / stride;
        size_t shared_siz = sub_k * sizeof(float);
        gemvSumKernelN<<<m, sub_k, shared_siz>>>(y, A, x, b, stride);
    } else {
        size_t shared_siz = k * sizeof(float);
        gemvSumKernel<<<m, k, shared_siz>>>(y, A, x, b);
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in gemvKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}

void gemvGelu(float *y, float *A, float *x, float *b, int m, int k) {
    size_t shared_siz = k * sizeof(float);
    gemvGeluKernel<<<m, k, shared_siz>>>(y, A, x, b);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in gemvKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}