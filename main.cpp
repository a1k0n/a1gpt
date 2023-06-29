#include <math.h>
#include <stdio.h>

#include <string>

#include "bpe.h"
#include "blas.h"
#include "tensor.h"
#include "model.h"

/*
extern float* gpuTransferFloats(float *data, int size);
extern void gpuDumpMemoryInfo();
*/

int main(int argc, char **argv) {
  BPEDecoder decoder;
  if (!decoder.Init("vocab.bin")) {
    fprintf(stderr, "Failed to init decoder from vocab.bin\n");
    exit(1);
  }

  BPEEncoder encoder;
  encoder.Init(decoder.vocab_);

  Model m;
  if (!m.Load("model.safetensors")) {
    fprintf(stderr, "Failed to load model\n");
    exit(1);
  }
  // tokenize("The rain in spain falls mainly on the")
  // benchmark 100 iterations
  // get t0 first
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  float sampling_temperature = 0.9;
  srand(0);

  const char *prompt = "The rain in spain falls mainly on the";
  if (argc > 1) {
    prompt = argv[1];
  }
  {
    int input_vector[1024];
    int N = encoder.Encode(prompt, input_vector, 1024);
    int ctx_max = 1024;
    {
      printf("encoded prompt: ");
      for (int i = 0; i < N; i++) {
        printf("%d ", input_vector[i]);
      }
      char buf[4096];
      decoder.Decode(input_vector, N, buf, 4096);
      printf("\nGenerating:\n%s", buf);
      fflush(stdout);
    }
    Tensorf<3> kvbuf(12, ctx_max, 2*m.embedding_dim);
    Tensorf<1> logits(m.ntokens);
    Tensorf<1> input_buf(m.embedding_dim);
    for (int j = 0; j < ctx_max; j++) {
      float *input = input_buf.data;
      for (int k = 0; k < m.embedding_dim; k++) {
        input[k] = m.wte_weight(input_vector[j], k) + m.wpe_weight(j, k);
      }
      for (int l = 0; l < 12; l++) {
        m.h[l].apply(input_buf, kvbuf.slice(l), j);
      }
      // at this point we could apply lm_head but we only really need it for prediction

      if (j < N-1) {
        // no need to run lm_head on the prompt; afterwards, we start to generate
        continue;
      }
      Tensorf<1> ybuf(m.embedding_dim);
      // finally, layernorm and dot with embedding matrix
      {
        m.ln_f.apply(ybuf, input_buf);
        float *w = m.wte_weight.data;
        int largmax = 0;
        for (int j = 0; j < m.ntokens; j++) {
          logits[j] = sdot(ybuf.data, w, m.embedding_dim);
          w += m.embedding_dim;
          if (logits[j] > logits[largmax]) {
            largmax = j;
          }
        }

        // sample from logits
        int sampled_token = largmax;
        float sum = 0;
        for (int j = 0; j < m.ntokens; j++) {
          logits[j] = expf(logits[j] / sampling_temperature);
          sum += logits[j];
        }
        for (int j = 0; j < m.ntokens; j++) {
          logits[j] /= sum;
        }
        float r = (float)rand() / RAND_MAX;
        float acc = 0;
        for (int j = 0; j < m.ntokens; j++) {
          acc += logits[j];
          if (r < acc) {
            sampled_token = j;
            break;
          }
        }

        input_vector[j+1] = sampled_token;

        // printf("logits: "); logits.show();
        // printf("argmax: %d (%s) = %f\n", largmax, decoder.vocab_[largmax].c_str(), logits[largmax]);
        {
          std::string& token = decoder.vocab_[sampled_token];
          fwrite(token.c_str(), 1, token.size(), stdout);
          fflush(stdout);
        }
      }
    }
    printf("\n");
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  printf("elapsed: %f\n", elapsed);
}
