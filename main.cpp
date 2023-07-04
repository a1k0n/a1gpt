#include <math.h>
#include <stdio.h>
#include <time.h>

#include <string>

#include "bpe.h"
#include "blas.h"
#include "tensor.h"
#include "model.h"

extern bool load_gpt2_model(Model &m);

const char *DEFAULT_PROMPT =
    "In a shocking finding, scientist discovered a herd of unicorns living in "
    "a remote, previously unexplored valley, in the Andes Mountains. Even more "
    "surprising to the researchers was the fact that the unicorns spoke "
    "perfect English.";

void usage() {
  fprintf(stderr, "Usage: ./gpt2 [-s seed] [-t sampling_temperature] [-p prompt]\n");
  fprintf(stderr, "  -s seed: random seed (default: time(NULL))\n");
  fprintf(stderr, "  -t sampling_temperature: temperature for sampling (default: 1.0)\n");
  fprintf(stderr, "  -p prompt: prompt to start with (default: English-speaking unicorns)\n");
  exit(1);
}

int main(int argc, char **argv) {
  unsigned int seed = time(NULL);
  float sampling_temperature = 1.0;
  const char *prompt = DEFAULT_PROMPT;

  /* parse flags */
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      switch (argv[i][1]) {
        case 's':
          seed = atoi(argv[++i]);
          break;
        case 't':
          sampling_temperature = atof(argv[++i]);
          break;
        case 'p':
          prompt = argv[++i];
          break;
        case 'h':
          usage();
          break;
        default:
          fprintf(stderr, "Unknown flag: %s\n", argv[i]);
          usage();
      }
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      usage();
    }
  }

  srand(seed);

  BPEDecoder decoder;
  if (!decoder.Init("model/vocab.bin")) {
    if (!decoder.Init("../model/vocab.bin")) {
      fprintf(stderr, "Failed to init decoder from ../model/vocab.bin\n");
      exit(1);
    }
  }

  BPEEncoder encoder;
  encoder.Init(decoder.vocab_);

  fprintf(stderr, "a1gpt seed=%u sampling_temperature=%f\n", seed, sampling_temperature);

  Model m;
  if (!load_gpt2_model(m)) {
    fprintf(stderr, "Failed to load model\n");
    exit(1);
  }

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  {
    const int ctx_max = 1024;
    Tensorf<3> kvbuf(12, ctx_max, 2*m.embedding_dim);
    Tensorf<1> ybuf(m.embedding_dim);
    Tensorf<1> logitbuf(m.ntokens);
    int ctx_tokens[1024];
    int N = encoder.Encode(prompt, ctx_tokens, 1024);
    if (N == 0) {
      ctx_tokens[0] = 50256; // <|endoftext|>
      N = 1;
    }
    {
      printf("encoded prompt: ");
      for (int i = 0; i < N; i++) {
        printf("%d ", ctx_tokens[i]);
      }
      char buf[4096];
      decoder.Decode(ctx_tokens, N, buf, 4096);
      printf("\nGenerating:\n%s", buf);
      fflush(stdout);
    }
    for (int j = 0; j < ctx_max; j++) {
      m.apply(ctx_tokens[j], j, kvbuf, ybuf);

      if (j < N - 1) {
        // no need to run lm_head on the prompt; afterwards, we start to
        // generate
        continue;
      }

      float r = (float)rand() / RAND_MAX;
      int sampled_token = m.sample_head(ybuf, sampling_temperature, r, logitbuf);

      ctx_tokens[j + 1] = sampled_token;

      // printf("logits: "); logits.show();
      // printf("argmax: %d (%s) = %f\n", largmax,
      // decoder.vocab_[largmax].c_str(), logits[largmax]);
      {
        std::string &token = decoder.vocab_[sampled_token];
        fwrite(token.c_str(), 1, token.size(), stdout);
        fflush(stdout);
      }
    }
    printf("\n");
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  printf("elapsed: %fs, %fms per token\n", elapsed, 1000 * elapsed / 1024.);
}
