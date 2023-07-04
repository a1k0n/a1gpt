#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <string>

#include "bpe.h"
#include "blas.h"
#include "tensor.h"
#include "model.h"

const int ctx_max = 1024;
extern bool load_gpt2_model(Model &m);
float sampling_temperature = 0.9;
float classifier_free_guidance = 1.0;

const char *DEFAULT_PROMPT =
    "In a shocking finding, scientist discovered a herd of unicorns living in "
    "a remote, previously unexplored valley, in the Andes Mountains. Even more "
    "surprising to the researchers was the fact that the unicorns spoke "
    "perfect English.";

void usage() {
  fprintf(stderr, "Usage: ./gpt2 [-s seed] [-t sampling_temperature] [-p prompt]\n");
  fprintf(stderr, "  -s seed: random seed (default: time(NULL))\n");
  fprintf(stderr, "  -t sampling_temperature: temperature for sampling (default: %0.2f)\n", sampling_temperature);
  fprintf(stderr, "  -p prompt: prompt to start with (default: English-speaking unicorns)\n");
  fprintf(stderr, "  -n ntokens: number of tokens to generate (default=max: %d)\n", ctx_max);
  fprintf(stderr, "  -c cfg_scale: classifier-free guidance scale; 1.0 means no CFG (default: %0.1f)\n", classifier_free_guidance);
  exit(1);
}

int main(int argc, char **argv) {
  unsigned int seed = time(NULL);
  const char *prompt = DEFAULT_PROMPT;
  int ntokens_gen = 1024;

  int c;
  while ((c = getopt(argc, argv, "s:t:p:n:c:h")) != -1) {
    switch (c) {
      case 's':
        seed = atoi(optarg);
        break;
      case 't':
        sampling_temperature = atof(optarg);
        break;
      case 'p':
        prompt = optarg;
        break;
      case 'n':
        ntokens_gen = atoi(optarg);
        if (ntokens_gen > ctx_max) {
          fprintf(stderr, "ERROR: ntokens must be <= %d\n", ctx_max);
          usage();
        }
        break;
      case 'c':
        classifier_free_guidance = atof(optarg);
        if (classifier_free_guidance <= 0) {
          fprintf(stderr, "ERROR: cfg_scale must be > 0\n");
          usage();
        }
        break;
      case 'h':
      default:
        usage();
        usage();
    }
  }

  // if there are any extra args, warn
  if (optind < argc) {
    fprintf(stderr, "ERROR: extra args %s... ignored; use -p \"prompt\" to set the prompt\n", argv[optind]);
    usage();
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

  Model m;
  if (!load_gpt2_model(m)) {
    fprintf(stderr, "Failed to load model\n");
    exit(1);
  }

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  {
    Tensorf<3> kvbuf(12, ctx_max, 2*m.embedding_dim);
    Tensorf<1> ybuf(m.embedding_dim);

    Tensorf<3> cfg_kvbuf(12, ctx_max, 2*m.embedding_dim);
    Tensorf<1> cfg_ybuf(m.embedding_dim);
    int cfg_ptr;

    // forbid generation of <|endoftext|> by cutting it out of the logit buffer (it's the last token)
    Tensorf<1> logitbuf(m.ntokens - 1);

    int ctx_tokens[ctx_max+1];
    // always start with <|endoftext|>
    ctx_tokens[0] = 50256; // <|endoftext|>
    int N = encoder.Encode(prompt, ctx_tokens+1, ctx_max - 1);
    ntokens_gen += N;
    if (ntokens_gen > ctx_max) {
      ntokens_gen = ctx_max;
    }
    N++;

    if (classifier_free_guidance != 1.0) {
      m.apply(ctx_tokens[0], 0, cfg_kvbuf, cfg_ybuf);
      cfg_ptr = 1;
    }

    fprintf(stderr, "a1gpt seed=%u sampling_temperature=%0.2f ntokens=%d\n", seed,
            sampling_temperature, ntokens_gen);


    {
      fprintf(stderr, "encoded prompt: ");
      for (int i = 0; i < N; i++) {
        fprintf(stderr, "%d ", ctx_tokens[i]);
      }
      char buf[4096];
      int decoded_siz = decoder.Decode(ctx_tokens, N, buf, 4096);
      fprintf(stderr, "\nGenerating:\n");
      fwrite(buf, 1, decoded_siz, stdout);
      fflush(stdout);
    }
    for (int j = 0; j < ntokens_gen; j++) {
      m.apply(ctx_tokens[j], j, kvbuf, ybuf);

      if (j < N - 1) {
        // no need to run lm_head on the prompt; afterwards, we start to
        // generate
        continue;
      }

      if (classifier_free_guidance != 1.0) {
        for (int k = 0; k < m.embedding_dim; k++) {
          ybuf[k] = classifier_free_guidance * ybuf[k] - (classifier_free_guidance-1) * cfg_ybuf[k];
        }
      }

      float r = (float)rand() / RAND_MAX;
      int sampled_token = m.sample_head(ybuf, sampling_temperature, r, logitbuf);

      ctx_tokens[j + 1] = sampled_token;
      if (classifier_free_guidance != 1.0) {
        m.apply(ctx_tokens[j+1], cfg_ptr, cfg_kvbuf, cfg_ybuf);
        cfg_ptr++;
      }

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
