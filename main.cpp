#include <getopt.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#include <string>

#include "bpe.h"
#include "blas.h"
#include "tensor.h"
#include "model.h"

const int ctx_max = 1024;
extern bool load_gpt2_model(Model &m);
float sampling_temperature = 0.9;
float classifier_free_guidance = 1.0;
const char *eval_filename = NULL;

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
  fprintf(stderr, "  -e filename: evaluate perplexity on a file\n");
  exit(1);
}

int generate(const char *prompt, int ntokens_gen, Model &m, BPEDecoder &decoder, BPEEncoder &encoder, unsigned int seed) {
  srand(seed);
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
  int N;
  const char *leftover = encoder.Encode(prompt, ctx_tokens+1, ctx_max - 1, &N);
  if (*leftover) {
    fprintf(stderr, "WARNING: prompt was truncated to %d tokens\nleftover input: %s", N, leftover);
  }
  ntokens_gen += N;
  if (ntokens_gen > ctx_max) {
    ntokens_gen = ctx_max;
  }
  N++;

  if (classifier_free_guidance != 1.0) {
    m.apply_transformer(ctx_tokens[0], 0, cfg_kvbuf, cfg_ybuf);
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
    // skip displaying the initial <|endoftext|> token
    int decoded_siz = decoder.Decode(ctx_tokens+1, N-1, buf, 4096);
    fprintf(stderr, "\nGenerating:\n");
    fwrite(buf, 1, decoded_siz, stdout);
    fflush(stdout);
  }
  for (int j = 0; j < ntokens_gen; j++) {
    m.apply_transformer(ctx_tokens[j], j, kvbuf, ybuf);

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

    m.apply_lm_head(ybuf, logitbuf);
    float r = (float)rand() / RAND_MAX;
    int sampled_token = sample_logits(sampling_temperature, r, logitbuf);

    ctx_tokens[j + 1] = sampled_token;
    if (classifier_free_guidance != 1.0) {
      m.apply_transformer(ctx_tokens[j+1], cfg_ptr, cfg_kvbuf, cfg_ybuf);
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
  return ntokens_gen + N;
}

int evaluate(Model &m, BPEDecoder &decoder, BPEEncoder &encoder, const char *filename) {
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Failed to open %s\n", filename);
    exit(1);
  }
  struct stat st;
  if (fstat(fd, &st) < 0) {
    fprintf(stderr, "Failed to stat %s\n", filename);
    exit(1);
  }
  size_t siz = st.st_size;
  char *buf = (char *)mmap(NULL, siz, PROT_READ, MAP_PRIVATE, fd, 0);
  if (buf == MAP_FAILED) {
    fprintf(stderr, "Failed to mmap %s\n", filename);
    exit(1);
  }

  Tensorf<3> kvbuf(12, ctx_max, 2*m.embedding_dim);
  Tensorf<1> ybuf(m.embedding_dim);
  Tensorf<1> logitbuf(m.ntokens);
  int tokenbuf[1024];
  const char *read_ptr = buf;
  double sum_logp = 0;
  int Ntokens = 0;
  while (read_ptr < buf + siz) {
    int N;
    read_ptr = encoder.Encode(read_ptr, tokenbuf, 1024, &N);
    {
      char buf[4096];
      int decoded_siz = decoder.Decode(tokenbuf, N, buf, 4096);
      fwrite(buf, 1, decoded_siz, stdout);
      fflush(stdout);
    }
    size_t off = read_ptr - buf;
    for (int i = 0; i < N; i++) {
      m.apply_transformer(tokenbuf[i], i, kvbuf, ybuf);
      m.apply_lm_head(ybuf, logitbuf);
      sum_logp += cross_entropy(logitbuf, tokenbuf[i]);
      Ntokens++;
      fprintf(stderr, "%lu/%lu %d/%d perplexity: %f\r", off, siz, i, N, -sum_logp / Ntokens);
      fflush(stderr);
    }
  }
  fprintf(stderr, "\n");
  return Ntokens;
}

int main(int argc, char **argv) {
  unsigned int seed = time(NULL);
  const char *prompt = DEFAULT_PROMPT;
  int ntokens_gen = 1024;

  int c;
  while ((c = getopt(argc, argv, "s:t:p:n:c:e:h")) != -1) {
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
      case 'e':
        eval_filename = optarg;
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

  m.to_device();

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  if (eval_filename != NULL) {
    ntokens_gen = evaluate(m, decoder, encoder, eval_filename);
  } else {
    ntokens_gen = generate(prompt, ntokens_gen, m, decoder, encoder, seed);
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  printf("elapsed: %fs, %fms per token\n", elapsed, 1000 * elapsed / ntokens_gen);
}
