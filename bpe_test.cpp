#include "bpe.h"

int main(int argc, char **argv) {
  BPEDecoder decoder;
  if (!decoder.Init("model/vocab.bin")) {
    printf("failed to init decoder\n");
    return 1;
  }
  BPEEncoder encoder;
  if (!encoder.Init(decoder.vocab_)) {
    printf("failed to init encoder\n");
    return 1;
  }
  const char *prompt = "The rain in spain falls mainly on the";
  if (argc > 1) {
    prompt = argv[1];
  }
  int outbuf[1024];
  int ntokens;
  const char *leftover = encoder.Encode(prompt, outbuf, 1024, &ntokens);
  printf("encoding: ");
  for (int i = 0; i < ntokens; i++) {
    printf("%d ", outbuf[i]);
  }
  printf("\n");
  char outbuf2[256];
  decoder.Decode(outbuf, ntokens, outbuf2, 256);
  printf("re-decoded: %s\n", outbuf2);

  {
    // test partial input
    const char *prompt = "The rain in spain falls mainly on the";
    for (;;) {
      int buf[4];
      prompt = encoder.Encode(prompt, buf, 4, &ntokens);
      printf("partial-encoding(%d): ", ntokens);
      for (int i = 0; i < ntokens; i++) {
        printf("%d ", buf[i]);
      }
      if (!*prompt) {
        break;
      }
      printf("\nleftover: %s\n", prompt);
    }
  }

  return 0;
}
