#include <stdio.h>
#include <stdint.h>

#include "bpe.h"

BPEDecoder::BPEDecoder() {}

BPEDecoder::~BPEDecoder() {}

bool BPEDecoder::Init(const char* vocab_path) {
  FILE *fp = fopen(vocab_path, "rb");
  if (!fp) {
    return false;
  }
  // each entry is just <length byte> <string>
  while (!feof(fp)) {
    uint8_t len;
    if (fread(&len, 1, 1, fp) != 1) {
      break;
    }
    char buf[256];
    if (fread(buf, 1, len, fp) != len) {
      break;
    }
    buf[len] = 0;
    vocab_.push_back(buf);
  }
  return true;
}

int BPEDecoder::Decode(const int* tokens, int ntokens, char* outbuf, int outbuf_size) {
  int j = 0;
  for (int i = 0; i < ntokens; i++) {
    if (j >= outbuf_size) {
      break;
    }
    if (tokens[i] < 0 || tokens[i] >= vocab_.size()) {
      break;
    }
    int len = vocab_[tokens[i]].size();
    if (j + len >= outbuf_size) {
      break;
    }
    const char* s = vocab_[tokens[i]].c_str();
    for (int k = 0; k < len; k++) {
      outbuf[j++] = *s++;
    }
  }
  outbuf[j] = 0;
  return j;
}