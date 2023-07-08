#include <stdio.h>
#include <stdint.h>
#include <unordered_map>

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
    if (tokens[i] == -1) {
      outbuf[j++] = '(';
      outbuf[j++] = '?';
      outbuf[j++] = ')';
      continue;
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

struct BPETrieNode {
  int token_length = -1;  // -1 indicates there is no token ending at this node
  int token_id;
  std::unordered_map<char, BPETrieNode*> children;

  ~BPETrieNode() {
    for (auto it = children.begin(); it != children.end(); it++) {
      delete it->second;
    }
  }
};

BPEEncoder::BPEEncoder() {
  root_ = new BPETrieNode();
}

BPEEncoder::~BPEEncoder() {
  delete root_;
}

bool BPEEncoder::Init(const std::vector<std::string>& vocab) {
  for (int i = 0; i < vocab.size(); i++) {
    auto token = vocab[i];
    BPETrieNode* node = root_;
    for (size_t i = 0; i < token.size(); i++) {
      char key = token[i];
      if (node->children.count(key) == 0) {
        node->children[key] = new BPETrieNode();
      }
      node = node->children[key];
    }
    node->token_length = token.size();
    node->token_id = i;
  }
  return true;
}

const char* BPEEncoder::Encode(const char *string, int *outbuf, int outbuf_size, int *ntokens) {
  *ntokens = 0;
  while(*string && *ntokens < outbuf_size) {
    BPETrieNode* node = root_;
    int last_token_length = -1;
    int last_token_id = -1;
    for (size_t i = 0; string[i]; i++) {
      char key = string[i];
      if (node->children.count(key) == 0) {
        break;
      }
      node = node->children[key];
      if (node->token_length != -1) {
        last_token_length = node->token_length;
        last_token_id = node->token_id;
      }
    }
    if (last_token_length == -1) {
      return string;
    } else {
      *outbuf++ = last_token_id;
      string += last_token_length;
      (*ntokens)++;
    }
  }
  return string;
}
