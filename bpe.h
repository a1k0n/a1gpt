#pragma once

#include <string>
#include <vector>

class BPEDecoder {
 public:
  BPEDecoder();
  ~BPEDecoder();

  bool Init(const char *vocab_path);

  // returns size of output
  int Decode(const int *tokens, int ntokens, char *outbuf, int outbuf_size);

  // intentionally also public
  std::vector<std::string> vocab_;
};

class BPETrieNode;
class BPEEncoder {
 public:
  BPEEncoder();
  ~BPEEncoder();

  bool Init(const std::vector<std::string> &vocab);

  // returns unconsumed part of string
  const char *Encode(const char *string, int *outbuf, int outbuf_size, int *ntokens);

 private:
  BPETrieNode *root_;
};