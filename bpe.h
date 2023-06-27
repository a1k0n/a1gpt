#ifndef BPE_H_
#define BPE_H_

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

#endif /* BPE_H_ */