#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "model.h"

template <int N>
static Tensorf<N> fetch_weights(const nlohmann::json &j, void *addr, const char *name) {
  Tensorf<N> out;

  int offset = j[name]["data_offsets"][0];
  auto shape = j[name]["shape"];
  if (N != shape.size()) {
    int shape_size = shape.size();
    fprintf(stderr, "fetch_weights: %s: expected %d dimensions, got %d\n", name, N, shape_size);
    exit(1);
  }
  for (int i = 0; i < N; i++) {
    out.shape[i] = shape[i];
  }
  
  float *weights = (float *)((char *)addr + offset);
  out.data = weights;

  return out;
}

template <int N>
static Tensorf<N> fetch_layer_weights(const nlohmann::json &j, void *addr, int layer, const char *name) {
  char buf[256];
  snprintf(buf, sizeof(buf), "h.%d.%s", layer, name);
  return fetch_weights<N>(j, addr, buf);
}

bool Model::Load(const char *path) {
  // mmap "model.safetensors" into memory
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    return false;
    fprintf(stderr, "Failed to open model.safetensors\n");
    exit(1);
  }
  struct stat sb;
  fstat(fd, &sb);
  char *addr = (char*) mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  uint64_t header_len = *((uint64_t *)addr);

  std::string header(addr + sizeof(uint64_t), header_len);
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(header);
  } catch (nlohmann::json::parse_error &e) {
    fprintf(stderr,
            "Failed to parse header: %s exception id: %d position %lu\n",
            e.what(), e.id, e.byte);
  }
  addr += header_len + sizeof(uint64_t);

  // assumed to be GPT-2, 12 layers
  h = new TransformerBlock[12];
  wte_weight = fetch_weights<2>(j, addr, "wte.weight");
  wpe_weight = fetch_weights<2>(j, addr, "wpe.weight");
  ln_f.bias = fetch_weights<1>(j, addr, "ln_f.bias");
  ln_f.weight = fetch_weights<1>(j, addr, "ln_f.weight");
  embedding_dim = wte_weight.shape[1];
  context_len = wpe_weight.shape[0];
  ntokens = wte_weight.shape[0];
  printf("embedding_dim: %d\n", embedding_dim);
  printf("context_len: %d\n", context_len);
  printf("ntokens: %d\n", ntokens);

  for (int i = 0; i < 12; i++) {
    h[i].attn.num_heads = 12;
    h[i].attn.c_attn_bias = fetch_layer_weights<1>(j, addr, i, "attn.c_attn.bias");
    h[i].attn.c_attn_weight = *fetch_layer_weights<2>(j, addr, i, "attn.c_attn.weight").TransposedCopy();
    h[i].attn.c_proj_bias = fetch_layer_weights<1>(j, addr, i, "attn.c_proj.bias");
    h[i].attn.c_proj_weight = *fetch_layer_weights<2>(j, addr, i, "attn.c_proj.weight").TransposedCopy();
    h[i].ln_1.bias = fetch_layer_weights<1>(j, addr, i, "ln_1.bias");
    h[i].ln_1.weight = fetch_layer_weights<1>(j, addr, i, "ln_1.weight");
    h[i].ln_2.bias = fetch_layer_weights<1>(j, addr, i, "ln_2.bias");
    h[i].ln_2.weight = fetch_layer_weights<1>(j, addr, i, "ln_2.weight");
    h[i].mlp.c_fc_bias = fetch_layer_weights<1>(j, addr, i, "mlp.c_fc.bias");
    h[i].mlp.c_fc_weight = *fetch_layer_weights<2>(j, addr, i, "mlp.c_fc.weight").TransposedCopy();
    h[i].mlp.c_proj_bias = fetch_layer_weights<1>(j, addr, i, "mlp.c_proj.bias");
    h[i].mlp.c_proj_weight = *fetch_layer_weights<2>(j, addr, i, "mlp.c_proj.weight").TransposedCopy();
  }
  return true;
}
