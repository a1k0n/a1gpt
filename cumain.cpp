extern float* gpuTransferFloats(float *data, int size);
extern void gpuDumpMemoryInfo();

int main(int argc, char **argv) {
  gpuDumpMemoryInfo();
  return 0;
}
