## a1gpt

throwaway C++ GPT-2 inference engine from @a1k0n w/ minimal but optimized BLAS
ops for AVX and Apple Silicon

no external dependencies except for accelerate framework on macos

## build / run

First, download and convert the model

`$ python3 scripts/download_and_convert_gpt2.py`

CMake and build

note: RelWithDebInfo is the default build type, so it should run pretty quick

```
$ mkdir build
$ cd build
$ cmake ..
-- The CXX compiler identification is GNU 11.3.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: a1gpt/build
$ make -j
[ 12%] Building CXX object CMakeFiles/bpe_test.dir/bpe_test.cpp.o
[ 25%] Building CXX object CMakeFiles/bpe_test.dir/bpe.cpp.o
[ 37%] Building CXX object CMakeFiles/gpt2.dir/main.cpp.o
[ 50%] Building CXX object CMakeFiles/gpt2.dir/model_load_gpt2.cpp.o
[ 62%] Building CXX object CMakeFiles/gpt2.dir/model.cpp.o
[ 75%] Building CXX object CMakeFiles/gpt2.dir/bpe.cpp.o
[ 87%] Linking CXX executable bpe_test
[100%] Linking CXX executable gpt2
[100%] Built target bpe_test
[100%] Built target gpt2
$ ./gpt2 "provide your own prompt here"
```

