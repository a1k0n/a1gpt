## a1gpt

throwaway C++ GPT-2 inference engine from @a1k0n w/ minimal but optimized BLAS
ops for AVX and Apple Silicon

no external dependencies except for accelerate framework on macos

## build / run

 - First, download and convert the model

`$ python3 scripts/download_and_convert_gpt2.py`

This will require `numpy` and `huggingface_hub` to be installed in Python

 - CMake and build

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
$ ./gpt2 -h
Usage: ./gpt2 [-s seed] [-t sampling_temperature] [-p prompt]
  -s seed: random seed (default: time(NULL))
  -t sampling_temperature: temperature for sampling (default: 0.90)
  -p prompt: prompt to start with (default: English-speaking unicorns)
  -n ntokens: number of tokens to generate (default=max: 1024)
  -c cfg_scale: classifier-free guidance scale; 1.0 means no CFG (default: 1.0)

```

Example generation on a Macbook Air M2 with default prompt, temperature:
```
$ ./gpt2 -s 1688452945 -n 256
a1gpt seed=1688452945 sampling_temperature=0.90 ntokens=301
encoded prompt: 50256 818 257 14702 4917 11 11444 5071 257 27638 286 44986 82 2877 287 257 6569 11 4271 31286 1850 19272 11 287 262 843 274 21124 13 3412 517 6452 284 262 4837 373 262 1109 326 262 44986 82 5158 2818 3594 13
Generating:
```
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The unicorn, nicknamed Macalpine in the state of Montana, was the first animal ever to speak the language. The animal was first reported in 1972, during the discovery of the same region by the Inkocroft Rendezvous Lourd system in the Andes. The specimen's linguistic abilities were not extremely rare, but a few unknowns led the bewildering team to believe that the unicorn appeared to be communicating with a group that was silent.

This fluency in a language exam can prevent a unicorn from communicating with a specific person or group, but scientists believe it is rare for a unicorn to mantain such linguistic abilities. In a test they found, thousands of false Mexican translates were sent. This finding, along with other brilliant discoveries in the area, revealed that unicorns communicate with their synapses, essentially the same level of coordination as humans. The unicorn's API was claimed to evolve through a single ancestor known as the Amarr. But they were only known in California, and in many other places, as Amarr.

The legendary Amarr DNA has been widely used as a tool by cosmologists to identify flying squirrels, maple leaves and bees. In the near future, scientists hope that unicorn species and their mitochondrial DNA will
```
elapsed: 4.091053s, 3.995169ms per token
```

