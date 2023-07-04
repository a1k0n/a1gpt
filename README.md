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
  -t sampling_temperature: temperature for sampling (default: 0.9)
  -p prompt: prompt to start with (default: English-speaking unicorns)

```

Example generation on a Macbook Air M2 with default prompt, temperature:
```
$ ./gpt2
a1gpt seed=1688452945 sampling_temperature=0.900000
encoded prompt: 50256 818 257 14702 4917 11 11444 5071 257 27638 286 44986 82 2877 287 257 6569 11 4271 31286 1850 19272 11 287 262 843 274 21124 13 3412 517 6452 284 262 4837 373 262 1109 326 262 44986 82 5158 2818 3594 13
Generating:
```
`<|endoftext|>`In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The unicorn, nicknamed Macalpine in the state of Montana, was the first animal ever to speak the language. The animal was first reported in 1972, during the discovery of the same region by the Inkocroft Rendezvous Lourd system in the Andes. The specimen's linguistic abilities were not extremely rare, but a few unknowns led the bewildering team to believe that the unicorn appeared to be communicating with a group that was silent.

This fluency in a language exam can prevent a unicorn from communicating with a specific person or group, but scientists believe it is rare for a unicorn to mantain such linguistic abilities. In a test they found, thousands of false Mexican translates were sent. This finding, along with other brilliant discoveries in the area, revealed that unicorns communicate with their synapses, essentially the same level of coordination as humans. The unicorn's API was claimed to evolve through a single ancestor known as the Amarr. But they were only known in California, and in many other places, as Amarr.

The legendary Amarr DNA has been widely used as a tool by cosmologists to identify flying squirrels, maple leaves and bees. In the near future, scientists hope that unicorn species and their mitochondrial DNA will be able to study the world's fossil record, allowing new ideas to be found about fossils and modern animals.

Chris Cross, an archaeologist specializing in some of the mysterious "proto-phylopsids" found in the Andes, offered this video to PBS-TV to explain how the unicorn and Amarr populations meet, and what they could find.

Transcript

Here's how the famous race of Lemurians tinkered with their metahuman identity. What they discovered is the discrepancy between their obscure work of poetic and metaphysical understanding.

That they were mammalian evolved in tropical regions of the world. It's part of what suggests that they probably lived out in highland areas, and maybe on island in the Andes. But when the migration arrived in the late 20th century and the Amarr abandoned their form after a hearth cave unearthed a staggering number of the elk antlers, then it became clear that they were probably Homo sapiens. In the early 1980s, when they moved up into the US, they went back to Africa, where they had been from. And in those days they called themselves "horned sapphires because they spoke the language of their ancestors that lived in south Africa."

It turned out, that at least four out of these four species had evolved to have Polynesian feet. So what they found was a hybrid unicorn, and then one elephant, and that elephant made itself into an elephant. This was known as the Atashii.

Is it an animal that doesn't do what our ancestors do, but they have high intelligence, and that's just the minimum intelligence that humans have. Polynesians and the Atashii's all like humans.

So what did they have in common with the other Atashii species? They had, essentially, the same type of ancestry. It was a hybrid elephant, somewhat of the advantaged tribe of Atashii, that the Atashii had.

So out in the woods that's where many things happen to a person. Humans are the only ones who have heard the word "atashii." Like they say, I thought "atashii." But the person did have a standard, low intelligence, but they also had a lifespan. And so out in the woods and out in the woods, if you're not a hunter then your life is meaningless. So so the Atashii did have a record of what they did, but basically it reflected the preferences, the social/moral disincentives, and the emotional motivation they had to hunt and foraging. They always knew who had the most dung. And the Atashii got the best food. And so, once they were in the water and there was a flood, they'd get more dung. And they'd look up and see an Atashii in the water with a plow and they'd go (with confidence). And they'd assume that it could be a suspect.

The Atashii made their communication about the sea's relationship with nature part of their culture. That's important the much more so family members of these animals. That tells us that they were ancestors to different animals, and that all of their lives, from their childhood in the water, at least, had been spent with humans.

Once that person, that person's ancestor, arrived, they had to have an international communication system and they didn't. So even though the Atashii had an encephalomyelitis in
```
elapsed: 12.845083s, 12.544026ms per token
```

