# first, fetch https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json to ./vocab.json if it doesn't exist
# then export the vocab as a giant JSON array

import json
import sys
import os

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


if not os.access("gpt2-vocab.json", os.F_OK):
    print("Downloading vocab.json...")
    ok = os.system("wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json")
    if ok != 0:
        print("Failed to download vocab.json")
        sys.exit(1)

b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}

with open("gpt2-vocab.json", 'r') as f:
    vocab_json = json.load(f)
    vocab = ['']*len(vocab_json)
    for k, v in vocab_json.items():
        vocab[v] = bytearray([u2b[c] for c in k])
    
    out = open("vocab.bin", 'wb')
    for v in vocab:
        lenbyte = bytearray([len(v)])
        out.write(lenbyte)
        out.write(v)
    out.close()

