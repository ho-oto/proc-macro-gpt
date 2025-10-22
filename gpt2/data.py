# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "transformers",
#   "datasets",
# ]
# ///

from collections import Counter
from itertools import islice
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

dataset = load_dataset("roneneldan/TinyStories")
loader = DataLoader(dataset["train"], batch_size=2**17)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

max_length = 1024
vocab_size_tiny = 8000

c = Counter()
for d in islice(dataset["train"], 2**17):
    c.update(tokenizer(d["text"])["input_ids"])
vocab_tiny = [i for i, _ in c.most_common(vocab_size_tiny - 1)]
vocab_tiny.append(tokenizer.eos_token_id)

with open(f"tokens.{tokenizer.vocab_size}.json", "w") as f:
    json.dump([tokenizer.decode(i) for i in range(tokenizer.vocab_size)], f)
with open(f"tokens.{vocab_size_tiny}.json", "w") as f:
    json.dump([tokenizer.decode(i) for i in vocab_tiny], f)

id_table = {id_orig: id_new for id_new, id_orig in enumerate(vocab_tiny)}
for i in range(tokenizer.vocab_size):
    if i not in id_table:
        id_table[i] = vocab_size_tiny - 1

with open(f"tiny_stories.train.{max_length}.{tokenizer.vocab_size}.bin", "wb") as f:
    pass
with open(f"tiny_stories.train.{max_length}.{vocab_size_tiny}.bin", "wb") as f:
    pass
for data in islice(loader, 10):
    tokens_np = tokenizer(
        data["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )["input_ids"].flatten()
    tokens_np_tiny = np.vectorize(lambda x: id_table[x])(tokens_np)
    with open(f"tiny_stories.train.{max_length}.{tokenizer.vocab_size}.bin", "ab") as f:
        f.write(tokens_np.astype(np.uint16).tobytes())
    with open(f"tiny_stories.train.{max_length}.{vocab_size_tiny}.bin", "ab") as f:
        f.write(tokens_np_tiny.astype(np.uint16).tobytes())
