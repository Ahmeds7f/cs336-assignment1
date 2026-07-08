import os
import pickle
from cs336_basics.BPETraining import bpe_Training

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "owt_train.txt")
TOK = os.path.join(ROOT, "tokenizers")

vocab, merges = bpe_Training(DATA, 32_000, ["<|endoftext|>"], True)

longest = max(vocab.values(), key=len)
print("longest token:", longest, longest.decode("utf-8", errors="replace"))

os.makedirs(TOK, exist_ok=True)
with open(os.path.join(TOK, "owt_vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)
with open(os.path.join(TOK, "owt_merges.pkl"), "wb") as f:
    pickle.dump(merges, f)

print("done")
