from cs336_basics.BPETraining import bpe_Training
import time
import pickle



if __name__ == "__main__":
    start = time.time()
    vocab, merges = bpe_Training("/Users/ahmeds7f/Desktop/cs336/assignment1-basics/data/owt_train.txt",
                                 32_000, ["<|endoftext|>"],True)
    longest_token = max(vocab.values(), key = len)
    longest_token_str = longest_token.decode("utf-8", errors="replace")

    print(longest_token, longest_token_str)


    import os
    os.makedirs("/Users/ahmeds7f/Desktop/cs336/assignment1-basics/tokenizers", exist_ok=True)
    with open("/Users/ahmeds7f/Desktop/cs336/assignment1-basics/tokenizers/owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("/Users/ahmeds7f/Desktop/cs336/assignment1-basics/tokenizers/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
