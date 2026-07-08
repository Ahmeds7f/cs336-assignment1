import regex as re
import os
import time
from multiprocessing import Pool
import heapq
from tqdm import tqdm
from collections import Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries
import pickle
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class RevPair:
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        return self.pair > other.pair

def count_chunk(args):
    input_path, special_tokens, start, end = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        special_pattern = "|".join(re.escape(sptok) for sptok in special_tokens)
        segments = re.split(special_pattern, chunk)
    else:
        segments = [chunk]

    # this part is to actually get the freqencies of everything
    word_freq = {}

    for segment in segments:
        for m in re.finditer(PAT, segment):
            token_bytes = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            word_freq[token_bytes] = word_freq.get(token_bytes, 0) + 1

    return word_freq


def merge(word, pair):
    """
    takes in a word in the type of a tuple of bytes, returns the word merged at the pair
    """
    merged = pair[0] + pair[1]
    new_word = []
    i = 0
    while i < len(word):
        if word[i] == pair[0] and  i < len(word) - 1 and word[i+1] == pair[1]:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)


def bpe_Training(input_path: str, vocab_size: int, special_tokens: list[str], show_times = False):
    # this part is for preparing the data
    # BUT its for using parallel processing as well
        # this part is for initiallizing the vocab

    vocab = {i:bytes([i]) for i in range(256)}
    for i, sptok in enumerate(special_tokens):
        vocab[256 + i] = bytes(sptok, "UTF-8")


    num_processes = max(1, os.cpu_count() - 2)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # this part is for making the iterable for the pooling function
    iterable = [(input_path, special_tokens, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]


    # MAKING WORD_FREQ
    start1 = time.time()
    with Pool(num_processes) as pool:
        list_of_word_freq = pool.map(count_chunk, iterable)

    word_freq = {}

    for mini_word_freq in list_of_word_freq:
        for word, freq in mini_word_freq.items():
            word_freq[word] = word_freq.get(word, 0) + freq

    if show_times:
        print(f"Making word_freq takes {time.time() - start1} seconds")

    # MAKINF PAIR_FREQ AND PAIR_TO_WORDS
    start2 = time.time()
    pair_freq = {}
    pair_to_words = {}

    for word, freq in word_freq.items():
        for x,y in zip(word, word[1:]):
            pair_freq[(x,y)] = pair_freq.get((x,y), 0) + freq
            pair_to_words.setdefault((x, y), set()).add(word)

    if show_times:
        print(f"Making pair_freq and pair_to_freq takes {time.time() - start2} seconds")

    # now we find the most occuring pair, merge, then
    # repeat until we have the desired vocab size

    # one thing I ran into during training the tokenizer was that having to use max() every iteration
    #  of the merging was very inefficient, so instead i will use a heap which makes things logn instead of n

    heap = [(-count, RevPair(pair)) for pair, count in pair_freq.items()]
    heapq.heapify(heap)

    # MAKING THE MERGES
    start3 = time.time()
    pbar = tqdm(total=vocab_size - len(vocab), desc="merges")

    merges = []
    while vocab_size > len(vocab):
        if not pair_freq:
            break
        # find the most frequent pair:
        while True:
            negcount, Revp = heapq.heappop(heap)
            best_pair = Revp.pair
            count = -negcount
            if best_pair in pair_freq and count == pair_freq[best_pair]:
                break


        # start by adding the merge to vocab and merges
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # now we find the words that contain this best_pair:
        words = list(pair_to_words[best_pair])

        for word in words:
            new_word = merge(word, best_pair)

            # lets remove word and add new_word into word_freq with the same freq
            freq = word_freq[word]
            word_freq[new_word] = word_freq[word]
            del word_freq[word]

            old_pairs = list(zip(word, word[1:]))
            new_pairs = list(zip(new_word, new_word[1:]))

            for p in old_pairs:
                pair_to_words[p].discard(word)
            for p in new_pairs:
                pair_to_words.setdefault(p, set()).add(new_word)

            delta = Counter(new_pairs)
            delta.subtract(old_pairs)

            for p, occ in delta.items():
                if occ ==0:
                    continue
                pair_freq[p] = pair_freq.get(p, 0) + occ * freq

                if pair_freq[p] == 0:
                    del pair_freq[p]
                else:
                    heapq.heappush(heap, (-pair_freq[p], RevPair(p)))

        pair_freq.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)
        pbar.update(1)
    pbar.close()
    if show_times:
        print(f"Making merges takes {time.time() - start3} seconds")
    return vocab, merges


# here ill just implement the Tokenizer class for encoding and decoding text

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """# Construct a tokenizer from a given
        # vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
        # the following parameters:"""
        self.vocab = vocab
        self.bytes_to_token = {item: key for key, item in vocab.items()}
        self.pair_rank = {pair: i for i, pair in enumerate(merges)}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """# that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
        # same format that your BPE training code output) and (optionally) a list of special tokens.
        # This method should accept the following additional parameters:"""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str):
        """list[int] Encode an input text into a sequence of token IDs."""
        if self.special_tokens is None:
            chunks = [text]
        else:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(s) for s in specials) + ")"
            chunks = re.split(pattern, text)

        final_state = []
        for text in chunks:
            if not text:
                continue
            elif self.special_tokens and text in self.special_tokens:
                final_state.append(text.encode("utf-8"))
            else:
                for m in re.finditer(PAT, text):
                    pretoken_bytes = tuple(bytes([b]) for b in m.group().encode("utf-8"))

                    current_pairs = [(x,y) for x,y in zip(pretoken_bytes, pretoken_bytes[1:]) if (x,y) in self.merges]

                    pair_rank = {pair: self.pair_rank[pair] for pair in current_pairs}

                    pair_to_merge = min(pair_rank.keys(), key = lambda x: pair_rank[x], default= None)

                    while pair_to_merge is not None:
                        pretoken_bytes_updated = []

                        i = 0
                        while i< len(pretoken_bytes):
                            if pretoken_bytes[i] == pair_to_merge[0] and i + 1 < len(pretoken_bytes) and pretoken_bytes[i + 1] == pair_to_merge[1]:
                                pretoken_bytes_updated.append(pair_to_merge[0] + pair_to_merge[1])
                                i += 2
                            else:
                                pretoken_bytes_updated.append(pretoken_bytes[i])
                                i += 1

                        pretoken_bytes = pretoken_bytes_updated

                        current_pairs = [(x,y) for x,y in zip(pretoken_bytes, pretoken_bytes[1:]) if (x,y) in self.merges]

                        pair_rank = {pair: self.pair_rank[pair] for pair in current_pairs}

                        pair_to_merge = min(pair_rank.keys(), key = lambda x: pair_rank[x], default= None)

                    final_state.extend(pretoken_bytes)

        return [self.bytes_to_token[byte] for byte in final_state]


    def encode_iterable(self, iterable):
        """-> Iterator[int] Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory."""

        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]):
        """# str Decode a sequence of token IDs into text."""
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
