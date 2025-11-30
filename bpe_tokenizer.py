import copy
from collections import defaultdict
from typing import List, Dict, Tuple

class BPETokenizer():
    def __init__(self):
        self.special_tokens = ["<unk>","<pad>", "<bos>", "<eos>"]
        self.vocabulary = []
        self.token_to_id_map = {}
        self.merge_order = []

    def preprocess(self, text):
        raise NotImplementedError
    
    def postprocess(self, tokens):
        raise NotImplementedError

    def initialize_vocabulary(self, training_data):
        raise NotImplementedError

    def special_token_to_token(self, token_str: str):
        return token_str

    def encode(self, text:str):
        tokens = self.preprocess(text)

        for a, b in self.merge_order:
            merged = a + b
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        unk_token = self.special_token_to_token("<unk>")
        unk_id = self.token_to_id_map.get(unk_token, 0)
        ids = [self.token_to_id_map.get(t, unk_id) for t in tokens]
        return ids
    
    def decode(self, tokens:list[int]):
        result_tokens = []
        for idx in tokens:
            result_tokens.append(self.vocabulary[idx])

        return self.postprocess(result_tokens)

    def train(self, training_data: List[str], vocab_size: int = 10000):
            self.initialize_vocabulary(training_data)

            token_freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
            for text in training_data:
                tokens = tuple(self.preprocess(text))
                token_freqs[tokens] += 1

            while len(self.vocabulary) < vocab_size:
                pair_freq = defaultdict(int)
                for tokens, freq in token_freqs.items():
                    for a, b in zip(tokens[:-1], tokens[1:]):
                        pair_freq[(a, b)] += freq

                if not pair_freq:
                    break

                pair_to_merge = max(pair_freq, key=pair_freq.get)
                a, b = pair_to_merge
                merged_token = a + b

                new_token_freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
                
                for tokens, freq in token_freqs.items():
                    i = 0
                    new_tokens: List[str] = []
                    while i < len(tokens):
                        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                            new_tokens.append(merged_token)
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    
                    new_token_freqs[tuple(new_tokens)] += freq
                
                token_freqs = new_token_freqs 

                next_id = len(self.vocabulary)
                self.vocabulary.append(merged_token)
                self.token_to_id_map[merged_token] = next_id
                self.merge_order.append((a, b))


class BPECharacterTokenizer(BPETokenizer):
    def preprocess(self, text):
        return list(text)
    
    def postprocess(self, tokens):
        return "".join(tokens)
    
    def initialize_vocabulary(self, training_data):
        self.vocabulary = []
        self.token_to_id_map = {}
        for char in self.special_tokens + sorted(list(set("".join(training_data)))):
            self.token_to_id_map[char] = len(self.vocabulary)
            self.vocabulary.append(char)


class BPEByteTokenizer(BPETokenizer):
    def preprocess(self, text):
        return [bytes([b]) for b in text.encode("utf-8")]
    
    def postprocess(self, tokens):
        return b"".join(tokens).decode("utf-8")

    def special_token_to_token(self, token_str: str):
        return token_str.encode("utf-8")

    def initialize_vocabulary(self, _):
        self.vocabulary = []
        self.token_to_id_map = {}
        for token in [self.special_token_to_token(t) for t in self.special_tokens] +[bytes([i]) for i in range(256)]:
            self.token_to_id_map[token] = len(self.vocabulary)
            self.vocabulary.append(token)
