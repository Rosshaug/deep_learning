import copy
from collections import defaultdict

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

    def train(self, training_data: list[str], vocab_size = 10000):
        self.initialize_vocabulary(training_data)
        preprocessed_data = [self.preprocess(s) for s in training_data]

        while len(self.vocabulary) < vocab_size:
            pair_freq = defaultdict(int)
            for sample in preprocessed_data:
                for a, b in zip(sample[:-1], sample[1:]):
                    pair_freq[(a, b)] += 1

            pair_to_merge = max(pair_freq, key=pair_freq.get)
            preprocessed_data = self.merge(preprocessed_data, pair_to_merge)

            next_id = len(self.vocabulary)
            new_token = pair_to_merge[0] + pair_to_merge[1]
            self.vocabulary.append(new_token)
            self.token_to_id_map[new_token] = next_id
            self.merge_order.append((pair_to_merge[0], pair_to_merge[1]))

    
    def merge(self, training_data, pair_to_merge):
        a, b = pair_to_merge
        merged_token = a + b

        new_data = []
        for sample in training_data:
            i = 0
            new_sample = []
            while i < len(sample):
                if i < len(sample) - 1 and sample[i] == a and sample[i + 1] == b:
                    new_sample.append(merged_token)
                    i += 2
                else:
                    new_sample.append(sample[i])
                    i += 1
            new_data.append(new_sample)

        return new_data

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
