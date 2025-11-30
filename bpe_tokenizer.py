import copy
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set

class BPETokenizer():
    def __init__(self):
        self.special_tokens = ["<unk>","<pad>", "<bos>", "<eos>", "<mask>"]
        self.vocabulary: List = [] # Stores bytes or strings
        self.token_to_id_map: Dict = {}  

        self.merge_order: List[Tuple[int, int, int]] = [] 

    def preprocess(self, text): raise NotImplementedError
    def postprocess(self, tokens): raise NotImplementedError
    def initialize_vocabulary(self, training_data): raise NotImplementedError
    def special_token_to_token(self, token_str: str): return token_str


    def _replace_pair_in_sequence(self, seq: List[int], a: int, b: int, merged_id: int) -> List[int]:
        """Performs a single pass of pair replacement on a sequence of IDs."""
        i = 0
        new_seq = []
        while i < len(seq):
            # Check for the target pair (a, b)
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(merged_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return new_seq


    def train(self, training_data: List[str], vocab_size: int = 10000, write_progress: bool = False):
        self.initialize_vocabulary(training_data)

        corpus: List[List[int]] = [
            [self.token_to_id_map[t] for t in self.preprocess(text)]
            for text in training_data
        ]
        
        pair_freq: Counter[Tuple[int, int]] = Counter()
        pair_to_sequences: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        
        for idx, seq in enumerate(corpus):
            for a, b in zip(seq[:-1], seq[1:]):
                pair_freq[(a, b)] += 1
                pair_to_sequences[(a, b)].add(idx)


        while len(self.vocabulary) < vocab_size and pair_freq:
            (a, b), freq = pair_freq.most_common(1)[0]

            merged_token = self.vocabulary[a] + self.vocabulary[b]

            new_id = len(self.vocabulary)
            
            # Update vocabulary and merge order
            self.vocabulary.append(merged_token)
            self.token_to_id_map[merged_token] = new_id
            self.merge_order.append((a, b, new_id))

            if write_progress:
                print(f"Merging pair ({self.vocabulary[a]}, {self.vocabulary[b]}) -> {merged_token} | New ID: {new_id} | Frequency: {freq}")

            affected_sequence_indices = list(pair_to_sequences[(a, b)])
            if a in [4 ,168] and b in [4 ,168] and a != b:
                print(affected_sequence_indices)

            del pair_to_sequences[(a, b)] # Delete the old pair globally

            for seq_idx in affected_sequence_indices:
                seq = corpus[seq_idx]
                
                for i in range(len(seq) - 1):
                    old_pair = (seq[i], seq[i+1])
                    if pair_freq[old_pair] > 0:
                        pair_freq[old_pair] -= 1
                    pair_to_sequences[old_pair].discard(seq_idx)
                
                new_seq = self._replace_pair_in_sequence(seq, a, b, new_id)
                corpus[seq_idx] = new_seq
                
                for i in range(len(new_seq) - 1):
                    new_pair = (new_seq[i], new_seq[i+1])
                    pair_freq[new_pair] += 1
                    pair_to_sequences[new_pair].add(seq_idx)

            pair_freq = +pair_freq

    def encode(self, text: str):
        tokens = [self.token_to_id_map.get(t) for t in self.preprocess(text)]
    
        unk_token = self.special_token_to_token("<unk>")
        unk_id = self.token_to_id_map.get(unk_token)
        tokens = [t if t is not None else unk_id for t in tokens]

        for a_id, b_id, merged_id in self.merge_order:
            tokens = self._replace_pair_in_sequence(tokens, a_id, b_id, merged_id)
        return tokens


    def decode(self, token_ids: List[int]):
        tokens = [self.vocabulary[idx] for idx in token_ids]
        return self.postprocess(tokens)
    


class BPECharacterTokenizer(BPETokenizer):
    def preprocess(self, text):
        return list(text)
    
    def postprocess(self, tokens):
        return "".join(tokens)
    
    def initialize_vocabulary(self, training_data):
        self.vocabulary = []
        self.token_to_id_map = {}
        chars = sorted(set("".join(training_data)))

        # Add special tokens
        for sp in self.special_tokens:
            self.token_to_id_map[sp] = len(self.vocabulary)
            self.vocabulary.append(sp)

        # Add chars
        for ch in chars:
            self.token_to_id_map[ch] = len(self.vocabulary)
            self.vocabulary.append(ch)


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

        for sp in self.special_tokens:
            token = sp.encode("utf-8")
            self.token_to_id_map[token] = len(self.vocabulary)
            self.vocabulary.append(token)

        for i in range(256):
            b = bytes([i])
            self.token_to_id_map[b] = len(self.vocabulary)
            self.vocabulary.append(b)
