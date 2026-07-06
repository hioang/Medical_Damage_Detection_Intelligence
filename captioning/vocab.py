import re
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {0:"<pad>",1:"<start>",2:"<end>",3:"<unk>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def tokenizer(self, text):
        if not isinstance(text, str):
            return []
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        return [
            self.stoi.get(token, self.stoi["<unk>"])
            for token in self.tokenizer(text)
        ]