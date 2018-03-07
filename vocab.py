import torch
import torch.utils.data as data
from collections import Counter

class Characters(object):
    def __init__(self, filename):
        self.char2idx = {}
        self.idx2char = {}
        self.idx = 0

        with open(filename, 'r') as f:
            self.text = ''
            for line in f:
                self.text = self.text + line

        counter = Counter(self.text)
        d = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        #for i, word in enumerate(words):
        for i, (c, _) in enumerate(d):
            self.char2idx[c] = i
            self.idx2char[i] = c

    def get_char(self, idx):
        assert idx < len(self.idx2char)
        return self.idx2char[idx]

    def __call__(self, char):
        return self.char2idx[char]

    def __len__(self):
        return len(self.char2idx)

class MyData(data.Dataset):
    def __init__(self, seq_len, filename):
        self.seq_len = seq_len
        self.chars = Characters(filename)
        self.vocab_size = len(self.chars)
        self.data = self.chars.text

    def __getitem__(self, idx):
        chars = self.data[idx:(idx + self.seq_len + 1)]
        chars = torch.LongTensor([self.chars(c) for c in chars])
        chars_in = chars[:-1]
        chars_out = chars[1:]
        return chars_in, chars_out

    def __len__(self):
        return len(self.data) - self.seq_len
