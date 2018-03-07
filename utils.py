import string
import time, math
import numpy.random as random

import torch
from torch.autograd import Variable

all_chars = string.printable
n_chars = len(all_chars)

with open('input.txt', 'r') as file:
    text = file.read()

text_len = len(text)

seq_len = 200

def random_seq(seq_len=200):
    start = random.randint(0, text_len - seq_len + 1)
    end = start + seq_len
    return text[start:end]

def char_index(chars):
    return Variable(torch.LongTensor([all_chars.index(c) for c in chars]).view(1,-1))

def training_batch(batch_size, seq_len=200):
    chars_in = []
    chars_out = []
    for i in range(batch_size):
        char_seq = random_seq(seq_len)
        chars_in.append(char_index(char_seq[:-1]))
        chars_out.append(char_index(char_seq[1:]))
    chars_in = torch.cat(chars_in, dim=0)
    chars_out = torch.cat(chars_out, dim=0)
    return chars_in, chars_out

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
