from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from vocab import Characters, MyData

class CharSeqRNN(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_dim, batch_size):
        super(self.__class__, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0.3)
        #self.hidden = self.hidden0
        #self.init_hidden(batch_size)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, charseq, hidden):
        x = self.char_embed(charseq)
        y, hidden = self.rnn(x, hidden)
        y = self.output(y)
        y = F.log_softmax(y, dim=2)
        return y, hidden

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                Variable(torch.zeros(1, batch_size, self.hidden_dim)))

if __name__ == '__main__':
    mydata = MyData(10, "input.txt")
    data_loader = data.DataLoader(dataset=mydata, batch_size=256, shuffle=True)
    chars = mydata.chars
    vocab_size = mydata.vocab_size
    charseqrnn = CharSeqRNN(vocab_size, 64, 256, 256)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(charseqrnn.parameters(), lr=1e-3)
    epochs = 10
    for epoch in range(epochs):
        charseqrnn.zero_grad()
        for i, (x,y) in enumerate(data_loader):
            #if i > 500: break;
            hidden = charseqrnn.init_hidden(x.size(0))
            x = Variable(x)
            y = Variable(y)
            optimizer.zero_grad()
            loss = 0.0
            pred_chars, hidden = charseqrnn(x, hidden)
            pred = pred_chars.permute(dims=(0,2,1))
            loss = loss_fn(pred, y)
            if i%100 == 0: print(epoch, i, loss.data.numpy())
            loss.backward()
            optimizer.step()

    textout = 'A'
    c = 'A'

    x = Variable(torch.LongTensor([chars(c)]).view(1,-1))
    h = charseqrnn.init_hidden(1)
    #charseqrnn.eval()
    for i in range(100):
        y, h = charseqrnn(x, h)
        y_max, idx = torch.max(y, dim=2)
        idx = idx.data.numpy()[0,0]
        print(chars.get_char(idx), end='')
    print('\n')
