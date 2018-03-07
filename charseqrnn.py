import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import *

class CharSeqRNN(nn.Module):
    def __init__(self, num_chars, embed_dim, hidden_dim):
        super(self.__class__, self).__init__()

        self.num_chars = num_chars # number of chars for this case
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim # we could keep this same as hidden dim to reduce one variable

        self.encode = nn.Embedding(num_chars, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True) # we can try dropout
        self.decode = nn.Linear(hidden_dim, num_chars)

    def forward(self, inp, hidden):
        inp = self.encode(inp) #input must be N x T
        output, hidden = self.rnn(inp, hidden)
        output = self.decode(output)
        #output = F.log_softmax(output, dim=2) # we can  do this at output
        return output, hidden

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                Variable(torch.zeros(1, batch_size, self.hidden_dim)))

class CharSeqModel():
    def __init__(self, n_chars, embed_dim, hidden_dim, pretrained=False):
        self.args = (n_chars, embed_dim, hidden_dim)
        self.model = CharSeqRNN(n_chars, embed_dim, hidden_dim)
        if pretrained:
            self.load()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, batch_size):
        hidden = self.model.init_hidden(batch_size)
        self.model.zero_grad()
        loss = 0
        c_in, c_out = training_batch(batch_size)

        output, hidden = self.model(c_in, hidden)
        loss = self.criterion(output.view(-1, n_chars), c_out.view(-1))
        # we have global variable n_chars so it should be fine

        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def train(self, batch_size, epochs):
        plot_fq = int(epochs/100) + 1
        print_fq = int(epochs/10) + 1
        losses = []
        loss_avg = 0
        for epoch in range(1, epochs+1):
            loss1 = self.train_epoch(batch_size)
            loss_avg += loss1
            if epoch % print_fq == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / epochs * 100, loss1))
                print(run('\n', 150, 0.5), '\n')

            if epoch % plot_fq == 0:
                losses.append(loss_avg / plot_fq)
                loss_avg = 0
        self.save()
        return losses

    def save(self):
        state = {
            'args': self.args,
            'state_dict': self.model.state_dict()
        }
        torch.save(state, 'pretrained/model.pth.tar')
        print("Saved.")

    def load(self):
        state = torch.load('pretrained/model.pth.tar')
        self.args = state['args']
        self.model = CharSeqRNN(*self.args)
        self.model.load_state_dict(state['state_dict'])
        print("Loaded.")

    def run(self, init_str='A', length=200, temp=0.4):
        hidden = self.model.init_hidden(1) # batch_size=1
        pred = init_str
        if len(init_str) > 1:
            input = char_index(init_str[:-1])
            _, hidden = model(input, hidden)

        input = char_index(init_str[-1])

        for i in range(length):
            output, hidden = self.model(input, hidden)
            output_dist = F.softmax(output.view(-1)/temp, dim=0).data
            idx = torch.multinomial(output_dist, 1)[0]
            pred_char = all_chars[idx]
            pred += pred_char
            input = char_index(pred_char)
        return pred

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained', type=bool, default=False)
    args = parser.parse_args()

    # n_chars is global from utils
    embed_dim = 128
    hidden_dim = 128
    pretrained = args.pretrained
    model = CharSeqModel(n_chars, embed_dim, hidden_dim, pretrained)

    batch_size = 64
    epochs = 2000
    if not pretrained:
        losses = model.train(batch_size, epochs)
    # can plot losses

    model.run('\n', 500, 0.2)
