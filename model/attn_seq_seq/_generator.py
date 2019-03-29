import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()

        self.vocab_size = kwargs.get('vocab_size')
        self.rnn_hidden_size = 256 #kwargs.get('rnn_hidden_size')

        self.V = nn.Linear(self.rnn_hidden_size, self.vocab_size)

    def forward(self, input):

        output = self.V(input) # (b, vocab_size)
        output = F.softmax(output, dim=1)

        return output
