import torch.nn as nn
import torch
import torch.nn.functional as F

class DotAttention(nn.Module):
    def __init__(self, **kwargs):
        super(DotAttention, self).__init__()

        self.rnn_hidden_size = 256 #kwargs.get('rnn_hidden_size')
        #self.W = nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size)

        self.linear_in  = nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size)
        self.linear_out = None

    def forward(self, encoder_outputs, decoder_input):
        # encoder_outputs: (b, max_len, hidden_size)
        # decoder_input: (b, hidden_size)

        decoder_input_ = decoder_input.unsqueeze(1) # (b, 1 , hidden_size)

        # (b_size, 1, hidden_size)
        decoder_input_ = self.linear_in(decoder_input_) #self.W(encoder_outputs) #(b_size, max_len, hidden_size)

        # (b_size, hidden_size, max_len)
        encoder_outputs_ = encoder_outputs.permute(0,2,1)

        # (b_size, 1, max_len)
        output = torch.bmm(decoder_input_, encoder_outputs_)

        # (b_size, max_len)
        output = output.squeeze(1)

        # (b_size, max_len)
        weights = F.softmax(output, dim=-1)

        # (b_size, 1, max_len)
        weights = weights.unsqueeze(1)

        #print (self.linear_in)

        return weights
        #
        #
        # # ------------------------
        # output = output * decoder_input # output in (b_size, max_len, hidden_size)
        # output = torch.sum(output, dim = 2) #(b_size, max_len)
        # # ------------------------
        #
        # weights =  F.softmax(output, dim=1) # (b_size, max_len)
        #
        # return weights.unsqueeze(2) # (b_size, max_len, 1)