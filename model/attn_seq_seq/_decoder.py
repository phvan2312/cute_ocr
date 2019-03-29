import torch
import torch.nn as nn
import torch.nn.functional as F
from ocr.model.attn_seq_seq.attentions.dot_attention import DotAttention

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

        self.rnn_hidden_size = 256 #kwargs.get('rnn_hidden_size')
        self.char_embedding_size = 256 # kwargs.get('embedding_size')
        self.rnn_n_layer = 2 #kwargs.get('rnn_n_layer')
        self.dropout = 0.1 #kwargs.get('dropout')

        self.embedding = nn.Embedding(kwargs.get('vocab_size') ,self.char_embedding_size) #kwargs.get('embedding')
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.attention_h_dropout = nn.Dropout(self.dropout)


        self.rnn_cell = nn.LSTM(self.char_embedding_size + self.rnn_hidden_size, self.rnn_hidden_size, self.rnn_n_layer,
                                dropout=self.dropout,
                                bidirectional=False)

        # self.rnn = nn.LSTM(256) #nn.LSTMCell(self.char_embedding_size,self.rnn_hidden_size)


        self.attention_mechanism = DotAttention(**kwargs)
        self.concat_linear = nn.Linear(self.rnn_hidden_size * 2, self.rnn_hidden_size)

        self.concat = nn.Linear(self.rnn_hidden_size * 2, self.rnn_hidden_size)

    def forward(self, input_at_timestep, encoder_outputs, last_state, decoder_pre_context_vector):
        # input_at_timestep: (batch_size, )
        # encoder_outputs: (max_len, batch_size, hidden_size)
        # last_hidden_state: (batch_size, hidden_size)
        #
        # output:

        #(n_layer, batch_size, hidden_dim) * 2
        h_prev_t, c_prev_t = last_state

        #(b, max_len, hidden_size)
        encoder_outputs_ = encoder_outputs.permute(1,0,2)

        #(b, char_embedding)
        embedding = self.embedding(input_at_timestep)
        #embedding = self.embedding_dropout(embedding)

        #(b, char_embedding + hidden_size)
        embedding = torch.cat([embedding, decoder_pre_context_vector], dim=1) # (b, c_emb)

        #(1,b,char_embedding + hidden_size)
        embedding = embedding.unsqueeze(0)

        # (1, b, hidden_size), ...
        rnn_out, (h_t, c_t) = self.rnn_cell(embedding, (h_prev_t, c_prev_t)) #out:(1, b, h), last_state:(b,h)

        # (b, hidden_size)
        rnn_out = rnn_out.squeeze(0)

        # (b, 1, max_len)
        weights = self.attention_mechanism(encoder_outputs_, rnn_out)

        # (b, 1, hidden_size)
        context_vector = torch.bmm(weights, encoder_outputs_)

        # (b, 1, hidden_size)
        rnn_out_ = rnn_out.unsqueeze(1)

        # (b, 1, hidden_size * 2)
        concat_c = torch.cat([context_vector, rnn_out_], dim=2)

        # (b, 1, hidden_size)
        attn_h   = torch.tanh(self.concat_linear(concat_c))

        # (b, hidden_size)
        attn_h = self.attention_h_dropout(attn_h.squeeze(1))

        return attn_h, (h_t, c_t)

        #
        # context_vector = torch.sum(weights * encoder_outputs, dim=1) #(b, max_len, hidden_size) -> (b, hidden_size)
        #
        # concat_vector = torch.cat((context_vector, h_t), dim=1) #(b, 2 * hidden_size)
        # concat_vector = torch.tanh(self.concat(concat_vector)) #(b, hidden_size)
        #
        # return concat_vector, (h_t, c_t)





