import torch
from torch import nn
import torch.nn.functional as F
from warpctc_pytorch import CTCLoss

import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layer1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)

    def forward(self, src):
        batch_size = src.size(0)

        # (batch_size, 64, imgH, imgW)
        # layer 1
        src = F.relu(self.layer1(src[:, :, :, :] - 0.5), True)

        # (batch_size, 64, imgH/2, imgW/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        # (batch_size, 128, imgH/2, imgW/2)
        # layer 2
        src = F.relu(self.layer2(src), True)

        # (batch_size, 128, imgH/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        #  (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer 3
        # batch norm 1
        src = F.relu(self.batch_norm1(self.layer3(src)), True)

        # (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer4
        src = F.relu(self.layer4(src), True)

        # (batch_size, 256, imgH/2/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(1, 2), stride=(1, 2))

        # (batch_size, 512, imgH/2/2/2, imgW/2/2)
        # layer 5
        # batch norm 2
        src = F.relu(self.batch_norm2(self.layer5(src)), True)

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 1), stride=(2, 1))

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.relu(self.batch_norm3(self.layer6(src)), True)

        return src

class SingleHeadAttention(nn.Module):
    def __init__(self, input_dim):
        super(SingleHeadAttention, self).__init__()

        self.linear_in_Q = nn.Linear(input_dim, input_dim)
        self.linear_in_K = nn.Linear(input_dim, input_dim)
        self.linear_in_V = nn.Linear(input_dim, input_dim)

    def forward(self, src):
        # src: (b, max_len, h)

        b, max_len, h = src.size()

        Q = self.linear_in_Q(src) # -> decoder
        K = self.linear_in_K(src) # -> encoder
        V = self.linear_in_V(src) # -> encoder

        # (b, h, max_len)
        K_ = K.permute(0,2,1)

        # (b, max_lenQ, max_lenK)
        scores = torch.bmm(Q, K_) / math.sqrt(float(h))

        # (b, max_len, max_len)
        weights = F.softmax(scores, dim=-1)

        # (b, max_len, h)
        attn_h = torch.bmm(weights, V)

        return attn_h


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):


        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim = 256 * 2, num_directions = 2, num_layers = 1, dropout = 0.1,
                 bidirectional = True, n_classes = 200, n_head = 1):
        super(Decoder, self).__init__()

        self.rnn1 = nn.LSTM(input_size=int(input_dim), hidden_size=int(hidden_dim / num_directions),
                            num_layers=num_layers, dropout=0, bidirectional=bidirectional)

        self.rnn2 = nn.LSTM(input_size=int(hidden_dim), hidden_size=int(hidden_dim / num_directions),
                           num_layers=num_layers, dropout=0,bidirectional=bidirectional)

        self.logit = nn.Linear(int(hidden_dim), n_classes, bias=False)

        self.norm1 = Norm(48 // 8 * 512)
        self.norm2 = Norm(hidden_dim)
        self.norm3 = Norm(hidden_dim)

        #self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

        # self.norm4 = Norm(hidden_dim)
        #
        #self.attn1 = MultiHeadAttention(1, 512 * 48 // 8, dropout=0.1)
        # self.attn2 = MultiHeadAttention(1, hidden_dim, dropout=0.1)
        #
        # self.ff = FeedForward(hidden_dim, hidden_dim)

        #
        # self.rnn2 = nn.LSTM(input_size=int(hidden_dim), hidden_size=int(hidden_dim / num_directions),
        #                     num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        #
        # self.transformers = [SingleHeadAttention(int(hidden_dim)).to(device) for _ in range(n_head)]
        # self.transformers_liner_out = nn.Linear(int(hidden_dim * n_head), int(hidden_dim))

    def forward(self, src):
        #input: (b, c ,48//8 , w)
        b,c,h,w = src.size()

        #(w, b, c, 48 // 8 = h)
        src = src.permute(3,0,1,2)

        #(w, b, c * h)
        src = src.view(w,b,c*h)


        #(b, w, ch)
        #_src = src.permute(1,0,2)

        #(w, b, ch)
        #src = self.norm1(src + self.attn1(q=_src, k=_src, v=_src).permute(1,0,2))

        #(w, b, hid)
        src = self.norm1(src)
        outputs, _ = self.rnn1(src)
        outputs1 = self.norm2(self.dropout2(outputs))
        outputs, _ = self.rnn2(outputs1)
        outputs = self.norm3(self.dropout3(outputs) + outputs1)

        #(w, b, hid)
        #outputs = self.norm2(outputs)

        #(w, b, hid)
        #outputs, _ = self.rnn2(outputs)



        # #(b, w, hid)
        # outputs = outputs.permute(1,0,2)
        #

        #
        # #
        # outputs = outputs + self.attn1(_outputs, _outputs, _outputs)
        #
        # _outputs = self.norm3(outputs)
        # outputs = outputs + self.attn2(_outputs, _outputs, _outputs)
        #
        # _outputs = self.norm4(outputs)
        # outputs = outputs + self.ff(_outputs)
        #
        # outputs = outputs.permute(1,0,2)

        # outputs_ = outputs.permute(1,0,2)
        #
        # trans_outputs = []
        # for id, transformer in enumerate(self.transformers):
        #     _output = transformer(outputs_)
        #     trans_outputs += [_output]
        #
        # # (b , w, hid_dim)
        # trans_output = torch.cat(trans_outputs, dim=-1)
        #
        # # (w, b, hid_dim)
        # trans_output_ = trans_output.permute(1,0,2)
        # #
        # # # (w, b, hid_dim)
        # outputs, _ = self.rnn2(trans_output_)

        # (w, b, n_classes)
        logit = self.logit(outputs)

        return logit

class Model(nn.Module):
    def __init__(self, n_classes, fixed_height = 48):
        super(Model, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(input_dim=fixed_height * 512 / 8, n_classes=n_classes)

        self.crnn = nn.Sequential(
            self.encoder,
            self.decoder
        )

    def forward(self, input):
        output = self.crnn(input)

        return output

    def process_label(self, labels, labels_len):
        assert len(labels) == len(labels_len)

        new_labels = []
        for label, label_len in zip(labels, labels_len):
            new_labels += label[:label_len]

        return new_labels
#
# class Manager:
#     def __init__(self, vocab, device, **kwargs):
#         self.model = Model(n_classes=vocab.num_chars)
#         self.criteria = CTCLoss()
#
#         self.model = self.model.to(device)
#         self.criteria = self.criteria.to(device)
#
#     def forward_one_step(self, images, labels, label_lens):
#         pass
#
#



def forward_one_step(images, labels, label_lens, criteria, model, **kwrags):
    """ctc_loss
    forward(acts, labels, act_lens, label_lens)
    # acts: Tensor of (seqLength x batch x outputDim) containing output activations from network (before softmax)
    # labels: 1 dimensional Tensor containing all the targets of the batch in one large sequence
    # act_lens: Tensor of size (batch) containing size of each output sequence from the network
    # label_lens: Tensor of (batch) containing label length of each example
    """

    acts = model(images) # (w,b,n_classes)
    w, b, _ = acts.size()
    act_lens = torch.IntTensor([w] * b)

    loss = criteria(acts, labels, act_lens, label_lens) / b

    return loss, acts, act_lens

if __name__ == "__main__":
    pass