import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

        self.image_channel_size = 1 #kwargs.get('image_channel_size')
        self.rnn_hidden_size = 256 #kwargs.get('rnn_hidden_size')
        self.rnn_n_layer = 2 #kwargs.get('rnn_n_layer')
        self.dropout = 0.1 #kwargs.get('dropout')

        self.layer1 = nn.Conv2d(self.image_channel_size, 64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)

        src_size = 512
        self.rnn = nn.LSTM(src_size, self.rnn_hidden_size,
                           num_layers=self.rnn_n_layer,
                           dropout=self.dropout,
                           bidirectional=False)
        self.pos_lut = nn.Embedding(1000, src_size)

    def forward(self, src):
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""

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

        # # (batch_size, 512, H, W)
        all_outputs = []
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2) \
                .transpose(1, 2)
            row_vec = torch.Tensor(batch_size).type_as(inp.data) \
                .long().fill_(row)
            pos_emb = self.pos_lut(row_vec)
            with_pos = torch.cat(
                (pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0)
            outputs, state_t = self.rnn(with_pos)
            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)

        return state_t, out # (n_layer, b, 256) * 2, (w * h + h, b, 256)