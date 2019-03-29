import torch
import torch.nn as nn

from ocr.model.attn_seq_seq._decoder import Decoder
from ocr.model.attn_seq_seq._encoder import Encoder
from ocr.model.attn_seq_seq._generator import Generator

from ocr.constants import *

class NNModel(nn.Module):
    def __init__(self, loss_func, config_json):
        super(NNModel, self).__init__()

        self.config_json = config_json

        self.encoder = Encoder() #Encoder(**self.config_json)
        self.decoder = Decoder() # Decoder(**self.config_json)
        self.generator = Generator(**self.config_json)

        self.max_len = 150 #self.config_json.get('max_len')
        self.device  = torch.device('cuda:0') #self.config_json.get('device')

        self.loss_func = loss_func

    def forward(self, images, labels, labels_len, mode='train'):
        #input --gray_image :(b_size,1,height,width)
        #target: (b_size, max_len)
        #mask: (b_size, max_len)

        b_size = images.size(0)
        mask = None

        decoder_hidden, encoder_outputs = self.encoder(images) # (b, 256, )

        decoder_input = torch.LongTensor([SOS_token for _ in range(b_size)]) # (b_size, )
        decoder_input = decoder_input.to(self.device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # teacher forcing
        for t in range(self.max_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_outputs, decoder_hidden)
            generator_output = self.generator(decoder_output)

            masked_loss, n_total = self.loss_func(generator_output, labels[:, t], mask[:, t], self.device)

            decoder_input = labels[:, t]

            loss += masked_loss
            print_losses.append(masked_loss.item() * n_total)
            n_totals += n_total

        return loss, print_losses, n_totals

def forward_one_step_inference(images, model, device, max_len = 20):
    (encoder, decoder, generator) = model

    b, c, h, w = images.size()

    # (num_rnn_layer, b, hidden_size) * 2, (max_len, b, hidden_size)
    decoder_pre_state, encoder_outputs = encoder(images)

    hidden_size = encoder_outputs.size(2)
    h_size = (b, hidden_size)

    # (b, hidden_size)
    decoder_pre_context_vector = decoder_pre_state[0].data.new(*h_size).zero_()

    # (b)
    decoder_input = torch.LongTensor([SOS_token] * b).to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    all_tokens = torch.LongTensor([SOS_token] * b).view(1, -1)

    for t in range(max_len):
        # (b, hidden), (num_layer, b, hidden) * 2
        decoder_output_t, decoder_pre_state = decoder(decoder_input, encoder_outputs, decoder_pre_state,
                                                      decoder_pre_context_vector)

        # (b, vocab_size)
        generator_output_t = generator(decoder_output_t)

        _, decoder_input = generator_output_t.max(dim=1)

        decoder_pre_context_vector = decoder_output_t

        pred_scores, pred_tokens = generator_output_t.cpu().max(dim=1)

        all_tokens = torch.cat((all_tokens, pred_tokens.unsqueeze(0)), dim=0)

    return all_tokens.permute(1, 0).contiguous()[:,1:]

def forward_one_step(images, labels, masks, criteria, model, device, **kwrags):
    (encoder, decoder, generator) = model

    b,c,h,w = images.size()
    _,max_len = labels.size()

    # (num_rnn_layer, b, hidden_size) * 2, (max_len, b, hidden_size)
    decoder_pre_state, encoder_outputs = encoder(images)

    hidden_size = encoder_outputs.size(2)
    h_size = (b, hidden_size)

    # (b, hidden_size)
    decoder_pre_context_vector  = decoder_pre_state[0].data.new(*h_size).zero_()

    # (b)
    decoder_input = torch.LongTensor([SOS_token] * b).to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    all_tokens = torch.LongTensor([SOS_token] * b).view(1,-1)

    for t in range(max_len):
        # (b, hidden), (num_layer, b, hidden) * 2
        decoder_output_t, decoder_pre_state = decoder(decoder_input, encoder_outputs, decoder_pre_state,
                                                      decoder_pre_context_vector)

        # (b, vocab_size)
        generator_output_t = generator(decoder_output_t)

        masked_loss_t, n_total_t = criteria(generator_output_t, labels[:,t], masks[:,t])

        # teacher forcing
        decoder_input = labels[:, t]
        # ---

        decoder_pre_context_vector = decoder_output_t

        loss += masked_loss_t
        print_losses.append(masked_loss_t.item() * n_total_t)
        n_totals += n_total_t

        pred_scores, pred_tokens = generator_output_t.cpu().max(dim=1)

        all_tokens = torch.cat((all_tokens, pred_tokens.unsqueeze(0)), dim=0)

    return loss, all_tokens.permute(1, 0).contiguous()[:,1:], sum(print_losses) / n_totals
