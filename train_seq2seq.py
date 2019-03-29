import sys
sys.path.append('..')

from termcolor import colored
import datetime, time
import numpy as np

import json
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

from ocr import utils
from ocr.model.attn_seq_seq import _decoder, _encoder, _generator
from ocr.statistics import masked_cross_entropy_at_timestep, create_ctc_loss, decode_ctc, calculate_accuracy, calculate_accuracy_seq2seq
from ocr.loader import OCRDataset, OCRCollate
from ocr.constants import *

from ocr.model.attn_seq_seq.nn_model import forward_one_step, forward_one_step_inference

#from ocr.model.crnn.model import Model, forward_one_step

model = None
epoch = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def print_color(text, color='red'):
    print (colored(text, color))

def get_time():
    return "[%s]" % str(datetime.datetime.now())

def make_report(loss, acc_char, acc_field, n_step, max_step, lr, started_time):
    report_string = "%s Step: %.5d/%.5d; acc_char: %.4f; acc_field: %.4f; lr: %.6f; loss: %.4f; %d s" \
                    % (get_time(), n_step, max_step, acc_char, acc_field, lr, loss, int(time.time() - started_time))

    return report_string

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator


    def forward(self, input_seq, input_length, max_length): # single batch only
        # Forward input through encoder model
        encoder_hidden, encoder_outputs = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden#[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def train_from_scratch(train_lbl_fn, train_src_root, valid_lbl_fn, valid_src_root, train_batch_size, valid_batch_size,
                       config_fn):
    global model

    config = json.load(open(config_fn))

    print_color (">>> Reading training samples ...")
    train_labels = utils.read_file_from_source(train_lbl_fn, train_src_root)

    print_color (">>> Reading validating samples ...")
    valid_labels = utils.read_file_from_source(valid_lbl_fn, valid_src_root)

    vocab = utils.build_vocab(train_labels)

    train_transfroms = transforms.Compose([
        transforms.ToTensor()
    ])
    valid_transfroms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = OCRDataset(train_labels, vocab, transform=train_transfroms)
    valid_dataset = OCRDataset(valid_labels, vocab, transform=valid_transfroms)

    train_collate_fn = OCRCollate()
    valid_collate_fn = OCRCollate()

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=train_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, collate_fn=valid_collate_fn)

    encoder = _encoder.Encoder()
    decoder = _decoder.Decoder(**{'vocab_size':vocab.nuform_chars})
    generator = _generator.Generator(**{'vocab_size':vocab.num_chars})


    encoder.to(device)
    decoder.to(device)
    generator.to(device)


    criteria = masked_cross_entropy_at_timestep

    optimizer = None #optim.Adam(model.parameters(), lr=0.0002)
    scheduler = None #optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.7)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr = 0.02)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = 0.02)
    generator_optimizer = optim.Adam(generator.parameters(), lr = 0.02)

    # model = model.to(device)
    # criteria = criteria.to(device)

    print_color (colored(">>> Model information ...",'red'))
    #print (model.crnn)
    #exit()

    n_step = 0
    max_step = 1000000
    valid_after_step = 500
    run_report_every_step = 50
    started_time = time.time()
    max_grad_norm = 10

    while(n_step < max_step):

        for train_id, train_data in enumerate(train_loader):
            #scheduler.step()

            #model.train()
            encoder.train()
            decoder.train()
            generator.train()

            images, labels, labels_len = train_data[0], train_data[1], train_data[2]
            #labels = model.process_label(labels=_labels, labels_len=labels_len)

            # utils.test_view_batch_tensor_images(images)
            # continue

            # processing labels and mask (ading EOS)
            masks = []
            for (label, label_len) in zip(labels, labels_len):
                label.insert(label_len, EOS_token)
                mask = [0] * len(label)
                mask[:label_len + 1] = [1] * (label_len + 1)

                masks += [mask]

            labels = torch.LongTensor(labels)
            masks = torch.ByteTensor(masks)

            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            #model.zero_grad()
            encoder.zero_grad()
            decoder.zero_grad()
            generator.zero_grad()

            loss, preds, printed_loss = forward_one_step(images=images, labels=labels, masks=masks, criteria=criteria,
                                                      model=(encoder, decoder, generator),device=device)
            loss.backward()

            # clip
            if max_grad_norm > 0:
                _ = clip_grad_norm_(encoder.parameters(), max_grad_norm)
                _ = clip_grad_norm_(decoder.parameters(), max_grad_norm)
                _ = clip_grad_norm_(generator.parameters(), max_grad_norm)

            encoder_optimizer.step()
            decoder_optimizer.step()
            generator_optimizer.step()

            n_step += 1

            if n_step % run_report_every_step == 0 :
                # _, pred_ids = preds.max(dim=2)
                # pred_ids = pred_ids.permute(1,0).contiguous().view(-1) # (b,w)

                #preds = decode_ctc(pred_ids, preds_len, vocab) # (list of text)
                #tagts = [vocab.idx2sent(_label, remove_padding=True) for _label in _labels]

                acc_char, acc_field = calculate_accuracy_seq2seq(preds=preds.cpu().tolist(), tgts=labels.cpu().tolist(), vocab=vocab)

                # acc_char, acc_field = torch.sum(labels.cpu() == preds, dtype=torch.float32) / preds.numel(), \
                #                       torch.sum((labels.cpu() == preds).all(dim=1), dtype=torch.float32) / preds.size(0)
                #
                # acc_char  = acc_char.item()
                # acc_field = acc_field.item()

                report_string = make_report(printed_loss, acc_char, acc_field, n_step, max_step,
                                            0.0002, started_time)
                print (report_string)

            if n_step % valid_after_step == 0:
                encoder.eval()
                decoder.eval()
                generator.eval()

                valid_losses = []
                valid_preds = []
                valid_tagts = []

                with torch.no_grad():
                    for valid_id, valid_data in enumerate(valid_loader):
                        images, labels, labels_len = valid_data[0], valid_data[1], valid_data[2]

                        # processing labels and mask (ading EOS)
                        masks = []
                        for (label, label_len) in zip(labels, labels_len):
                            label.insert(label_len, EOS_token)
                            mask = [0] * len(label)
                            mask[:label_len + 1] = [1] * (label_len + 1)

                            masks += [mask]

                        labels = torch.LongTensor(labels)

                        images = images.to(device)
                        labels = labels.to(device)

                        #labels = torch.LongTensor(labels)
                        preds = forward_one_step_inference(images, model=(encoder, decoder, generator),device=device)
                        #calculate_accuracy_seq2seq(preds=preds.cpu().tolist(), tgts=labels.cpu().tolist(), vocab=vocab)
                        valid_preds += preds.cpu().tolist()
                        valid_tagts += labels.cpu().tolist()

                acc_char, acc_field = calculate_accuracy_seq2seq(valid_preds, valid_tagts,vocab=vocab)

                print ("%s Number examples: %d" % (get_time(), len(valid_preds)))
                #print ("%s Validation loss: %.3f" % (get_time(), np.mean(valid_losses)))
                print ("%s Validation accuracy by char: %.3f" % (get_time(), acc_char))
                print ("%s Validation accuracy by field: %.3f" % (get_time(), acc_field))



if __name__ == "__main__":
    train_lbl_fn = "/home/vanph/Desktop/pets/IAM_DATA/data_IAM/train-labels_v2.json"
    train_src_root = "/home/vanph/Desktop/pets/IAM_DATA/data_IAM/words"
    train_batch_size = 16

    valid_lbl_fn = "/home/vanph/Desktop/pets/IAM_DATA/data_IAM/valid-labels.json"
    valid_src_root = "/home/vanph/Desktop/pets/IAM_DATA/data_IAM/words"
    valid_batch_size = 16

    config_fn = "./configs/_config.json"

    train_from_scratch(train_lbl_fn, train_src_root, valid_lbl_fn, valid_src_root, train_batch_size, valid_batch_size,
                       config_fn)

