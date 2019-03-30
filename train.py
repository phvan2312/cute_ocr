import sys
sys.path.append('..')

from termcolor import colored
import datetime, time
import numpy as np
from PIL import Image

import json
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from ocr import utils
from ocr.model.attn_seq_seq.nn_model import NNModel
from ocr.statistics import masked_cross_entropy_at_timestep, create_ctc_loss, decode_ctc, calculate_accuracy
from ocr.loader import OCRDataset, OCRCollate

from ocr.model.crnn.model import Model, forward_one_step

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

from imgaug import augmenters as iaa
import cv2
import random

def random_dilate(img):
    img = np.array(img)
    img = cv2.dilate(img, np.ones(shape=(random.randint(1,3), random.randint(1,3)), dtype=np.uint8))

    return Image.fromarray(img)

def random_erode(img):
    img = np.array(img)
    img = cv2.erode(img, np.ones(shape=(random.randint(1,3), random.randint(1,3)), dtype=np.uint8))

    return Image.fromarray(img)

# always resize to 48
FIXED_HEIGHT = 48
def resize_to_fixed_height(img):
   img = np.array(img)
   h,w = img.shape[:2]

   new_w = int(FIXED_HEIGHT / h * w)
   img = cv2.resize(img, (new_w, FIXED_HEIGHT), interpolation=cv2.INTER_CUBIC)

   return Image.fromarray(img)

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.OneOf(
                              [
                                  iaa.GaussianBlur(sigma=(0, 3.0)),
                                  iaa.AverageBlur(k=(3, 11)),
                                  iaa.MedianBlur(k=(3, 11))
                              ])
                          ),

        ])

    def __call__(self, img):
        img = np.array(img)
        transformed_img =  self.aug.augment_image(img)

        return Image.fromarray(transformed_img)

def train_from_scratch(train_lbl_fn, train_src_root, valid_lbl_fn, valid_src_root, train_batch_size, valid_batch_size,
                       config_fn):
    global model

    config = json.load(open(config_fn))

    print_color (">>> Reading training samples ...")
    train_labels = utils.read_file_from_source(train_lbl_fn, train_src_root)

    # # just for testing
    # train_labels = {
    #     '/home/vanph/Desktop/pets/IAM_DATA/data_IAM/words/a01/a01-000u/a01-000u-00-00.png':'A',
    #     '/home/vanph/Desktop/pets/IAM_DATA/data_IAM/words/a01/a01-000u/a01-000u-00-01.png':'MOVE',
    #     '/home/vanph/Desktop/pets/IAM_DATA/data_IAM/words/a01/a01-000u/a01-000u-00-02.png':'to'
    # }

    print_color (">>> Reading validating samples ...")
    valid_labels = utils.read_file_from_source(valid_lbl_fn, valid_src_root)

    vocab = utils.build_vocab(train_labels)

    train_transfroms = transforms.Compose([

        transforms.RandomApply(
            [
                random_dilate,
            ],
            p=0.15),

        transforms.RandomApply(
            [
                random_erode,
            ],
            p=0.15),

        transforms.RandomApply(
            [
                ImgAugTransform(),
            ],
            p=0.15),

        transforms.RandomApply(
            [
                transforms.Pad(3, fill=255, padding_mode='constant'),
            ],
            p=0.15),

        transforms.RandomApply(
            [
                transforms.Pad(3, fill=255, padding_mode='reflect'),
            ],
            p=0.15),

        transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=5, resample=Image.NEAREST),
        resize_to_fixed_height,
        transforms.ToTensor()
    ])
    valid_transfroms = transforms.Compose([
        resize_to_fixed_height,
        transforms.ToTensor()
    ])

    train_dataset = OCRDataset(train_labels, vocab, transform=train_transfroms)
    valid_dataset = OCRDataset(valid_labels, vocab, transform=valid_transfroms)

    train_collate_fn = OCRCollate(enable_augment=True)
    valid_collate_fn = OCRCollate(enable_augment=False)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=train_collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, collate_fn=valid_collate_fn, num_workers=4)

    model = Model(n_classes = vocab.num_chars)
    criteria = create_ctc_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)

    model = model.to(device)
    criteria = criteria.to(device)

    print_color (colored(">>> Model information ...",'red'))
    print (model.crnn)
    #exit()

    n_step = 0
    max_step = 1000000
    valid_after_step = 5000
    run_report_every_step = 50
    started_time = time.time()
    max_grad_norm = -2

    while(n_step < max_step):

        for train_id, train_data in enumerate(train_loader):
            scheduler.step()
            model.train()

            images, _labels, labels_len = train_data[0], train_data[1], train_data[2]
            labels = model.process_label(labels=_labels, labels_len=labels_len)

            # utils.test_view_batch_tensor_images(images)
            # continue

            labels = torch.IntTensor(labels)
            labels_len = torch.IntTensor(labels_len)
            images = images.to(device)

            model.zero_grad()
            loss, preds, preds_len = forward_one_step(images=images, labels=labels, label_lens=labels_len, criteria=criteria,
                                                      model=model)
            loss.backward()

            # clip
            if max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            n_step += 1

            if n_step % run_report_every_step == 0 :
                _, pred_ids = preds.max(dim=2)
                pred_ids = pred_ids.permute(1,0).contiguous().view(-1) # (b,w)

                preds = decode_ctc(pred_ids, preds_len, vocab) # (list of text)
                tagts = [vocab.idx2sent(_label, remove_padding=True) for _label in _labels]

                acc_char, acc_field = calculate_accuracy(preds, tagts)

                report_string = make_report(loss.detach().numpy(), acc_char, acc_field, n_step, max_step,
                                            scheduler.get_lr()[0], started_time)
                print (report_string)


            if n_step % valid_after_step == 0:
                model.eval()

                valid_losses = []
                valid_preds = []
                valid_tagts = []

                with torch.no_grad():
                    for valid_id, valid_data in enumerate(valid_loader):
                        images, _labels, labels_len = valid_data[0], valid_data[1], valid_data[2]
                        labels = model.process_label(labels=_labels, labels_len=labels_len)

                        labels = torch.IntTensor(labels)
                        labels_len = torch.IntTensor(labels_len)
                        images = images.to(device)

                        loss, preds, preds_len = forward_one_step(images=images, labels=labels, label_lens=labels_len,
                                                                  criteria=criteria, model=model)

                        valid_losses += [loss.cpu().numpy()]

                        _, pred_ids = preds.max(dim=2)
                        pred_ids = pred_ids.permute(1, 0).contiguous().view(-1)  # (b,w)

                        preds = decode_ctc(pred_ids, preds_len, vocab)  # (list of text)
                        tagts = [vocab.idx2sent(_label, remove_padding=True) for _label in _labels]

                        valid_preds += preds
                        valid_tagts += tagts

                acc_char, acc_field = calculate_accuracy(valid_preds, valid_tagts)

                print ("%s Number examples: %d" % (get_time(), len(valid_preds)))
                print ("%s Validation loss: %.3f" % (get_time(), np.mean(valid_losses)))
                print ("%s Validation accuracy by char: %.3f" % (get_time(), acc_char))
                print ("%s Validation accuracy by field: %.3f" % (get_time(), acc_field))

import click
@click.command()
@click.option('--train_lbl_fn', default='/home/vanph/Desktop/pets/IAM_DATA/data_IAM/train-labels_v2.json', help='1')
@click.option('--train_src_root', default='/home/vanph/Desktop/pets/IAM_DATA/data_IAM/words')
@click.option('--train_batch_size', default=16)

@click.option('--valid_lbl_fn', default='/home/vanph/Desktop/pets/IAM_DATA/data_IAM/valid-labels.json', help='1')
@click.option('--valid_src_root', default='/home/vanph/Desktop/pets/IAM_DATA/data_IAM/words')
@click.option('--valid_batch_size', default=16)

@click.option('--config_fn',default='./configs/_config.json')
def main(train_lbl_fn, train_src_root, train_batch_size, valid_lbl_fn, valid_src_root, valid_batch_size, config_fn):
    train_from_scratch(train_lbl_fn, train_src_root, valid_lbl_fn, valid_src_root, train_batch_size, valid_batch_size,
                       config_fn)

if __name__ == "__main__":
    main()

    # train_lbl_fn = "/home/vanph/Desktop/pets/IAM_DATA/data_IAM/train-labels_v2.json"
    # train_src_root = "/home/vanph/Desktop/pets/IAM_DATA/data_IAM/words"
    # train_batch_size = 16
    #
    # valid_lbl_fn = "/home/vanph/Desktop/pets/IAM_DATA/data_IAM/valid-labels.json"
    # valid_src_root = "/home/vanph/Desktop/pets/IAM_DATA/data_IAM/words"
    # valid_batch_size = 16
    #
    # config_fn = "./configs/_config.json"
    #
    # train_from_scratch(train_lbl_fn, train_src_root, valid_lbl_fn, valid_src_root, train_batch_size, valid_batch_size,
    #                    config_fn)

