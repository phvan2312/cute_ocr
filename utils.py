import json
import os
import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from ocr.vocab import Vocab
from ocr.constants import *

import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def imgshow(img):
    plt.imshow(img)
    plt.show()

def tensor_imgshow(tensor_img):
    img = transforms.ToPILImage(mode='L')(tensor_img)
    return imgshow(img)

def pad_batch_image_tensor(tensor_images, enable_augment = False):
    """
    :param tensor_images: list(c,h,w)
    :return: tensor(n_batch, max_height, max_width, n_channel)
    """
    c = tensor_images[0].size(0)
    h = max([e.size(1) for e in tensor_images])
    w = max([e.size(2) for e in tensor_images])

    batch_images = torch.zeros(len(tensor_images), c, h, w).fill_(1)

    for i, image in enumerate(tensor_images):
        #(b, c,  h, w)

        started_h = max(0, random.randint(0, h - image.size(1)) )
        started_w = max(0, random.randint(0, w - image.size(2)) )

        if enable_augment is False:
            started_h = 0
            started_w = 0

        batch_images[i,:, started_h:started_h+image.size(1),started_w:started_w+image.size(2)] = image

    return batch_images

def pad_batch_label_tensor(tensor_labels):
    """
    :param tensor_labels: list(l)
    :return: list(n_batch, max_len)
    """
    l = max([len(e) for e in tensor_labels])
    batch_labels = []

    for i, label in enumerate(tensor_labels):
        batch_labels += [label + [PAD_token] * (l - len(label))]

    return batch_labels


def read_file_from_source(lbl_fn, src_root, processing_func = lambda x: x):
    labels = json.load(open(lbl_fn,'r'))
    print ("Length: ", len(labels))

    new_labels = {}
    for img_path, img_lbl in labels.items():
        new_img_path = os.path.join(src_root, img_path)
        assert os.path.exists(new_img_path)

        new_labels[new_img_path] = processing_func(img_lbl)

    return new_labels

def build_vocab(train_labels, existed_vocab = None):
    vocab = Vocab() if existed_vocab is None else existed_vocab

    for img_path, img_label in train_labels.items():
        vocab.add_sentence(img_label)

    return vocab

def test_view_batch_tensor_images(images):
    n_images = len(images)

    for i in range(n_images):
        tensor_imgshow(images[i])



