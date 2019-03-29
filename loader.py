import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from ocr.constants import *
import ocr.utils as utils


class OCRDataset(Dataset):
    def __init__(self, labels_json, vocab, transform=transforms.ToTensor(), fixed_height=48):
        super(OCRDataset, self).__init__()

        self.labels = labels_json
        self.id2paths = {i:path for i, path in enumerate(self.labels.keys())}

        self.vocab  = vocab
        self.trasform = transform
        self.fixed_height = fixed_height


    def __len__(self):
        return len(self.id2paths)

    # ToTensor: from PIL Image(w,h) --> Tensor (C, H, W)
    def __getitem__(self, id):
        img_path  = self.id2paths[id]
        img_label = self.labels[img_path]
        img_label_idx = self.vocab.sent2idx(img_label)

        img = Image.open(img_path).convert('L')

        # if self.fixed_height > 0:
        #     (w, h) = img.size
        #
        #     new_w = int(self.fixed_height / h * w)
        #     img = img.resize(size=(new_w, self.fixed_height), resample=Image.ANTIALIAS)

            #
            # (wt, ht) = (100, 48)
            #
            # if h > self.fixed_height // 2:
            #     new_w = int(ht / h * w)
            #     img = img.resize(size=(new_w, self.fixed_height), resample=Image.ANTIALIAS)
            # else:
            #     new_img = Image.new('L', (w,ht),color=255)
            #     new_img.paste(img, box=(0,0))
            #
            #     img = new_img

        img = self.trasform(img)

        return img, img_label_idx, len(img_label_idx)

class OCRCollate:
    def __init__(self, enable_augment = False):
        self.enable_augment = enable_augment

    def __call__(self, batchs):
        images, labels, labels_len = zip(*batchs)

        new_images  = utils.pad_batch_image_tensor(images, enable_augment = self.enable_augment)
        new_labels  = utils.pad_batch_label_tensor(labels)

        return new_images, new_labels, labels_len