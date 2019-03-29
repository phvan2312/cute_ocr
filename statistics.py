import torch
import editdistance
from warpctc_pytorch import CTCLoss
from ocr.constants import *

def masked_cross_entropy_at_timestep(input, target, mask):
    #input:(b_size, vocab_size)
    #target:(b_size,)
    #mask(b_size,)

    total = torch.sum(mask)
    q = torch.gather(input, 1, target.view(-1,1)).squeeze(1) # (b_size,)
    log_q = torch.log(q)

    xentropy = -1.0 * log_q
    xentropy = xentropy.masked_select(mask).mean()

    xentropy = xentropy.to(torch.device('cuda:0'))

    return xentropy, total.item()

def create_ctc_loss():
    ctc_loss = CTCLoss()
    return ctc_loss

def decode_ctc(t, length, vocab, raw=False):
    if length.numel() == 1:
        length = length[0]
        assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
        if raw:
            return ''.join([vocab.index2char[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    #char_list.append(vocab.index2char[t[i]])
                    char_list.append(vocab.index2char[t[i].cpu().tolist()])
            return ''.join(char_list)
    else:
        # batch mode
        assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                            length.sum())
        texts = []
        index = 0
        for i in range(length.numel()):
            l = length[i]
            texts.append(
                decode_ctc(
                    t[index:index + l], torch.IntTensor([l]), vocab, raw=raw))
            index += l
        return texts

def calculate_accuracy_seq2seq(preds, tgts, vocab):
    assert len(preds) == len(tgts)

    normed_preds, normed_tgts = [], []

    for pred, tgt in zip(preds, tgts):
        end_pred_id = pred.index(EOS_token) if EOS_token in pred else len(pred)
        end_tgt_id  = tgt.index(EOS_token) if EOS_token in tgt else len(tgt)

        pred_ids = pred[:end_pred_id]
        tgt_ids  = tgt[:end_tgt_id]

        pred = vocab.idx2sent(pred_ids)
        tgt  = vocab.idx2sent(tgt_ids)

        normed_preds += [pred]
        normed_tgts  += [tgt]

    return calculate_accuracy(normed_preds, normed_tgts)

def calculate_accuracy(preds, tgts):
    """
    :param pred: (b,w)
    :param tgt: (b,w)
    :return: score
    """

    total_char_error  = 0
    total_field_error = 0

    total_char  = 0 #sum([len(tgt) for tgt in tgts])
    total_field = len(tgts)

    for pred, tgt in zip(preds, tgts):
        n_char_error = editdistance.eval(pred, tgt)
        total_char_error += n_char_error

        if pred != tgt: total_field_error += 1

        total_char += max(len(pred), len(tgt))

    return 1. - total_char_error / float(total_char), 1. - total_field_error / float(total_field)


