import jaconv
from .constants import *

class Tokenizer:
    def __init__(self):
        pass

    def __normalize(self, text):
        return jaconv.normalize(text)

    def token(self, sentece):
        normed_sentence = self.__normalize(sentece)

        return [c for c in normed_sentence]

class Vocab:
    def __init__(self):
        self.char2index = {}
        self.char2count = {}
        self.index2char = {CTC_BLANK_token:"CTC_BLANK" ,PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_chars = len(self.index2char)  # Count CTC_BLANK,SOS, EOS, PAD, UNK

        self.tokenizer = Tokenizer()

    def add_sentence(self, sentence):
        for c in self.tokenizer.token(sentence):
            self.add_character(character=c)

    def add_character(self, character):
        if character not in self.char2index:
            self.char2index[character] = self.num_chars
            self.char2count[character] = 1
            self.index2char[self.num_chars] = character
            self.num_chars += 1
        else:
            self.char2count[character] += 1

    def sent2idx(self, sentence):
        idx = [self.char2index[c] if c in self.char2index else UNK_token for c in sentence]
        return idx

    def idx2sent(self, idx, remove_padding=True):
        cs = [self.index2char[id] for id in idx if (remove_padding and id != PAD_token)]

        return "".join(cs)




