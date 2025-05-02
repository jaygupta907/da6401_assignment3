import os
import torch
from torch.utils.data import Dataset
from collections import Counter

SPECIAL_TOKENS = {'<pad>': 0, '<sos>': 1, '<eos>': 2}

class CharVocab:
    def __init__(self, tokens):
        unique_chars = sorted(set(char for token in tokens for char in token))
        self.char2idx = {char: idx + len(SPECIAL_TOKENS) for idx, char in enumerate(unique_chars)}
        self.char2idx.update(SPECIAL_TOKENS)
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def encode(self, word):
        return [self.char2idx['<sos>']] + [self.char2idx[c] for c in word] + [self.char2idx['<eos>']]

    def decode(self, indices):
        return ''.join(self.idx2char[i] for i in indices if i not in (self.char2idx['<sos>'], self.char2idx['<eos>'], self.char2idx['<pad>']))

    def __len__(self):
        return len(self.char2idx)


class DakshinaDataset(Dataset):
    def __init__(self, path):
        self.deva_words = []
        self.latin_words = []

        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                self.deva_words.append(parts[0])
                self.latin_words.append(parts[1])

        self.input_vocab = CharVocab(self.latin_words)
        self.output_vocab = CharVocab(self.deva_words)

        self.data = [(self.input_vocab.encode(lat), self.output_vocab.encode(dev))
                     for lat, dev in zip(self.latin_words, self.deva_words)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    input_seqs, target_seqs = zip(*batch)
    input_lens = [len(seq) for seq in input_seqs]
    target_lens = [len(seq) for seq in target_seqs]

    input_pad = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in input_seqs], batch_first=True, padding_value=0)
    target_pad = torch.nn.utils.rnn.pad_sequence([torch.tensor(y) for y in target_seqs], batch_first=True, padding_value=0)

    return input_pad, target_pad, input_lens, target_lens
