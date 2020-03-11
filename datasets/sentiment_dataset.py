import torch
import torchtext
import tokenizers
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, data, word_tokenizer):
        super().__init__()
        """Initialization"""
        self.labels = [[row[1]] for row in data]
        self.list_IDs = [[row[0]] for row in data]
        self.word_tokenizer = word_tokenizer
        self.vocab_size = 0
        self.vocab = self.build_vocab(self.list_IDs)
        self.word2idx, self.idx2word = self.word2index(self.vocab)


    def build_vocab(self, corpus):
        word_count = {}
        for sentence in corpus:
            tokens = self.word_tokenizer(sentence[0])
            for tok in tokens:
                if tok not in word_count:
                    word_count[tok] = 1
                    self.vocab_size += 1
                else:
                    word_count[tok] += 1
        return word_count

    def word2index(self, word_count):
        word_index = {w: i for i, w in enumerate(word_count)}
        idx_word = {i: w for i, w in enumerate(word_count)}
        return word_index, idx_word

    def __len__(self):
        """Denotes total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates the sample of data"""
        if torch.is_tensor(index):
            index = index.tolist()
        inputs_ = self.word_tokenizer(self.list_IDs[index][0])
        inputs_ = [self.word2idx[w] for w in inputs_]

        # label = {"positive":1, "negative":0}
        label_ = self.labels[index]
        # Load data and get label

        return inputs_, label_
