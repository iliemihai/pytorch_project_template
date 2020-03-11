import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datasets.sentiment_dataset import SentimentDataset
from torchtext import data
import zipfile
import wget
import codecs
from tokenizers import (ByteLevelBPETokenizer,
                            CharBPETokenizer,
                            SentencePieceBPETokenizer,
                            BertWordPieceTokenizer)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#torch.backends.cudnn.benchmark = True

#tokenizer_en = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)

tokenizer_ro = BertWordPieceTokenizer("./roberto-vocab.txt", lowercase=True)

#def tokenize_en(text):
#    return tokenizer_en.encode(text).tokens

def tokenize_ro(text):
    return tokenizer_ro.encode(text).tokens

def create_dataset(path):
    df = pd.DataFrame(columns=['review', 'sentiment'])
    for directory in os.listdir(path):
        if "pos" in directory:
            for filename in os.listdir(path + directory):
                with codecs.open(os.path.join(path + directory, filename), encoding="utf-8", errors="ignore") as f:
                    observation = f.read()
                    current_df = pd.DataFrame({'review': [observation], 'sentiment': 1})
                    df = df.append(current_df, ignore_index=True)
        else:
            for filename in os.listdir(path + directory):
                with codecs.open(os.path.join(path + directory, filename), encoding="utf-8", errors="ignore") as f:
                    observation = f.read()
                    current_df = pd.DataFrame({'review': [observation], 'sentiment': 0})
                    df = df.append(current_df, ignore_index=True)
    return df

class SentimentDataLoader:
    def __init__(self, config):
        """
        :param config
        """
        self.config = config
        params = {"batch_size":self.config.batch_size,
                  "shuffle": True,
                  "num_workers": self.config.data_loader_workers,
                  "pin_memory": self.config.pin_memory
                  }

        self.TEXT = data.Field(tokenize=tokenize_ro)
        self.LABEL = data.LabelField(dtype=torch.float)

        if config.mode == "download":
            url = "https://keg.utcluj.ro/datasets/russu_vlad.zip"
            path = os.path.abspath(os.getcwd()) + '/movies/'
            if not os.path.isdir(path):
                path_to_zip_file= wget.download(url)
                with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(os.path.abspath(os.getcwd()))

            df = create_dataset(path)

            train_data, valid_data, test_data = np.split(df.values, [int(.8 * len(df.values)), int(.9 * len(df.values))])

            self.training_set = SentimentDataset(train_data,tokenize_ro)
            self.train_loader = DataLoader(self.training_set, **params)

            self.validation_set = SentimentDataset(valid_data, tokenize_ro)
            self.validation_loader = DataLoader(self.validation_set, **params)

            test_set = SentimentDataset(test_data, tokenize_ro)
            self.test_loader = DataLoader(test_set, **params)

            self.TEXT.build_vocab(self.training_set.list_IDs, max_size=self.config.max_vocab_size)
            self.LABEL.build_vocab(self.training_set.list_IDs)

            self.VOCAB_SIZE = len(self.TEXT.vocab)
            #data_fields = [('review', self.TEXT), ('sentiment', self.LABEL)]
            #self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            #    (train_loader, validation_loader, test_loader),
            #    batch_size=self.config.batch_size,
            #    device=device,
            #    sort_key=lambda x: len(x.text),
            #    shuffle=False)


        elif config.mode == "en_download":
            # download IMDB dataset
            pass

