import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import SentimentDataset
from torchtext import data
import zipfile
import wget
import codecs

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

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

        if config.mode == "download":

            url = "https://keg.utcluj.ro/datasets/russu_vlad.zip"
            path_to_zip_file= wget.download(url)
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.abspath(os.getcwd()))
            path = os.path.abspath(os.getcwd()) + 'movies/'
            df = create_dataset(path)

            train_data, valid_data, test_data = np.split(df.values, [int(.8 * len(df.values)), int(.9 * len(df.values))])

            training_set = SentimentDataset(train_data)
            train_loader = DataLoader(training_set, **params)

            validation_set = SentimentDataset(valid_data)
            validation_loader = DataLoader(validation_set, **params)

            test_set = SentimentDataset(test_data)
            test_loader = DataLoader(test_set, **params)

            TEXT = data.Field(tokenize='spacy')
            LABEL = data.LabelField(dtype=torch.float)

            MAX_VOCAB_SIZE = 25_000

            TEXT.build_vocab(training_set, max_size=MAX_VOCAB_SIZE)
            LABEL.build_vocab(training_set)

            self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
                (train_loader, validation_loader, test_loader),
                batch_size=self.config.batch_size,
                device=device)


        elif config.mode == "en_download":
            # download IMDB dataset
            pass

