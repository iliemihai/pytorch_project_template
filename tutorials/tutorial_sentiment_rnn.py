import torch
from torchtext import data

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)

"""or use IMDB dataset"""

from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

print(vars(train_data.examples[0]))

import random

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

def tokenize_en(text):
    return tokenizer_en.encode(text).tokens[::-1]

def tokenize_ro(text):
    return tokenizer_ro.encode(text).tokens

ro_lang = Field(tokenize = tokenize_ro,
           lower = True)

en_lang  = Field(tokenize=tokenize_en,
            lower=True)


ro_lang.build_vocab(train, valid, test)

en_lang.build_vocab(train, valid, test)



BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

