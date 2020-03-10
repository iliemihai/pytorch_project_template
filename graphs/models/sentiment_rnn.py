import torch
import torch.nn as nn
import torch.nn.functional as F
from ..weights_initializer import weights_init


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        """
        text_dim = [sent_len x batch_dim]
        embedded_dim = [sent_dim x batch_dim x emb_dim]
        output_dim = [sent_dim x batch_dim x emb_dim]
        hidden_dim = [1 x batch_dim x emb_dim]
        """
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))
