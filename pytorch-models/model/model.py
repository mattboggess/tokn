import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class MaxPoolBaseline(BaseModel):
    """
    Arbritrary baseline model that collapses LSTM output with random word embeddings into
    single vector that can be max-pooled across sentences. Sanity check.
    """
    def __init__(self, num_embeddings, embedding_dim, hidden_size):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.maxpool = nn.MaxPool1d(1)
        self.fc = nn.Linear(hidden_size * 50, 2)
    
    def forward(self, x):
        x = self.embeddings(x)
        old_size = x.size()
        x = x.view(-1, old_size[-2], old_size[-1])
        x = self.lstm(x)[0]
        x = x.view(old_size[0], old_size[1], -1)
        x = x.permute(0, 2, 1)
        x = self.maxpool(x)
        x = x.squeeze(-1)
        return x
