import torch
from torch import nn
import math

class TransformerEncoderModel(nn.Module):
    def __init__(self, num_embeddings, d_model, padding_idx, nhead, dim_feedforward,num_layers):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_embeddings)
    
    def forward(self, x):
        padding_mask = self.get_padding_mask(x)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.fc(x)
        return x
    
    def get_padding_mask(self, x):
        #x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=1)
        padding_mask = (x == self.padding_idx)
        return padding_mask

# https://towardsdev.com/positional-encoding-in-transformers-using-pytorch-63b5c3f57d54
# Numbers seems to come from "Attention is all you need" paper
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    

class TransformerClassifierModel(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(nn.Linear(encoder.embedding.embedding_dim, encoder.embedding.embedding_dim), nn.ReLU(), nn.Linear(encoder.embedding.embedding_dim, n_classes))
    
    def forward(self, x):
        padding_mask = self.encoder.get_padding_mask(x)
        x = self.encoder.embedding(x)
        x = self.encoder.positional_encoding(x)
        x = self.encoder.encoder(x, src_key_padding_mask=padding_mask)
        # Pooling: take the mean over the sequence dimension
        x = x.mean(dim=1)
        x = self.fc(x)
        return x