import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class EncoderL(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(EncoderL, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True
        )
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((batch_size, self.embedding_dim)) 

class DecoderL(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(DecoderL, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
        input_size=input_dim,
        hidden_size=input_dim,
        num_layers=1,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=input_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
        
    def forward(self, x):       
        batch_size = x.shape[0]
        x = x.repeat(1, self.seq_len, 1)        
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
        return x

class INDRA_IDS_L(nn.Module):

        def __init__(self, seq_len, n_features, device, embedding_dim=64):
            super(INDRA_IDS_L, self).__init__()         
            self.fc1 = nn.Linear(n_features, 128).to(device)
            self.encoder = EncoderL(seq_len, 128, embedding_dim).to(device)
            self.decoder = DecoderL(seq_len, embedding_dim, n_features).to(device)
            self.fc2 = nn.Linear(128, n_features)
            self.tanh = nn.Tanh()
        def forward(self, x):
            x = self.fc1(x)
            x = self.tanh(self.encoder(x))
            x = self.tanh(self.decoder(x))
            x = self.fc2(x)
            return x
