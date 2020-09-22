import math

import torch
import torch.nn.functional as F
from torch import nn


class IntentionFFNN(nn.Module):
    def __init__(
        self,
        input_size=51*15,
        hidden_size=64,
        output_size=2,
        num_output=1,
        num_layers=2
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_size] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_size]+h[:-1], h))
        self.fc = nn.ModuleList(nn.Linear(hidden_size, output_size) for _ in range(num_output))
        for layer in self.layers: nn.init.xavier_uniform_(layer.weight)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
        out = [linear(x) for linear in self.fc]
        return out


class IntentionRNN(nn.Module):
    def __init__(
        self,
        input_size=51,
        hidden_size=128,
        output_size=2,
        num_output=1,
        num_layers=2,
        rnn_type='gru'
    ):
        super().__init__()
        self.recurrent_module = None
        if rnn_type.lower() == 'lstm':
            self.recurrent_module = nn.LSTM(
                input_size,   # input_size
                hidden_size,  # hidden_size
                num_layers,
                batch_first=True,
                bidirectional=False)
        elif rnn_type.lower() == 'gru':
            self.recurrent_module = nn.GRU(
                input_size,   # input_size
                hidden_size,  # hidden_size
                num_layers,
                batch_first=True,
                bidirectional=False)
        self.fc = nn.ModuleList(nn.Linear(hidden_size, output_size) for _ in range(num_output))

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        x, _ = self.recurrent_module(x)
        x = F.relu(x[:, -1])
        out = [linear(x) for linear in self.fc]
        return out


class IntentionTransformer(nn.Module):
    def __init__(
        self,
        input_size=52,
        hidden_size=128,
        output_size=2,
        num_output=1,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.ModuleList(nn.Linear(input_size, output_size) for _ in range(num_output))

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        out = [linear(x[:, -1]) for linear in self.fc]
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    input_size       = 52
    sample_x         = torch.rand(2, 15, input_size)

    test_ffpn        = IntentionFFNN(input_size=input_size, num_output=2)
    test_rnn         = IntentionRNN(input_size=input_size, num_output=2)
    test_transformer = IntentionTransformer(input_size=input_size, num_head=4, num_output=2)
    a = test_transformer(sample_x)
    print(a)
