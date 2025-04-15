import torch
import torch.nn as nn

class ConvLSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, kernel_size=3, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size=64*64, hidden_size=512, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(512, 64*64)

    def forward(self, x):
        B, T, C, H, W = x.shape  # x: (B, T, 1, 64, 64)
        x = x.view(B, T, -1)     # (B, T, 4096)
        lstm_out, _ = self.encoder(x)
        pred = self.decoder(lstm_out[:, -1])  # take output of last timestep
        return pred.view(B, 1, 64, 64)        # (B, 1, 64, 64)
