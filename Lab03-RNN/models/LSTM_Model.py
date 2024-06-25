"""
LSTM (Long Short-Term Memory) Model
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM Model
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        Constructor

        :param input_size: feature dimension of input
        :param hidden_size: number of features in the hidden state
        :param num_layers: number of recurrent layers
        :param num_classes: number of output classes
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h0=None, c0=None, lengths=None):
        """
        Forward

        :param x: input (L x N x D) where L is the sequence length, N is the batch size, D is the feature dimension
        :param h0: initial hidden state (num_layers x N x hidden_size) (default: None -> zero initialization)
        :param c0: initial cell state (num_layers x N x hidden_size) (default: None -> zero initialization)
        :param lengths: lengths of sequences (N) (default: None for sequences of the same length)
        :return: output (N x num_classes), (hidden state, cell state)
        """
        batch_size = x.size(1)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=True)

        out, (h, c) = self.lstm(x, (h0, c0))

        idx = [-1] * batch_size
        if lengths is not None:
            out, idx = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
            idx = [i - 1 for i in idx]

        last_sequence_list = []
        for i in range(batch_size):
            last_sequence_list.append(out[idx[i], i, :])
        out = torch.stack(last_sequence_list)

        out = self.fc(out)
        return out, (h, c)
