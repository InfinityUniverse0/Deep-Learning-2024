"""
LSTM (Long Short-Term Memory) Model without nn.LSTM
"""

import torch
import torch.nn as nn


class CustomLSTMCell(nn.Module):
    """
    Custom LSTM Cell
    """
    def __init__(self, input_size, hidden_size):
        """
        Constructor

        :param input_size: feature dimension of input
        :param hidden_size: number of features in the hidden state
        """
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h, c):
        """
        Forward pass for a single time step

        :param x: input tensor at current time step (N x D)
        :param h: hidden state tensor from previous time step (N x hidden_size)
        :param c: cell state tensor from previous time step (N x hidden_size)
        :return: (new hidden state, new cell state)
        """
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h))
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h))
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h))
        c_tilda = torch.tanh(self.W_c(x) + self.U_c(h))
        c_new = f_t * c + i_t * c_tilda
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new


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
        self.lstm_cells = nn.ModuleList([CustomLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
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
        lengths = None
        seq_len, batch_size, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=True)

        h = [h0[i] for i in range(self.num_layers)]
        c = [c0[i] for i in range(self.num_layers)]

        out = []
        for t in range(seq_len):
            x_t = x[t]
            for i, lstm_cell in enumerate(self.lstm_cells):
                h[i], c[i] = lstm_cell(x_t, h[i], c[i])
                x_t = h[i]
            out.append(h[-1])

        out = torch.stack(out)

        idx = [-1] * batch_size
        if lengths is not None:
            out, idx = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
            idx = [i - 1 for i in idx]

        last_sequence_list = []
        for i in range(batch_size):
            last_sequence_list.append(out[idx[i], i, :])
        out = torch.stack(last_sequence_list)

        out = self.fc(out)
        return out, (torch.stack(h), torch.stack(c))
