"""
Data Preprocessing for Name-Language Classification
"""

import torch
from torch.utils.data.dataset import Dataset
import os
import string
import unicodedata


all_letters = string.ascii_letters + " .,;'"  # All letters


def unicode_to_ascii(s):
    # Unicode to ASCII
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    ).encode('ascii', 'ignore').decode('ascii')


class NameDataset(Dataset):
    """
    Name Dataset
    """
    def __init__(self, data_path):
        """
        Constructor

        :param data_path: data path
        """
        self.all_letters = all_letters  # All letters
        self.all_categories = []  # All categories
        self.names = []  # Name list
        self.labels = []  # Label list
        files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        for i, file in enumerate(files):
            category = os.path.splitext(os.path.basename(file))[0]
            self.all_categories.append(category)
            with open(os.path.join(data_path, file), 'r') as f:
                lines = f.read().strip().splitlines()
                lines = [unicode_to_ascii(line) for line in lines]
                self.names.extend(lines)
                self.labels.extend([i] * len(lines))

    def __len__(self):
        """
        Length

        :return: length
        """
        return len(self.names)

    def __getitem__(self, idx):
        """
        Get Item

        :param idx: index
        :return: item (name, label) where name is a tensor of shape (L x D) and label is a tensor of shape (1)
        """
        name = torch.zeros(len(self.names[idx]), len(all_letters))
        for i, letter in enumerate(self.names[idx]):
            name[i][all_letters.find(letter)] = 1
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return name, label


def collate_fn(data):
    """
    Collate Function

    :param data: data
    :return: collated data
    """
    data.sort(key=lambda x: x[0].size(0), reverse=True)
    names, labels = zip(*data)
    lengths = torch.tensor([name.size(0) for name in names])  # Convert to tensor
    names = torch.nn.utils.rnn.pad_sequence(names, batch_first=False, padding_value=0)
    labels = torch.stack(labels)

    return names, labels, lengths
