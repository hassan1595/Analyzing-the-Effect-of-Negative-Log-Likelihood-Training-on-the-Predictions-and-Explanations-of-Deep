
import torch
from torch import nn

class SimpleNet(nn.Module):

    def __init__(self, input_size,  n_hidden, n_output = 1, dropout = 0.):

        super().__init__()
        self.l_1 = nn.Linear(input_size, n_hidden)
        self.l_2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x,):
        x = self.relu(self.l_1(x))
        x = self.dropout(x)
        x = self.l_2(x)

        return x
    

class SimpleDensityNet(nn.Module):

    def __init__(self, input_size,  n_hidden, n_output = 1, dropout = 0.):

        super().__init__()
        self.l_1 = nn.Linear(input_size, n_hidden)
        self.l_2 = nn.Linear(n_hidden, n_output)
        self.l_3 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        x = self.relu(self.l_1(x))
        x = self.dropout(x)
        mu = self.l_2(x)
        var = self.l_3(x).abs()
        return mu, var

