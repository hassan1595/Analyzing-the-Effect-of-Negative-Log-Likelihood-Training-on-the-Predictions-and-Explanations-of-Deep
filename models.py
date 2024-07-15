
import torch
from torch import nn
import torchvision

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


class vgg16SimpleNet(nn.Module):
    def __init__(self, n_output = 1):
        super().__init__()
        self.vgg16 = torchvision.models.vgg16(weights = "IMAGENET1K_V1")
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(self.vgg16.classifier[0].in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, n_output)
        )

    def forward(self, x):
        return self.vgg16(x)
    


class vgg16DensityNet(nn.Module):
    def __init__(self, n_output = 1):
        super().__init__()
        self.vgg16 = torchvision.models.vgg16(weights = "IMAGENET1K_V1")
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(self.vgg16.classifier[0].in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.l_2 = nn.Linear(4096, n_output)
        self.l_3 = nn.Linear(4096, n_output)

    def forward(self, x):
        x = self.vgg16(x)
        mu = self.l_2(x)
        var = self.l_3(x).abs()
        return mu, var

    

