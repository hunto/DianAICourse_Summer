import torch
import torch.nn as nn
from torch.nn import functional as F
from .LeNet5 import LeNet5


class LeNetRGB(LeNet5):
    def __init__(self, image_height, num_classes):
        super(LeNetRGB, self).__init__(image_height, num_classes)
        self.fc4 = nn.Linear(in_features=10*3, out_features=10)

    def forward(self, x):
        x1 = self.forward2(x[:, 0:1])
        x2 = self.forward2(x[:, 1:2])
        x3 = self.forward2(x[:, 2:3])
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc4(x)
        return x

    def forward2(self, x):
        return super(LeNetRGB, self).forward(x)

    @staticmethod
    def num_flat_features(x):
        size = x.shape[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
