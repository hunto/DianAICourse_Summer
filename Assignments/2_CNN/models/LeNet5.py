import torch.nn as nn
from torch.nn import functional as F


class LeNet5(nn.Module):
    def __init__(self, image_height, num_classes):
        super(LeNet5, self).__init__()

        # conv1 25x25
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # fc input = (size / 2 - 4) / 2
        fc1_dim = image_height / 4 - 2
        self.fc1 = nn.Linear(16 * fc1_dim * fc1_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(self.dropout(x))
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.shape[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
