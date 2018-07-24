import torch.nn as nn


class FCNet(nn.Module):
    """
    This class defined a single layer neural network
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        This initialize function defines network struct
        :param input_size: input x length, exp: for (28x28) image, length can be 28*28=784
        :param hidden_size: it defines how much neural cells net will use
        :param output_size: classes num, in MNIST, it's 10
        """
        self.input_size = input_size
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        # reshape x: from (batch_size, 28, 28, 1) to (batch_size, 784)
        x = x.view(-1, self.input_size)

        # first cal lineal layer
        out = self.fc1(x)

        # cal activation function
        out = self.relu(out)

        # use another linear layer to get output, shape: (batch_size, output_size)
        out = self.fc2(out)
        return out

