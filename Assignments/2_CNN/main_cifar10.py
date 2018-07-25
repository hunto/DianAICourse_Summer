import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.ResNet import resnet20
from models.vgg import vgg11
from torch import optim as optim


def train(model, data, target, loss_func, optimizer):
    """
    train step, input a batch data, return accuracy and loss
    :param model: network model object
    :param data: input data, shape: (batch_size, 28, 28, 1)
    :param target: input labels, shape: (batch_size, 1)
    :param loss_func: the loss function you use
    :param optimizer: the optimizer you use
    :return: accuracy, loss
    """
    model.train()
    # initial optimizer
    optimizer.zero_grad()

    # net work will do forward computation defined in net's [forward] function
    output = model(data)

    # get predictions from outputs, the highest score's index in a vector is predict class
    predictions = output.max(1, keepdim=True)[1]

    # cal correct predictions num
    correct = predictions.eq(target.view_as(predictions)).sum().item()

    # cal accuracy
    acc = correct / len(target)

    # use loss func to cal loss
    loss = loss_func(output, target)

    # backward will back propagate loss
    loss.backward()

    # this will update all weights use the loss we just back propagate
    optimizer.step()

    return acc, loss


def test(model, test_loader, loss_func, use_cuda):
    """
    use a test set to test model
    NOTE: test step will not change network weights, so we don't use backward and optimizer
    :param model: net object
    :param test_loader: type: torch.utils.data.Dataloader
    :param loss_func: loss function
    :return: accuracy, loss
    """
    model.eval()
    acc_all = 0
    loss_all = 0
    step = 0
    with torch.no_grad():
        for data, target in test_loader:
            step += 1
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            predictions = output.max(1, keepdim=True)[1]
            correct = predictions.eq(target.view_as(predictions)).sum().item()
            acc = correct / len(target)
            loss = loss_func(output, target)
            acc_all += acc
            loss_all += loss
    return acc_all / step, loss_all / step


def main():
    """
    main function
    """

    # define some hyper parameters
    num_classes = 10
    eval_step = 1000
    num_epochs = 100
    batch_size = 64

    # first check directories, if not exist, create
    dir_list = ('../data', '../data/MNIST', '../data/CIFAR-10')
    for directory in dir_list:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # if cuda available -> use cuda
    use_cuda = torch.cuda.is_available()
    # this step will create train_loader and  test_loader

    train_loader = DataLoader(
        datasets.CIFAR10(root='../data/CIFAR-10', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='../data/CIFAR-10', train=False,
                         transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size
    )

    model = resnet20()
    model = vgg11()

    # define network

    if use_cuda:
        model = model.cuda()

    # define loss function
    ce_loss = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # start train
    train_step = 0
    for _ in range(num_epochs):
        for data, target in train_loader:
            train_step += 1
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            acc, loss = train(model, data, target, ce_loss, optimizer)
            if train_step % 100 == 0:
                print('Train set: Step: {}, Loss: {:.4f}, Accuracy: {:.2f}'.format(train_step, loss, acc))
            if train_step % eval_step == 0:
                acc, loss = test(model, test_loader, ce_loss, use_cuda)
                print('\nTest set: Step: {}, Loss: {:.4f}, Accuracy: {:.2f}\n'.format(train_step, loss, acc))


if __name__ == '__main__':
    main()
