# DianAI培训1 - 作业讲解
### 贝贝组 黄涛
#### 2018.07.26

---

# Learning Pytorch
* [Official Site](https://pytorch.org/)
* [Pytorch Tutorials](http://pytorch.org/tutorials/)
* [Pytorch docs](https://pytorch.org/docs/stable/index.html)
* [Pytorch Examples](https://github.com/pytorch/examples)

---

# Lectures & Solutions
[https://github.com/hunto/DianAICourse_Summer](https://github.com/hunto/DianAICourse_Summer)

---

## 1. Define your network
A network is a subclass of `torch.nn.Model`

``` python
class FCNet(torch.nn.Module):
    # <-- class func. below -->
```

### 1.1 Complete Init Function
```Python
def __init__(self, input_size, hidden_size, output_size):
    super(FCNet, self).__init__()  # initialize father class
    self.input_size = input_size
    # then define your network layers, exp:
    self.fc1 = torch.nn.Linear(in_features=input_size, 
                               out_features=hidden_size)
    self.relu = torch.nn.ReLU()
    self.fc2 = nn.Linear(in_features=hidden_size, 
                         out_features=output_size)
```

---

### 1.2 Complete foward function
```Python
def forward(self, x):
    x = x.view(-1, self.input_size)
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out
```

#### NOTE: You don't need to define backward function, Pytorch can automatically cal gradients and update weights.

---

## 2. Process data
`torch.utils.data.Dataloader` 

```
train_loader = DataLoader(
    datasets.MNIST(root='../data/MNIST', train=True, 
       download=True,
       transform=transforms.Compose([transforms.ToTensor()])),
       batch_size=batch_size,
       shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST(root='../data/MNIST', train=False,
       transform=transforms.Compose([transforms.ToTensor()])),
       batch_size=batch_size
)
```

---

## 3. Loss function & Optimizer
```
# define network
model = FCNet(input_vec_length, cell_num, num_classes)

if use_cuda:
    model = model.cuda()

# define loss function
ce_loss = torch.nn.CrossEntropyLoss()

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-3)
```

---

## 4. Train & Evaluation
```
# start train
train_step = 0
for eopch in range(1, 101):
    for data, target in train_loader:
    train_step += 1
    if use_cuda:
        data = data.cuda()
    acc, loss = train(model, data, target, 
                      ce_loss, optimizer)
    if train_step % 100 == 0:
        print('Train set: Step: {}, Loss: {:.4f}, Accuracy: {:.2f}'.format(train_step, loss, acc))
    if train_step % 1000 == 0:
        acc, loss = test(model, test_loader, ce_loss)
        print('\nTest set: Step: {}, Loss: {:.4f}, Accuracy: {:.2f}\n'.format(train_step, loss, acc))
```

---

## 5. Train function
```python
def train(model, data, target, loss_func, optimizer):
    # initial optimizer
    optimizer.zero_grad()
    # net work will do forward computation defined in net's [forward] function
    output = model(data)
    # get predictions from outputs, the highest score's index in a vector is predict class
    predictions = output.max(1, keepdim=True)[1]
    # cal correct predictions num
    correct = predictions.eq(target.view_as(predictions))
                             .sum().item()
    # cal accuracy
    acc = correct / len(target)
    # use loss func to cal loss
    loss = loss_func(output, target)
    # backward will back propagate loss
    loss.backward()
    # this will update all weights use the loss we just back propagate
    optimizer.step()
    return acc, loss
```

---

# Run

![](/Users/hunto/Downloads/WX20180724-164050@2x.png)

