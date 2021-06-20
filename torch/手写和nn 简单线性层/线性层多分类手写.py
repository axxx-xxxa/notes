import torch
import numpy
import torch.nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
batchsz=200
learning_rate=0.01
epochs=10

train_loader = torch.utils.data.dataloader(datasets.MNIST('../data',train=True,download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0,1307),(0,3081))
                                        ])),batch_size=batchsz,shuffle=True)
train_loader = torch.utils.data.dataloader(datasets.MNIST('../data',train=False,download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0,1307),(0,3081))
                                        ])),batch_size=batchsz,shuffle=True)

w1,b1 = torch.randn(200,784,requires_grad=True),\
        torch.zeros(200,requires_grad=True)
w1,b1 = torch.randn(200,200,requires_grad=True),\
        torch.zeros(200,requires_grad=True)
w1,b1 = torch.randn(10,200,requires_grad=True),\
        torch.zeros(10,requires_grad=True)

def forward(x):
    x = x@w1.t()+b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return x

optimizer = torch.optim.SGD([w1,b1,w2,b2,w3,b3],lr=learning_rate)
criteon = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)
        logits = forward(data)
        loss = criteon(logits,target)
        loss.backward()
        optimizer.step()

        if batch_idx%100 ==0:
            print("^^^^^^^")

    test_loss = 0
    correct = 0
    ## ……………………………………求loss和correct