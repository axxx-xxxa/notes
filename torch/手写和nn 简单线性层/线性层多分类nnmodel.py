import torch
import numpy
from torch import  nn
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

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        x=self.model(x)
        return x

device = torch.device('cuda:0')
net = mynet().to(device)
optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data,target = data.to(device),target.cuda()
        logits = mynet(data)
        loss = criteon(logits,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("^^^^^^^")

    test_loss = 0
    correct = 0
    ## ……………………………………求loss和correct
