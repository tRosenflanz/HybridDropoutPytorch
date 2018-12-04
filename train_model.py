
import torch
import torch.utils.data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hybrid_drop import *
import numpy as np
import pickle 
import sys
 
n_epochs = 100
batch_size_train = 128
batch_size_test = 1000
log_interval = 100
drop_rate = .5
random_seed = np.random.randint(low=1,high=1000000000)
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Basic_Block(nn.Module):
    def __init__(self,inp,drop):
        super(Basic_Block, self).__init__()
        self.conv1 = nn.Conv2d(inp, 32, kernel_size=5,padding=2)
        self.regularization = reg_func(drop)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5,padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.regularization(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = self.regularization(x)
        return x
    
class Net(nn.Module):
    def __init__(self,drop):
        super(Net, self).__init__()
        self.regularization = reg_func(drop/2)
        self.block1 = Basic_Block(1,drop)
        self.block2 = Basic_Block(32,drop)
        self.fc1 = nn.Linear(1568, 10)

    def forward(self, x):
        x = self.regularization(x)
        x = self.block1(x)
        x = self.block2(x)
        x = Flatten()(x)
        x = self.fc1(x)
        return F.log_softmax(x,-1)


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target  = target.cuda()
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_correct.append(correct)
    network.train()




def train(epoch):
    network.train()
    correct =0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target  = target.cuda()
        optimizer.zero_grad()
        output = network(data)
        pred = output.data.max(1, keepdim=True)[1]     
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx%(2*log_interval)==0:
            test()
            test_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))

        correct+=pred.eq(target.data.view_as(pred)).sum().item()
        train_losses.append(loss.item())
        train_correct.append(correct)
        train_counter.append(
        (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
        
    train_correct.append(correct)

    
    
def main():
    type_name = sys.argv[1]
    assert type_name in ['normal','hybrid_normal','hybrid_spatial']
    
    global reg_func 
    if type_name=='normal':
        reg_func = nn.Dropout
    elif type_name=='hybrid_normal':
        reg_func = HybridDropout_Normal
    elif type_name=='hybrid_spatial':
        reg_func = HybridDropout_Spatial
        

    global train_loader,test_loader
    train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.MNIST('/files/', train=True,download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                            batch_size=batch_size_train, shuffle=True,pin_memory=True,num_workers=0)

    test_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.MNIST('/files/', train=False, download=True,
                                                         transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(
                                                             (0.1307,), (0.3081,))
                                                         ])),
                            batch_size=batch_size_test, shuffle=False,pin_memory=True,num_workers=0)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    global network,optimizer
    network = Net(drop_rate)
    network = network.cuda()
    optimizer = optim.Adam(network.parameters())
    
    global train_losses,train_correct,train_counter,\
           test_losses,test_counter,test_correct
    train_losses = []
    train_correct = []
    train_counter = []
    test_losses = []
    test_counter = []
    test_correct = []

    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
        test_counter.append(epoch*len(train_loader.dataset))



    with open('/data/{}_{}_{}'.format(type_name,random_seed,drop_rate),'wb') as f:
        pickle.dump([train_losses,train_correct,train_counter,test_losses,test_counter,test_correct],f)




if __name__ == "__main__":
    main()


