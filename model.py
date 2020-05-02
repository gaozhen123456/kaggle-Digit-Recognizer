import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入通道，输出10通道，kernel 5 * 5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(40, 10)# (in_features, out_features)
    def forward(self, x):
        # in_size = 64
        in_size = x.size(0)# one batch 此时的x是包含batchsize维度为4的Tensor,即(batchsize,channels,x,y)
                           # x.size(0)值batchsize的值，把batchsize的值作为网络的in_size.
        # x:64*1*28*28
        x = F.relu(F.dropout(self.mp(self.conv1(x),0.25)))
        # x:64*10*12*12 feature map = [(28-4)/2]^2=12*12
        x = F.relu(F.dropout(self.mp(self.conv2(x))))
        # x:64*20*4*4
        x = F.relu(F.dropout(self.mp(self.conv3(x))))

        x = x.view(in_size, -1)# flatten the tensor 相当于reshape
        # x:64*20
        x = self.fc(x)
        return F.log_softmax(x) # 64 * 10

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()


        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        #self.batchnormaliztion1 = nn.BatchNorm2d()
        # Max pool1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Convolution2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Max pool2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.25)
        # Fully connected1
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # out = nn.BatchNorm2d(out)
        # Max pool1
        out = self.maxpool1(out)
        out = self.dropout1(out)
        # Convoltion2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool2
        out = self.maxpool2(out)
        out = self.dropout2(out)
        # flatten
        out = out.view(out.size(0), -1)

        # Linear function(readout)
        out = self.fc1(out)
        return out



