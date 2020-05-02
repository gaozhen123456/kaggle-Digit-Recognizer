
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
from data_loader import CustomDatasetFromCSV,Test_CustomDatasetFromCSV
from model import Net, CNNModel
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# Training settings
batch_size = 10
# Data Loader (Input Pipeline)
transformations = transforms.Compose([transforms.ToTensor()])
train_custom_mnist_from_csv = CustomDatasetFromCSV("./data/train/data_train.csv", 28, 28, transformations)
#train_custom_mnist_from_csv = CustomDatasetFromCSV("./data/train/train.csv", 28, 28, transformations)

valid_custom_mnist_from_csv = CustomDatasetFromCSV("./data/train/data_valid.csv", 28, 28, transformations)
test_custom_mnist_from_csv = Test_CustomDatasetFromCSV("./data/test/test.csv", 28, 28, transformations)

train_dataset_loader = torch.utils.data.DataLoader(dataset=train_custom_mnist_from_csv,
                                                batch_size=batch_size,
                                                shuffle=False)
valid_dataset_loader = torch.utils.data.DataLoader(dataset=valid_custom_mnist_from_csv,
                                                   batch_size=batch_size,
                                                   shuffle=False)
test_dataset_loader = torch.utils.data.DataLoader(dataset=test_custom_mnist_from_csv,
                                                   batch_size=batch_size,
                                                   shuffle=False)

print("len(train_dataset_loader):", len(train_dataset_loader))
print("len(valid_dataset_loader):", len(valid_dataset_loader))
print("len(test_dataset_loader):", len(test_dataset_loader))

_model = CNNModel()
from torchsummary import summary
summary(_model.cuda(), input_size=(1, 28, 28))
optimizer = optim.SGD(_model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_dataset_loader):# batch_idx是enumerate()函数自带的索引，从0开始
        # data.size():[-1, 1, 28, 28]
        # target.size()：[64]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        output = _model(data)

        target = target.to(device)
        # output:64*10(batchsize,10类别)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target)

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataset_loader.dataset),
                       100. * batch_idx / len(train_dataset_loader), loss.item()))

        optimizer.zero_grad()# 所有参赛的梯度清零。
        loss.backward()# 即反向传播求梯度。
        optimizer.step()# 调用optimizer进行梯度下降更新参。
    torch.save(_model,"./data/model.pt")




def _valid():
    valid_loss = 0
    correct = 0

    for data, target in valid_dataset_loader:
        data, target = Variable(data).cuda(), Variable(target).cuda()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        target = target.to(device)

        output = _model(data)
        # print("output:", output)
        # print("output.shape:", output.shape)# (batchsize,10类别)
        # sum up batch loss
        criteria = nn.CrossEntropyLoss()
        # valid_loss += criteria(output, target, size_average=False).item()
        valid_loss += criteria(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]

        # print("---", pred)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()


    valid_loss /= len(valid_dataset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_dataset_loader.dataset),
        100. * correct / len(valid_dataset_loader.dataset)))
    temp = 100. * correct / len(valid_dataset_loader.dataset)
    print("temp:", temp.item())
    return temp.item()

def Test_infer():
    res = []
    for data in test_dataset_loader:
        data = Variable(data).cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        output = _model(data)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]

        temp = pred.cuda().data.cpu().numpy().squeeze()
        for i in range(len(temp)):
            res.append(temp[i])
    res_to_np = np.array(res)
    import pandas as pd
    pd = pd.read_csv("./data/sample_submission.csv")
    print("res_to_np.shape:", res_to_np.shape)
    pd["Label"] = res_to_np
    pd.to_csv("./data/mnist_res.csv", index=False)












if __name__ == "__main__":
    epochs = 2
    for epoch in range(epochs):
        train(epoch)
        print("训练结束！")
        max_score = 0
        score = _valid()
        if score > max_score:
            Test_infer()







