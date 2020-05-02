
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
#train_custom_mnist_from_csv = CustomDatasetFromCSV("./data/train/train.csv", 28, 28, transformations)

test_custom_mnist_from_csv = Test_CustomDatasetFromCSV("./data/test/test.csv", 28, 28, transformations)


test_dataset_loader = torch.utils.data.DataLoader(dataset=test_custom_mnist_from_csv,
                                                   batch_size=batch_size,
                                                   shuffle=False)

net2 = torch.load("./data/model.pt")

_model = net2







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
    Test_infer()