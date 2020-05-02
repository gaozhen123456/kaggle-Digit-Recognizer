import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch

class CustomDatasetFromCSV(Dataset):
    def __init__(self,csv_path, height, width, transforms=None):
        """

        :param csv_path:
        :param height:
        :param width:
        :param transforms:
        :param train: True is train data
        """
        self.data = pd.read_csv(csv_path)
        # print("self.data:\n", self.data)
        self.labels = np.array(self.data["label"])
        # print("self.labels:", self.labels)self.labels: [1 0 1 ... 7 6 9]
        #print("self.labels.shape:", self.labels.shape)#self.labels.shape: (42000,)
        self.height = height
        self.width = width
        self.transforms = transforms
    def __getitem__(self, index):
        single_image_label = self.labels[index]

        img_as_np = np.array(self.data.loc[index][1:]).reshape(self.height,self.width).astype("uint8")
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_np)

        return (img_as_tensor, single_image_label)
    def __len__(self):
        return len(self.data.index)
class Test_CustomDatasetFromCSV(Dataset):
    def __init__(self,csv_path, height, width, transforms=None):
        """

        :param csv_path:
        :param height:
        :param width:
        :param transforms:
        :param train: True is train data
        """
        self.data = pd.read_csv(csv_path)
        self.height = height
        self.width = width
        self.transforms = transforms
    def __getitem__(self, index):
        img_as_np = np.array(self.data.loc[index][0:]).reshape(self.height,self.width).astype("uint8")
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_np)

        return img_as_tensor
    def __len__(self):
        return len(self.data.index)
def get_data(path):
    """

    :param path: train
    :return: data_train, data_valid
    """
    # labels
    df = pd.DataFrame(pd.read_csv(path))
    labels = df["label"]
    labels_digit = []
    for temp in labels:
        labels_digit.append(temp)
    labels_digit_numpy = np.array(labels_digit)
    # data_train
    temp_df = pd.DataFrame(pd.read_csv(path))
    images = []
    image = []
    count = 0
    for temp in temp_df.index:
        print("count:", count)
        line = temp_df.loc[temp].values[0:-1]
        step = 28
        image += [line[i:i+step] for i in range(0,784,step)]
        # print("np.array(image).shape:", np.array(image).shape)
        # np.savetxt("./temp_data.txt", image, fmt="%3d", newline="\n\n")
        image = np.array(image)
        images.append(image)
        count += 1
        # if count == 200:
        #     break
    x_train = np.array(images)
    x_train = x_train / 255.0
    print("x_train.shape:", x_train.shape)
    X_train, X_test, y_train, y_test = train_test_split(x_train,labels_digit_numpy,test_size=0.2, random_state=42)

    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", y_train.shape)
    print("X_test.shape:", X_test.shape)
    print("y_test.shape:", y_test.shape)
    """
    x_train.shape: (42000, 28, 28)
    
    X_train.shape: (33600, 28, 28)
    y_train.shape: (33600,)
    X_test.shape: (8400, 28, 28)
    y_test.shape: (8400,)
    """
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # path = "./data/train/train.csv"
    # get_data(path)
    transformations = transforms.Compose([transforms.ToTensor()])
    train_custom_mnist_from_csv = CustomDatasetFromCSV("./data/train/train.csv", 28, 28, transformations,train=True)
    valid_custom_mnist_from_csv = CustomDatasetFromCSV("./data/train/train.csv", 28, 28, transformations, train=False)

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_custom_mnist_from_csv,
                                                    batch_size=10,
                                                    shuffle=False)
    valid_dataset_loader = torch.utils.data.DataLoader(dataset=valid_custom_mnist_from_csv,
                                                       batch_size=10,
                                                       shuffle=False)

    print("len(train_dataset_loader):", len(train_dataset_loader))
    print("len(test_dataset_loader):", len(valid_dataset_loader))