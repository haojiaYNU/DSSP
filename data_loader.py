# data_loader.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class TabularDataset(Dataset):

    def __init__(self, x, y):

        self.x = torch.tensor(x.astype(np.float32)).to(torch.float)
        self.y = torch.tensor(y.astype(np.float32)).to(torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class MaskDataset(Dataset):
    def __init__(self, x, mask):
        self.x = x
        self.mask = mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.mask[index]


def read_csv(path, label, columns, header=None):
    df = pd.read_csv(path, header=header, names=columns)
    x = df.drop(label, axis=1)
    y = pd.get_dummies(df[label])
    return x.to_numpy(), y.to_numpy()


def mnist_to_tabular(x, y):
    x = x / 255.0
    y = np.asarray(pd.get_dummies(y))

    # flatten
    no, dim_x, dim_y = np.shape(x)
    x = np.reshape(x, [no, dim_x * dim_y])
    return x, y


def get_dataset(data_name, label_data_rate):
    if data_name == 'OULAD':
        data = pd.read_csv("E:/科研/成果维护/SCI/semi-supervised and self-training method for early success prediction/2nd-test/AAA/1-0.1.csv")
        train, test = train_test_split(data.fillna(0), test_size=0.2, random_state=42)
        x_train = np.asarray(pd.get_dummies(train.drop(['final_result'], axis=1)))
        y_train = np.asarray(pd.get_dummies(train['final_result']))

        x_test = np.asarray(pd.get_dummies(test.drop(['final_result'], axis=1)))
        y_test = np.asarray(pd.get_dummies(test['final_result']))

    # Divide labeled and unlabeled data
    idx = np.random.permutation(len(y_train))
    # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    # Unlabeled data
    x_unlab = x_train[unlab_idx, :]
    y_unlab = y_train[unlab_idx, :]

    # Labeled data
    x_label = x_train[label_idx, :]
    y_label = y_train[label_idx, :]

    return TabularDataset(x_label, y_label), \
           TabularDataset(x_unlab, y_unlab), \
           TabularDataset(x_test, y_test)


if __name__ == '__main__':
    a, _, _ = get_dataset('OULAD', 0.1)
    print(a[[0, 1]])
