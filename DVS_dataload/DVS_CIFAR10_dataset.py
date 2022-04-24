import h5py
import os
from torch.utils.data import Dataset
from DVS_dataload.my_transforms import *
from PIL import Image
import torch
import numpy as np


class DVSCIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.num = 0
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            root_trian = os.path.join(self.root, 'DVS_CIFAR10_train_10ms_10step')
            for _, _, self.files_train in os.walk(root_trian):
                pass
            self.num = len(self.files_train)
        else:
            root_test = os.path.join(self.root, 'DVS_CIFAR10_test_10ms_10step')
            for _, _, self.files_test in os.walk(root_test):
                pass
            self.num = len(self.files_test)

    def __getitem__(self, idx):
        if self.train:
            root_trian = os.path.join(self.root, 'DVS_CIFAR10_train_10ms_10step')

            with h5py.File(os.path.join(root_trian, self.files_train[idx]), 'r', swmr=True, libver="latest") as f:
                target = f['label'][()]
                data = f['data'][()]
                if self.transform is not None:
                    data = self.transform(data)

                return data, target
        else:
            root_trian = os.path.join(self.root, 'DVS_CIFAR10_test_10ms_10step')

            with h5py.File(os.path.join(root_trian, self.files_test[idx]), 'r', swmr=True, libver="latest") as f:
                target = f['label'][()]
                data = f['data'][()]
                if self.transform is not None:
                    data = self.transform(data)

                return data, target

    def __len__(self):
        return self.num
