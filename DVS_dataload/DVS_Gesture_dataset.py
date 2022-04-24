import h5py
import os
from torch.utils.data import Dataset
from DVS_dataload.my_transforms import *
from PIL import Image
import torch
import numpy as np


class DVSGestureDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(DVSGestureDataset, self).__init__()
        self.n = 0
        self.root = root
        self.train = train
        self.transform = transform

        if train:
            root_train = os.path.join(self.root, 'DvsGesture_train_40step_downsample')
            for _, _, self.files_train in os.walk(root_train):
                pass
            self.n = len(self.files_train)
        else:
            root_test = os.path.join(self.root, 'DvsGesture_test_40step_downsample')
            for _, _, self.files_test in os.walk(root_test):
                pass
            self.n = len(self.files_test)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.train:
            root_test = os.path.join(self.root, 'DvsGesture_train_40step_downsample')

            with h5py.File(root_test + os.sep + self.files_train[idx], 'r', swmr=True, libver="latest") as f:
                target = f['label'][()]
                data = f['data'][()]
            if self.transform is not None:
                data = self.transform(data)

            return data, target
        else:
            root_test = os.path.join(self.root, 'DvsGesture_test_40step_downsample')

            with h5py.File(root_test + os.sep + self.files_test[idx], 'r', swmr=True, libver="latest") as f:
                target = f['label'][()]
                data = f['data'][()]
            if self.transform is not None:
                data = self.transform(data)

            return data, target
