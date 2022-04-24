import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda


class Normalize_ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor. """
    def __init__(self, adjust=1):
        self.adjust = adjust

    def __call__(self, frame):
        return torch.FloatTensor(frame)

    def __repr__(self):
        return self.__class__.__name__ + '()'


