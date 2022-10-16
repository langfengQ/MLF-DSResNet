from parallel_nets.spike_layer_for_Gesture import *
from math import sqrt
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, modified=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = tdBatchNorm(nn.BatchNorm2d(planes))
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = tdBatchNorm(nn.BatchNorm2d(planes))
        self.spike_func = MLF_unit()
        self.shortcut = nn.Sequential()
        self.modified = modified

        if stride != 1 or in_planes != planes:
            if self.modified:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    tdBatchNorm(nn.BatchNorm2d(planes)),
                    MLF_unit()
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    tdBatchNorm(nn.BatchNorm2d(planes)),
                )

    def forward(self, x):
        out = self.spike_func(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        if self.modified:
            out = self.spike_func(out)
            out += self.shortcut(x)         # Equivalent to union of all spikes
        else:
            out += self.shortcut(x)
            out = self.spike_func(out)
        return out


class BLock_Layer(nn.Module):
    def __init__(self, block, in_planes, planes, num_block, downsample, modified):
        super(BLock_Layer, self).__init__()
        layers = []
        if downsample:
            layers.append(block(in_planes, planes, 2, modified))
        else:
            layers.append(block(in_planes, planes, 1, modified))
        for _ in range(1, num_block):
            layers.append(block(planes, planes, 1, modified))
        self.execute = nn.Sequential(*layers)

    def forward(self, x):
        return self.execute(x)


class ResNet(nn.Module):
    """ Establish ResNet.
     Spiking DS-ResNet with “modified=True.”
     Spiking ResNet with “modified=False.”
     """
    def __init__(self, block, num_block_layers, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, 1, 1, bias=False)
        self.bn0 = tdBatchNorm(nn.BatchNorm2d(2, affine=False), 1)
        self.bn1 = tdBatchNorm(nn.BatchNorm2d(16), 1)
        self.layer1 = BLock_Layer(block, 16, 16, num_block_layers[0], False, modified=True)
        self.layer2 = BLock_Layer(block, 16, 32, num_block_layers[1], True, modified=True)
        self.layer3 = BLock_Layer(block, 32, 64, num_block_layers[2], True, modified=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.voting = nn.Linear(64, num_classes)
        self.spike_func = MLF_unit()

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.bn0(x)
        out = self.spike_func(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        
        out = self.voting(out)
        bs = int(out.shape[0] / TimeStep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(TimeStep):
            o += out[t*bs:(t+1)*bs, ...]
        o /= TimeStep
        return o


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3], 11)
