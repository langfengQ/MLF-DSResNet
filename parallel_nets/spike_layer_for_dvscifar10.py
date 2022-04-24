import torch
import torch.nn as nn
import torch.nn.functional as F
import math


Vth = 0.6
Vth2 = 1.6
Vth3 = 2.6
a = 1.0
TimeStep = 10
tau = 0.25
momentum_SGD = 0.9


class MLF_unit(nn.Module):
    """ MLF unit (K=3).
    MLF (K=2) can be got by commenting out the lines related to u3 and replace
    o[t*bs:(t+1)*bs, ...] = spikefunc(u) + spikefunc2(u2) + spikefunc3(u3) with
    o[t*bs:(t+1)*bs, ...] = spikefunc(u) + spikefunc2(u2).
    """
    def __init__(self):
        super(MLF_unit, self).__init__()

    def forward(self, x):
        bs = int(x.shape[0] / TimeStep)
        u = torch.zeros((bs,) + x.shape[1:], device=x.device)
        u2 = torch.zeros((bs,) + x.shape[1:], device=x.device)
        u3 = torch.zeros((bs,) + x.shape[1:], device=x.device) # comment this line if you want MLF (K=2)
        o = torch.zeros(x.shape, device=x.device)
        for t in range(TimeStep):
            u = tau * u * (1 - spikefunc(u)) + x[t * bs:(t + 1) * bs, ...]
            u2 = tau * u2 * (1 - spikefunc2(u2)) + x[t * bs:(t + 1) * bs, ...]
            u3 = tau * u3 * (1 - spikefunc3(u3)) + x[t * bs:(t + 1) * bs, ...] # comment this line if you want MLF (K=2)
            o[t*bs:(t+1)*bs, ...] = spikefunc(u) + spikefunc2(u2) + spikefunc3(u3) # Equivalent to union of all spikes
        return o


class tdBatchNorm(nn.Module):
    def __init__(self, bn, alpha=1):
        super(tdBatchNorm, self).__init__()
        self.bn = bn
        self.alpha = alpha

    def forward(self, x):
        exponential_average_factor = 0.0

        if self.training and self.bn.track_running_stats:
            if self.bn.num_batches_tracked is not None:
                self.bn.num_batches_tracked += 1
                if self.bn.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.bn.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.bn.momentum

        if self.training:
            mean = x.mean([0, 2, 3], keepdim=True)
            var = x.var([0, 2, 3], keepdim=True, unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.bn.running_mean = exponential_average_factor * mean[0, :, 0, 0]\
                                       + (1 - exponential_average_factor) * self.bn.running_mean
                self.bn.running_var = exponential_average_factor * var[0, :, 0, 0] * n / (n - 1) \
                                      + (1 - exponential_average_factor) * self.bn.running_var
        else:
            mean = self.bn.running_mean[None, :, None, None]
            var = self.bn.running_var[None, :, None, None]

        x = self.alpha * Vth * (x - mean) / (torch.sqrt(var) + self.bn.eps)

        if self.bn.affine:
            x = x * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]

        return x


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = torch.gt(input, Vth)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input - Vth) < (a/2)) / a
        return grad_input * hu


spikefunc = SpikeFunction.apply


class SpikeFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, Vth2)
        return output.float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input - Vth2) < (a/2)) / a
        return grad_input * hu

spikefunc2 = SpikeFunction2.apply

class SpikeFunction3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, Vth3)
        return output.float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (abs(input - Vth3) < (a/2)) / a
        return grad_input * hu

spikefunc3 = SpikeFunction3.apply

