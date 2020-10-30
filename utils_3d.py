import torch
import torch.nn as nn

from torch.nn import Parameter


def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :, :, :].contiguous()
    x2 = x[:, n:, :, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)



class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, input):
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / 4)
        s_width = int(d_width * 2)
        s_height = int(d_height * 2)
        t_1 = output.contiguous().view(batch_size, temp, d_height, d_width, 4, s_depth)
        spl = t_1.split(2, 4)
        stack = [t_t.contiguous().view(batch_size, temp, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).transpose(1, 2).permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, temp, s_height, s_width, s_depth)
        output = output.permute(0, 4, 1, 2, 3)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size,temp, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 2)
        output = output.permute(0, 4, 1, 3, 2)
        return output.contiguous()



class wavelet(nn.Module):
    def __init__(self, block_size):
        super(wavelet, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size



    def inverse(self, input):
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / 4)
        s_width = int(d_width * 2)
        s_height = int(d_height * 2)
        t_1 = output.contiguous().view(batch_size, temp, d_height, d_width, 4, s_depth)
        spl = t_1.split(2, 4)
        stack = [t_t.contiguous().view(batch_size, temp, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).transpose(1, 2).permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, temp, s_height, s_width, s_depth)
        output = output.permute(0, 4, 1, 2, 3)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size,temp, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 2)
        output = output.permute(0, 4, 1, 3, 2)
        return output.contiguous()


def circular_pad_2d(self, x, pad=(1, 1)):
    # Snipped by @zou3519 (https://github.com/zou3519)
    return x.repeat(*x_shape[:2])[
           (x.shape[0] - pad[0]):(2 * x.shape[0] + pad[0]),
           (x.shape[1] - pad[1]):(2 * x.shape[1] + pad[1])
           ]


