import torch

class NearestConv(torch.nn.ConvTranspose2d):
    def __init__(self, conv3x3):
        assert conv3x3.kernel_size == (3, 3)
        assert conv3x3.stride == (1, 1)
        assert conv3x3.padding == (1, 1)
        super(NNConv, self).__init__(conv3x3.in_channels,
                                     conv3x3.out_channels,
                                     kernel_size=4,
                                     stride=2,
                                     padding=1,
                                     bias=conv3x3.bias is not None)
        self.weight.data.zero_()
        self.weight.data[..., 1:,1:] += conv3x3.weight.data
        self.weight.data[..., 1:,:-1] += conv3x3.weight.data
        self.weight.data[..., :-1,1:] += conv3x3.weight.data
        self.weight.data[..., :-1,:-1] += conv3x3.weight.data
        self.weight.data = torch.flip(self.weight.data, (2,3))
        if self.bias is not None:
            self.bias.data.copy_(conv3x3.bias)
