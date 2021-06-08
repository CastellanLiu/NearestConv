import torch

class NearestConv(torch.nn.ConvTranspose2d):
    def __init__(self, conv):
        
        assert all(i % 2 == 1 for i in conv.kernel_size)
        assert conv.stride == (1, 1)
        assert all(i*2+1==j for i,j in zip(conv.padding, conv.kernel_size))
        
        kernel_size = tuple(i+1 for i in conv.kernel_size)
        
        super(NearestConv, self).__init__(conv.in_channels,
                                          conv.out_channels,
                                          kernel_size=kernel_size,
                                          stride=2,
                                          padding=conv.padding,
                                          bias=conv.bias is not None)
        self.to(conv.weight.data.device)
        
        raw_weight = conv.weight.data.transpose(0, 1).flip((2,3))
        self.weight.data.zero_()
        self.weight.data[..., 1:,1:] += raw_weight
        self.weight.data[..., 1:,:-1] += raw_weight
        self.weight.data[..., :-1,1:] += raw_weight
        self.weight.data[..., :-1,:-1] += raw_weight
        
        if self.bias is not None:
            self.bias.data.copy_(conv.bias)
