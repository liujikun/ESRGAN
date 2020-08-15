import functools
from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
#                 ESRGAN
# ----------------------------------------
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf = 64, gc = 32, bias = True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias = bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias = bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias = bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias = bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias = bias)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace = True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc = 32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc = 32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf = nf, gc = gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias = True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias = True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias = True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias = True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias = True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias = True)

        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace = True)

    def forward(self, x):                                       # assume x: [1, 3, 32, 32]
        fea = self.conv_first(x)                                # fea: [1, 64, 32, 32]
        trunk = self.RRDB_trunk(fea)                            # trunk: [1, 64, 32, 32]
        trunk = self.trunk_conv(trunk)                          # trunk: [1, 64, 32, 32]
        fea = fea + trunk                                       # fea: [1, 64, 32, 32]

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor = 2, mode = 'nearest'))) # fea: [1, 64, 64, 64]
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor = 2, mode = 'nearest'))) # fea: [1, 64, 128, 128]
        out = self.conv_last(self.lrelu(self.HRconv(fea)))      # out: [1, 3, 128, 128]

        return out

class RRDBNet1(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc = 32):
        super(RRDBNet1, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias = True)

    def forward(self, x):                                       # assume x: [1, 3, 32, 32]
        fea = self.conv_first(x)                                # fea: [1, 64, 32, 32]

        return fea

class RRDBNet2(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc = 32):
        super(RRDBNet2, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf = nf, gc = gc)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)

    def forward(self, fea):                                     # assume x: [1, 3, 32, 32]
        trunk = self.RRDB_trunk(fea)                            # trunk: [1, 64, 32, 32]

        return trunk

class RRDBNet3(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc = 32):
        super(RRDBNet3, self).__init__()
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias = True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias = True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias = True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias = True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias = True)

        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace = True)

    def forward(self, fea, trunk):                              # assume x: [1, 3, 32, 32]
        trunk = self.trunk_conv(trunk)                          # trunk: [1, 64, 32, 32]
        fea = fea + trunk                                       # fea: [1, 64, 32, 32]

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor = 2, mode = 'nearest'))) # fea: [1, 64, 64, 64]
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor = 2, mode = 'nearest'))) # fea: [1, 64, 128, 128]
        out = self.conv_last(self.lrelu(self.HRconv(fea)))      # out: [1, 3, 128, 128]

        return out

if __name__ == '__main__':
    
    a = torch.randn(1, 3, 32, 32).cuda()
    net1 = RRDBNet1(3, 3, 64, 23, gc = 32).cuda()
    net2 = RRDBNet2(3, 3, 64, 23, gc = 32).cuda()
    net3 = RRDBNet3(3, 3, 64, 23, gc = 32).cuda()
    fea = net1(a)
    flops, params = profile(net1, inputs = (a, ))
    print(flops) # 1966080
    trunk = net2(fea)
    flops, params = profile(net2, inputs = (fea, ))
    print(flops) # 18813714432
    out = net3(fea, trunk)
    flops, params = profile(net3, inputs = (fea, trunk, ))
    print(flops) # 1584398336
    print(out.shape)

    print(net2)
    