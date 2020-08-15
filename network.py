import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x

# ----------------------------------------
#                ESRGAN D
# ----------------------------------------
class SRPatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(SRPatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = nn.Sequential(
            Conv2dLayer(opt.out_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', sn = True),
            Conv2dLayer(opt.start_channels, opt.start_channels, 4, 2, 1, pad_type = opt.pad, norm = 'bn', sn = True)
        )
        self.block2 = nn.Sequential(
            Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = 'bn', sn = True),
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = 'bn', sn = True)
        )
        self.block3 = nn.Sequential(
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = 'bn', sn = True),
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = 'bn', sn = True)
        )
        self.block4 = nn.Sequential(
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = 'bn', sn = True),
            Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = 'bn', sn = True)
        )
        # Final output
        self.final1 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = 'bn', sn = True)
        self.final2 = Conv2dLayer(opt.start_channels * 2, 1, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none', sn = True)

    def forward(self, x):                                       # in: B * 3 * H * W
        x = self.block1(x)                                      # out: B * 64 * H/2 * W/2
        x = self.block2(x)                                      # out: B * 128 * H/4 * W/4
        x = self.block3(x)                                      # out: B * 256 * H/8 * W/8
        x = self.block4(x)                                      # out: B * 512 * H/16 * W/16
        x = self.final1(x)
        x = self.final2(x)                                      # out: B * 1 * H/16 * W/16
        return x


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# VGG style Discriminator with input size 128*128
def conv_block(in_nc, out_nc, kernel_size, stride = 1, dilation = 1, groups = 1, bias = True, \
               pad_type = 'zero', norm_type = None, act_type = 'relu', mode = 'CNA'):
    
    #Conv layer with padding, normalization, activation
    #mode: CNA --> Conv -> Norm -> Act
    #    NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    # cc = nn.Conv2d(in_nc, out_nc, kernel_size=1, stride=stride, padding=padding, \
    #         dilation=dilation, bias=bias, groups=groups)
    # aa = act(act_type) if act_type else None
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

class SR_VGG128_Discriminator(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type = 'batch', act_type = 'leakyrelu', mode = 'CNA'):
        super(SR_VGG128_Discriminator, self).__init__()
        # features
        # hxw, c
        # in_nc = 3
        # base_nf = 64
        # 128, 64
        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, mode=mode)
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        # 64, 64
        conv2 = conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv3 = conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        # 32, 128
        conv4 = conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv5 = conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        # 16, 256
        conv6 = conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv7 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        # 8, 512
        conv8 = conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv9 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        # 4, 512
        self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9)
        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#### network structures
# network_G:
#   which_model_G: MSRResNet
#   in_nc: 3
#   out_nc: 3
#   nf: 64
#   nb: 16
#   upscale: 4
# network_D:
#   which_model_D: discriminator_vgg_128
#   in_nc: 3
#   nf: 64
# ----------------------------------------
#                 ESRGAN G
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

    def forward(self, x, nz):                                       # assume x: [1, 3, 32, 32]
        fea = self.conv_first(x)                                # fea: [1, 64, 32, 32]
        fea = fea + 0.01*nz
        trunk = self.RRDB_trunk(fea)                            # trunk: [1, 64, 32, 32]
        trunk = self.trunk_conv(trunk)                          # trunk: [1, 64, 32, 32]
        fea = fea + trunk                                       # fea: [1, 64, 32, 32]

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor = 2, mode = 'nearest'))) # fea: [1, 64, 64, 64]
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor = 2, mode = 'nearest'))) # fea: [1, 64, 128, 128]
        out = self.conv_last(self.lrelu(self.HRconv(fea)))      # out: [1, 3, 128, 128]

        return out


class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, \
                device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load('./models/VGG16minc_53.pth'), strict=True)
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output