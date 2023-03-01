
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.custom_layers import EqualLinear
import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, input_dim, block, num_blocks, channel_multiplier=1, feature_dim=512):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, self.feature_dim, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(self.feature_dim*channel_multiplier, self.feature_dim, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(self.feature_dim*channel_multiplier, self.feature_dim, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(self.feature_dim*channel_multiplier, self.feature_dim, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, self.feature_dim*channel_multiplier, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, self.feature_dim*channel_multiplier, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, self.feature_dim*channel_multiplier, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4,p5

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
                conv3x3(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )

class ToStyleCode(nn.Module):
    def __init__(self, n_convs, input_dim=512, out_dim=512):
        super(ToStyleCode, self).__init__()
        self.convs = nn.ModuleList()
        self.out_dim = out_dim

        for i in range(n_convs):
            if i == 0:
                self.convs.append(
                nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=2))
                #self.convs.append(nn.BatchNorm2d(out_dim))
                #self.convs.append(nn.InstanceNorm2d(out_dim))
                self.convs.append(nn.LeakyReLU(inplace=True))
            else:
                self.convs.append(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=2))
                self.convs.append(nn.LeakyReLU(inplace=True))
        
        self.convs = nn.Sequential(*self.convs)
        self.linear = EqualLinear(out_dim, out_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_dim)
        x = self.linear(x)
        return x


class ToStyleHead(nn.Module):
    def __init__(self, input_dim=512, out_dim=512):
        super(ToStyleHead, self).__init__()
        self.out_dim = out_dim

        self.convs = nn.Sequential(
            conv3x3_bn_relu(input_dim, input_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            # output 1x1
            nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0],self.out_dim)
        return x


class ToInputNoise(nn.Module):
    def __init__(self, n_convs, input_dim=512, out_dim=512):
        super(ToInputNoise, self).__init__()
        self.convs = nn.ModuleList()
        self.out_dim = out_dim

        for i in range(n_convs):
            if i == 0:
                self.convs.append(
                    nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=2))
                # self.convs.append(nn.BatchNorm2d(out_dim))
                # self.convs.append(nn.InstanceNorm2d(out_dim))
                self.convs.append(nn.LeakyReLU(inplace=True))
            else:
                self.convs.append(
                    nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=2))
                self.convs.append(nn.LeakyReLU(inplace=True))

        self.convs = nn.Sequential(*self.convs)
        self.linear = EqualLinear(out_dim, out_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_dim)
        x = self.linear(x)
        return x


class ToLayerNoise(nn.Module):
    def __init__(self, n_convs, input_dim=512, out_dim=512):
        super(ToLayerNoise, self).__init__()
        self.convs = nn.ModuleList()
        self.out_dim = out_dim

        for i in range(n_convs):
            if i == 0:
                self.convs.append(
                    nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=1, padding=0, stride=1))
                self.convs.append(nn.LeakyReLU(inplace=True))
            else:
                self.convs.append(
                    nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, padding=0, stride=1))
                self.convs.append(nn.LeakyReLU(inplace=True))

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        return x


class ToSingleDimLayerNoise(nn.Module):
    def __init__(self, n_convs=6, input_dim=512, out_dim=1, upsample=1):
        super(ToSingleDimLayerNoise, self).__init__()
        self.convs = nn.ModuleList()
        self.out_dim = out_dim

        # 512 -> 512
        self.convs.append(
            nn.Conv2d(in_channels=input_dim, out_channels=512, kernel_size=1, padding=0, stride=1))
        self.convs.append(nn.LeakyReLU(inplace=True))

        # 512 -> 512
        self.convs.append(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1))
        self.convs.append(nn.LeakyReLU(inplace=True))

        # 512 -> 256
        if upsample == 4:
            self.convs.append(
                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, padding=1, stride=2))
            self.convs.append(nn.LeakyReLU(inplace=True))
        else:
            self.convs.append(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1))
            self.convs.append(nn.LeakyReLU(inplace=True))


        # 256 -> 128
        if upsample == 2  or upsample == 4:
            self.convs.append(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, padding=1, stride=2))
            self.convs.append(nn.LeakyReLU(inplace=True))
        else:
            self.convs.append(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1))
            self.convs.append(nn.LeakyReLU(inplace=True))

        # 128 -> 64
        self.convs.append(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1))
        self.convs.append(nn.LeakyReLU(inplace=True))

        # 64-> 1
        self.convs.append(
            nn.Conv2d(in_channels=64, out_channels=out_dim, kernel_size=1, padding=0, stride=1))
        self.convs.append(nn.LeakyReLU(inplace=True))

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        return x


class FPNEncoder(nn.Module):
    def __init__(self, input_dim, n_latent=14, use_style_head=False, style_layers=[4,5,6],fpn_feature_dim=512,
                 out_dim=512,noise_predict_from=64,noise_predict_until=64,
                 single_dim_noise=False, noise_dim=512, input_mode='random',resolution=256):
        # input_mode: random, const

        super(FPNEncoder, self).__init__()
        print('------> FPN style layers: ', style_layers)
        print('------> FPN feature dim: ', fpn_feature_dim)
        self.resolution = resolution
        self.n_latent = n_latent
        num_blocks = [3,4,6,3] #resnet 50
        self.FPN_module = FPN(input_dim, Bottleneck, num_blocks, feature_dim=fpn_feature_dim)

        # course block 0-2, 4x4->8x8
        self.course_styles = nn.ModuleList()
        for i in range(3):
            if use_style_head:
                self.course_styles.append(ToStyleHead())
            else:
                self.course_styles.append(ToStyleCode(n_convs=style_layers[0],input_dim=fpn_feature_dim,out_dim=out_dim))
        # medium1 block 3-6 16x16->32x32
        self.medium_styles = nn.ModuleList()
        for i in range(4):
            if use_style_head:
                self.medium_styles.append(ToStyleHead())
            else:
                self.medium_styles.append(ToStyleCode(n_convs=style_layers[1],input_dim=fpn_feature_dim,out_dim=out_dim))
        # fine block 7-13 64x64->256x256
        self.fine_styles = nn.ModuleList()
        for i in range(n_latent - 7):
            if use_style_head:
                self.fine_styles.append(ToStyleHead())
            else:
                self.fine_styles.append(ToStyleCode(n_convs=style_layers[2],input_dim=fpn_feature_dim,out_dim=out_dim))

        # To input noise
        self.input_mode = input_mode
        if input_mode == 'random':
            self.toInputNoise = ToStyleCode(n_convs=style_layers[0],input_dim=fpn_feature_dim,out_dim=out_dim)

        # To layer noise
        self.single_dim_noise = single_dim_noise
        self.noise_predict_from = noise_predict_from
        self.noise_predict_until = noise_predict_until
        log2_start = int(math.log2(noise_predict_from))
        log2_until = int(math.log2(noise_predict_until))
        self.num_noise = (log2_until - log2_start + 1) * 2
        self.resolution_list = [2**i for i in range(log2_start, log2_until+1)]
        print('Encoder nosie prediction at : ', self.resolution_list)
        channel_base = 32768  # Overall multiplier for the number of channels.
        channel_max = 512  # Maximum number of channels in any layer.

        if single_dim_noise:
            self.noise_dim = 1
        else:
            self.noise_dim = noise_dim

        if noise_predict_until == resolution//2 or noise_predict_until == resolution:
            assert self.noise_dim == 1

        if noise_predict_until == resolution//2:
            normal_noise_layer = self.num_noise - 2
        elif noise_predict_until == resolution:
            normal_noise_layer = self.num_noise - 4
        else:
            normal_noise_layer  = self.num_noise

        self.noise_net = nn.ModuleList()
        for i in range(int(normal_noise_layer)):
            cur_res = noise_predict_from * (2 ** (i // 2))
            out_dim = min(channel_base // cur_res, self.noise_dim)
            if out_dim <= 64:
                self.noise_net.append(ToSingleDimLayerNoise(input_dim=fpn_feature_dim,out_dim=out_dim))
            else:
                self.noise_net.append(ToLayerNoise(5,input_dim=fpn_feature_dim,out_dim=out_dim))

        if noise_predict_until == resolution//2 or noise_predict_until==resolution:
            out_dim = min(channel_base // (resolution//2), self.noise_dim)
            self.noise_net.append(ToSingleDimLayerNoise(input_dim=fpn_feature_dim,out_dim=out_dim,upsample=2))
            self.noise_net.append(ToSingleDimLayerNoise(input_dim=fpn_feature_dim,out_dim=out_dim,upsample=2))

        if noise_predict_until == resolution:
            out_dim = min(channel_base // resolution, self.noise_dim)
            self.noise_net.append(ToSingleDimLayerNoise(input_dim=fpn_feature_dim,out_dim=out_dim,upsample=4))
            self.noise_net.append(ToSingleDimLayerNoise(input_dim=fpn_feature_dim,out_dim=out_dim,upsample=4))


    def forward(self, x):
        styles = []
        layer_nosie = []

        # FPN feature
        p2, p3, p4, p5 = self.FPN_module(x)
        noise_layer_count = 0

        # input noise
        if self.input_mode == 'random':
            input_noise = self.toInputNoise(p4)
        else:
            input_noise = None

        if p5.shape[-1] in self.resolution_list:
            for i in range(2):
                layer_nosie.append(self.noise_net[i](p5))
                noise_layer_count += 1

        for k,style_map in enumerate(self.course_styles):
            styles.append(style_map(p4))
            if k < 2 and p4.shape[-1] in self.resolution_list:
                layer_nosie.append(self.noise_net[noise_layer_count](p4))
                noise_layer_count += 1

        for k,style_map in enumerate(self.medium_styles):
            styles.append(style_map(p3))
            if k < 2 and p3.shape[-1] in self.resolution_list:
                layer_nosie.append(self.noise_net[noise_layer_count](p3))
                noise_layer_count += 1
            
        for k,style_map in enumerate(self.fine_styles):
            styles.append(style_map(p2))
            if k < 2 and p2.shape[-1] in self.resolution_list:
                layer_nosie.append(self.noise_net[noise_layer_count](p2))
                noise_layer_count += 1

        if self.resolution//2 in self.resolution_list:
            for i in range(2):
                layer_nosie.append(self.noise_net[noise_layer_count](p2))
                noise_layer_count += 1

        if self.resolution in self.resolution_list:
            for i in range(2):
                layer_nosie.append(self.noise_net[noise_layer_count](p2))
                noise_layer_count += 1

        styles = torch.stack(styles, dim=1)

        return styles, input_noise, layer_nosie
