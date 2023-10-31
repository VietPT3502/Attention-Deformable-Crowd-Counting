import torch.nn as nn
import torch
from .backbone import MSCANet
import timm
import torch.nn.functional as F

class AttCrowd(nn.Module):
    def __init__(self):
        super(AttCrowd, self).__init__()

        self.backbone = MSCANet()
        # multi-scale combination
        self.deconv1 = nn.ConvTranspose2d(256, 460, 3, stride=2, padding=1, output_padding=1, dilation=2)
        self.deconv2 = nn.ConvTranspose2d(460, 64, 3, stride=2, padding=1, output_padding=1, dilation=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1, dilation=2)

        self.conv1 = nn.Conv2d(920, 460, 1)
        self.conv2 = nn.Conv2d(460, 460, 3, padding=1, dilation=1)

        self.conv3 = nn.Conv2d(128, 64, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)

        self.conv5 = nn.Conv2d(64, 32, 1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1, dilation=1)

        self.relu = nn.ReLU()
        # decoder
        self.decoder1 = Decoder(in_channels=32, out_channels=1)
        self.decoder2 = Decoder(in_channels=64, out_channels=1)
        self.decoder3 = Decoder(in_channels=460, out_channels=1)

        # attention layers
        self.s_weight1 = SpatialWeightLayer()
        self.s_weight2 = SpatialWeightLayer()
        self.s_weight3 = SpatialWeightLayer()

        self._initialize_weights()


    def forward(self, x):
        f1, f2, f3, f4= self.backbone(x)
        f1_size = (f1.shape[2], f1.shape[3])
        f2_size = (f2.shape[2], f2.shape[3])
        f3_size = (f3.shape[2], f3.shape[3])
        g3 = self.deconv1(f4)
        g3 = self.relu(g3)
        g3 = nn.Upsample(size=f3_size, mode='bilinear',
                         align_corners=True)(g3)
        g3 = torch.cat((g3, f3), 1)
        g3 = self.conv1(g3)
        g3 = self.relu(g3)
        g3 = self.conv2(g3)
        g3 = self.relu(g3)
        g2 = self.deconv2(g3)
        g2 = self.relu(g2)
        g2 = nn.Upsample(size=f2_size, mode='bilinear',
                         align_corners=True)(g2)
        g2 = torch.cat((g2, f2), 1)
        g2 = self.conv3(g2)
        g2 = self.relu(g2)
        g2 = self.conv4(g2)
        g2 = self.relu(g2)

        g1 = self.deconv3(g2)
        g1 = self.relu(g1)
        g1 = nn.Upsample(size=f1_size, mode='bilinear',
                         align_corners=True)(g1)
        g1 = torch.cat((g1, f1), 1)
        g1 = self.conv5(g1)
        g1 = self.relu(g1)
        g1 = self.conv6(g1)
        g1 = self.relu(g1)

        g3 = nn.Upsample(size=f2_size, mode='bilinear',
                        align_corners=True)(g3)
        g2 = nn.Upsample(size=f1_size, mode='bilinear',
                        align_corners=True)(g2)
        g1 = nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True)(g1)
        
        g1 = self.s_weight1(g1)
        g2 = self.s_weight2(g2)
        g3 = self.s_weight3(g3)

        # generate density map
        den1 = self.decoder1(g1)  # x2 resolution
        den2 = self.decoder2(g2)  # x4 resolution
        den3 = self.decoder3(g3)  # x8 resolution


        return den1, den2, den3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




class SpatialWeightLayer(nn.Module):
    def __init__(self):
        super(SpatialWeightLayer, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
            kernel_size - 1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        x_out = torch.clamp(x_out, max=15)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, groups, dilation, drop_p):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, groups, dilation)

        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.drop_out = nn.Dropout(drop_p)
        self.activation = nn.LeakyReLU()
        # define residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.residual(x)

        x = self.drop_out(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)

        # add residual connection
        x = x + residual
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.decoder_block_1 = DecoderBlock(in_channels=in_channels, out_channels=in_channels//2, 
                                            kernel=3, stride=1, padding=1, groups=1, dilation=2, drop_p=0.3)
        self.out = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels,
                             kernel_size=1, stride=1, padding=0, groups=1,  dilation=1)
        self.activation_out = nn.ReLU()

    def forward(self, x):
        x = self.decoder_block_1(x)
        x = self.activation_out(self.out(x))
        return x

# model = AttCrowd()
# model.to("cuda")


# y = torch.randn((1,3,1080,1920)).to('cuda' if torch.cuda.is_available() else 'cpu')
# x = model.forward(y)

# for i in x:
#     print(i.shape)