import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchviz import make_dot
import torch

def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MnasMulti(nn.Module):

    def __init__(self, alpha=1.0):
        super(MnasMulti, self).__init__()
        depths = _get_depths(alpha)
        if alpha == 1.0:
            MNASNet = torchvision.models.mnasnet1_0(pretrained=True, progress=True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)

        self.conv0 = nn.Sequential(
            MNASNet.layers._modules['0'],
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
            MNASNet.layers._modules['8'],
        )

        self.conv1 = MNASNet.layers._modules['9']
        self.conv2 = MNASNet.layers._modules['10']

        self.out1 = nn.Conv2d(depths[4], depths[4], 1, bias=False)
        self.out_channels = [depths[4]]

        final_chs = depths[4]
        self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True)

        self.out2 = nn.Conv2d(final_chs, depths[3], 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, depths[2], 3, padding=1, bias=False)
        self.out_channels.append(depths[3])
        self.out_channels.append(depths[2])

        self.model = nn.Sequential(self.conv0, self.conv1, self.conv2, self.out1)

    def forward(self, x):
        conv0 = self.conv0(x)
        print(conv0.shape)
        conv1 = self.conv1(conv0)
        print(conv1.shape)
        conv2 = self.conv2(conv1)
        print(conv2.shape)

        intra_feat = conv2
        outputs = []
        out = self.out1(intra_feat)
        print(out.shape)
        outputs.append(out)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        print(out.shape)
        outputs.append(out)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        print(out.shape)
        outputs.append(out)

        return outputs[::-1]


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        depths = [12, 6, 3]
        # 24, 120, 160
        self.out1 = nn.Conv2d(24, 24, 1, bias=False)
        self.out2 = nn.Conv2d(24, depths[0], 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(depths[0], depths[0], 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(depths[0], depths[1], 3, padding=1, bias=False)
        self.out5 = nn.Conv2d(depths[1], depths[2], 3, padding=1, bias=False)

    def forward(self, x):
        print(x.shape)
        out = self.out1(x)
        print(f'out1 shape: {out.shape}')

        interpolated_features = F.interpolate(out, scale_factor=2, mode="nearest")
        print(f'intp_feat: {interpolated_features.shape}')
        out = self.out2(interpolated_features)
        print(f'out2 shape: {out.shape}')

        out = self.out3(out)
        print(f'out3 shape: {out.shape}')

        interpolated_features = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.out4(interpolated_features)
        print(f'out4 shape: {out.shape}')

        out = self.out5(out)
        print(f'out5 shape: {out.shape}')

        return out

# model = MnasMulti()
# x = torch.randn(1, 3, 480, 640).requires_grad_(True)
# y = model(x)
# dot=make_dot(y, params=dict(list(model.named_parameters())))
# dot.format='png'
# dot.render("/home/gupta.anik/mnasneto")

model = Decoder()
x = torch.randn(1, 24, 120, 160).requires_grad_(True)
y = model(x)