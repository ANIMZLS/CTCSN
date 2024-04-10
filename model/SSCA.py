import torch
import torch.nn as nn
import torch.nn.functional as F


class SPE(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(SPE, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1, padding=0, bias=True)
        )
        self.conv2x1 = nn.Conv2d(gate_channels, gate_channels, kernel_size=(2, 1), groups=gate_channels)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_pool = self.mlp(avg_pool)
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = self.mlp(max_pool)
        cat = torch.cat([max_pool, avg_pool], dim=2)
        spectral_vector = self.conv2x1(cat)
        spectral_weight = torch.sigmoid(spectral_vector)

        return spectral_weight


class SPA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.convlayer = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        spatial_vector = self.convlayer(cat)
        spatial_weight = torch.sigmoid(spatial_vector)

        return spatial_weight


class SSCA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(SSCA, self).__init__()
        self.SPE = SPE(gate_channels, reduction_ratio)
        self.SPA = SPA(kernel_size=7)

    def forward(self, x):
        spatial_weight = self.SPA(x)
        spectral_weight = self.SPE(x)
        combine_weight = torch.sigmoid(spatial_weight + spectral_weight)

        return combine_weight

