
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath, trunc_normal_
from torch.nn import GELU


from SSCA import SSCA
from module_util import initialize_weights


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, heads):
        """ Transformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = S_MSA(input_dim, dim_head=head_dim, heads=heads)
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = Rearrange('b c h w -> b h w c')(x)
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        x = Rearrange('b h w c -> b c h w')(x)
        return x


class LFEncoder(nn.Module):  # Lightweight Feature Encoder
    def __init__(self, in_channels, last_ch, last_kernel_w, last_stride, last_padding_w, bit_num):
        super(LFEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 128, [3, 2], stride=[2, 2], padding=[1, 0])
        self.conv2 = nn.Conv2d(128, 64, [3, 1], stride=[1, 1], padding=[1, 0])
        self.conv3 = nn.Conv2d(64, last_ch, [3, last_kernel_w], stride=[last_stride, 2], padding=[1, last_padding_w])
        self.convlayer = nn.Conv2d(32, 32, 1, 1, 0)
        self.trans = Block(32, 32, 16, 2)
        self.relu = nn.LeakyReLU(True)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        conv_x, trans_x = torch.split(x, [32, 32], dim=1)
        conv_x = self.relu(self.convlayer(conv_x)) + conv_x
        trans_x = self.trans(trans_x)
        x = self.relu(self.conv3(torch.cat((conv_x, trans_x), dim=1)))

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class S_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        mask: [1,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class MS_FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        # 1x1 convolutions to adjust channel dimensions
        self.conv1x1_1 = nn.Conv2d(dim, dim * mult, 1, 1, bias=False)
        self.conv1x1_2 = nn.Conv2d(dim * mult, dim, 1, 1, bias=False)

        # Depthwise convolutions with different kernel sizes
        self.DWconv1x1 = nn.Conv2d(dim * mult // 4, dim * mult // 4, kernel_size=1, bias=False, groups=dim * mult // 4)
        self.DWconv3x3 = nn.Conv2d(dim * mult // 4, dim * mult // 4, kernel_size=3, padding=1, bias=False,
                                   groups=dim * mult // 4)
        self.DWconv5x5 = nn.Conv2d(dim * mult // 4, dim * mult // 4, kernel_size=5, padding=2, bias=False,
                                   groups=dim * mult // 4)
        self.DWconv7x7 = nn.Conv2d(dim * mult // 4, dim * mult // 4, kernel_size=7, padding=3, bias=False,
                                   groups=dim * mult // 4)

    def forward(self, x):
        x = self.conv1x1_1(x.permute(0, 3, 1, 2))

        # Split along the channel axis
        split_size = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, split_size, dim=1)

        # Apply depthwise convolutions
        out1 = self.DWconv1x1(x1)
        out2 = self.DWconv3x3(x2)
        out3 = self.DWconv5x5(x3)
        out4 = self.DWconv7x7(x4)


        # Concatenate the outputs along the channel axis
        out = torch.cat([out1, out2, out3, out4], dim=1) + x

        # Final 1x1 convolution
        out = self.conv1x1_2(out)

        return out.permute(0, 2, 3, 1)


class SpectralT_Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                S_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, MS_FFN(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x  # 应用自注意力并添加残差连接
            x = ff(x) + x  # 在经过自注意力的输出上应用前馈神经网络并添加残差连接
        out = x.permute(0, 3, 1, 2)
        return out


class RDFF(nn.Module):  # Residual Feature Extraction Block
    def __init__(self, nf, gc=32, bias=True):
        super(RDFF, self).__init__()

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class MAFE(nn.Module):  # Feature Enhanced Model With Attention
    def __init__(self, nf, gc=32):
        super(MAFE, self).__init__()
        self.RDFF1 = RDFF(nf, gc)
        self.RDFF2 = RDFF(nf, gc)
        self.RDFF3 = RDFF(nf, gc)
        self.SSCA = SSCA(nf)

    def forward(self, x):
        out = self.RDFF1(x)
        out = self.RDFF2(out)
        out = out * 0.2 + x
        out = self.RDFF3(out)
        att = self.SSCA(out)
        out = att * out
        return out * 0.2 + x


class CTMBlock(nn.Module):
    def __init__(self, dim, gc, head_dim, heads):
        """ Transformer and Conv Block
        """
        super(CTMBlock, self).__init__()
        self.conv_dim = dim
        self.trans_dim = dim
        self.head_dim = head_dim
        self.heads = heads
        self.trans_block = SpectralT_Block(self.trans_dim, self.head_dim, self.heads, num_blocks=2)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv_block = MAFE(self.conv_dim, gc)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), [self.conv_dim, self.trans_dim], dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x


class CTDecoder(nn.Module):  # CNN-Transformer SR Decoder
    def make_layer(self, block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def __init__(self, in_nc, out_nc, nf, nb, gc=32, up_scale=4):
        super(CTDecoder, self).__init__()
        ctmb_block_f = partial(CTMBlock, dim=32, gc=gc, head_dim=16, heads=2)
        self.up_scale = up_scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        self.CTMB_trunk = self.make_layer(ctmb_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.upconv1 = nn.Conv2d(nf // 4, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 2 * nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(2 * nf // 4, nf + gc, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv2d(nf + gc, out_nc, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(2 * nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        fea = self.conv_first(x)  # 用来嵌入的卷积，x是27，32，1，fea是64，32，1
        trunk = self.trunk_conv(self.CTMB_trunk(fea))  # 多尺度融合块 + 卷积
        fea = fea + trunk

        # spatial/spectral SR
        if self.up_scale == 2:
            fea = self.lrelu(self.upconv1(self.pixel_shuffle(fea)))
            fea = self.conv_last(self.lrelu(self.upconv2(fea)))
        if self.up_scale == 4:
            fea = self.lrelu(self.upconv2(self.lrelu(self.upconv1(self.pixel_shuffle(fea)))))
            fea = self.upconv4(self.lrelu(self.upconv3(self.pixel_shuffle(fea))))
        return fea


class CTCSN(nn.Module):  # Joint CNN and Transformer-Based Deep Learning Hyperspectral Image Compression-Sensing Network
    def __init__(self, snr=0, cr=1, bit_num=8):
        super(CTCSN, self).__init__()
        self.snr = snr
        if cr == 1:
            last_stride = 2
            last_ch = 27
            last_kernel_w = 1
            last_padding_w = 0
        else:
            last_stride = 1
            last_kernel_w = 2
            last_padding_w = 1

        up_scale = 4 if cr < 5 else 2

        if cr == 5:
            last_ch = 32
        elif cr == 10:
            last_ch = 64
        elif cr == 15:
            last_ch = 103
        elif cr == 20:
            last_ch = 140

        ## 128*4*172=88064 -->
        ## 32*1*27 --> cr=1%
        ## 64*2*64 --> cr=9.30%
        ## 64*2*32 --> cr=4.65%
        ## 64*2*103--->cr=14.97%
        ## 64*2*140 -->cr=20.3%

        self.encoder = LFEncoder(172, last_ch, last_kernel_w, last_stride, last_padding_w, bit_num=bit_num)

        self.decoder = CTDecoder(last_ch, 172, 64, 16, up_scale=up_scale)

    def awgn(self, x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = torch.sum(x ** 2) / x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape).cuda() * npower

    #def mixed_noise(self, x, snr, scale=1):
    #    """Add mixed Gaussian and salt-and-pepper noise to the input tensor."""

    #    snr = 10 ** (snr / 10.0)
    #    xpower = torch.sum(x ** 2) / x.numel()
    #    npower = torch.sqrt(xpower / snr)

    #    x = x + torch.randn(x.shape).cuda() * npower

    #    prob = snr * scale
    #    salt_mask = (torch.rand(x.shape) < prob).cuda()
    #    pepper_mask = (torch.rand(x.shape) < prob).cuda()
    #    salt_and_pepper = torch.zeros_like(x).cuda()
    #    salt_and_pepper[salt_mask] = 1
    #    salt_and_pepper[pepper_mask] = 0

    #    return x + salt_and_pepper

    def forward(self, data, mode=0):  # Mode=0, default, mode=1: encode only, mode=2: decoded only

        if mode == 0:
            x = self.encoder(data)
            if self.snr > 0:
                x = self.awgn(x, self.snr)
            y = self.decoder(x)
            return y, x
        elif mode == 1:
            x = self.encoder(data)
            return x
        elif mode == 2:
            return self.decoder(data)
        else:
            x = self.encoder(data)
            return self.decoder(x)