import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


from DBnetworks.DCSAU.csa_layer import CSA


class DWT_Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWT_Downsampling, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, 1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        out = torch.cat([x1 + x2 + x3 + x4,
                         -x1 - x2 + x3 + x4,
                         -x1 + x2 - x3 + x4,
                         x1 - x2 - x3 + x4], dim=1)
        return self.relu(self.bn(self.conv(out)))


class PFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels))
        self.depthwise = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, groups=out_channels, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels))
        self.pointwise = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x


class SpectralGatingNetwork(nn.Module):
    def __init__(self, channels):
        super(SpectralGatingNetwork, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, channels, 1, 1, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(torch.view_as_real(self.complex_weight))
        x_fft = x_fft * weight
        x_out = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')
        return x_out


class Light_SFFM(nn.Module):
    def __init__(self, in_channels, out_channels_needed):
        super(Light_SFFM, self).__init__()

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.spectral_branch = SpectralGatingNetwork(in_channels)

        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

        self.final_proj = None
        sffm_out_channels = in_channels * 2

        if out_channels_needed != sffm_out_channels:
            self.final_proj = nn.Conv2d(sffm_out_channels, out_channels_needed, 1, bias=False)
            self.final_bn = nn.BatchNorm2d(out_channels_needed)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_skip, x_deep):
        if x_skip.shape[1] != x_deep.shape[1]:
            raise RuntimeError(
                f"Channel Mismatch in SFFM: Skip={x_skip.shape[1]}, Deep={x_deep.shape[1]}. They must match for Element-wise Sum.")

        if x_deep.shape[2:] != x_skip.shape[2:]:
            x_deep = F.interpolate(x_deep, size=x_skip.shape[2:], mode='bilinear', align_corners=True)

        x_mixed = x_skip + x_deep

        feat_spatial = self.spatial_branch(x_mixed)
        feat_spectral = self.spectral_branch(x_mixed)

        gate_feat = torch.cat([feat_spatial, feat_spectral], dim=1)
        gate = self.gate_conv(gate_feat)

        feat_spatial = feat_spatial * gate
        feat_spectral = feat_spectral * (1 - gate)

        out = torch.cat([feat_spatial, feat_spectral], dim=1)

        if self.final_proj is not None:
            out = self.final_proj(out)
            out = self.final_bn(out)

        return self.relu(out)


class Edge_Guided_EGAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Edge_Guided_EGAB, self).__init__()

        self.mid_channels = in_channels // 2


        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, 1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )


        self.smooth_layer = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)


        self.edge_att_conv = nn.Sequential(
            nn.Conv2d(self.mid_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        self.final_conv = nn.Conv2d(in_channels + self.mid_channels, out_channels, 1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_reduced = self.reduce_conv(x)
        x_smooth = self.smooth_layer(x_reduced)
        x_edge = torch.abs(x_reduced - x_smooth)
        att_map = self.edge_att_conv(x_edge)
        x_sharpened = x_reduced * (1 + att_map)
        out = torch.cat([x, x_sharpened], dim=1)
        out = self.final_conv(out)
        out = self.relu(self.bn_final(out))
        return out


class Model(nn.Module):
    def __init__(self, img_channels=3, n_classes=1, base_c=32):
        super(Model, self).__init__()
        self.base_c = base_c

        csa_model = CSA(base_c=base_c)


        self.pfc = PFC(in_channels=img_channels, out_channels=base_c, kernel_size=3)
        self.dwt_stem = DWT_Downsampling(in_channels=base_c, out_channels=base_c)


        self.down1 = csa_model.layer1


        self.dwt2 = DWT_Downsampling(in_channels=base_c * 2, out_channels=base_c * 2)
        self.down2 = csa_model.layer2


        self.dwt3 = DWT_Downsampling(in_channels=base_c * 4, out_channels=base_c * 4)
        self.down3 = csa_model.layer3


        self.dwt4 = DWT_Downsampling(in_channels=base_c * 8, out_channels=base_c * 8)
        self.down4 = csa_model.layer4


        self.up1 = csa_model.layer5
        self.up2 = csa_model.layer6
        self.up3 = csa_model.layer7
        self.up4 = csa_model.layer8


        self.sffm1 = Light_SFFM(in_channels=base_c * 8, out_channels_needed=base_c * 16)


        self.egab1 = Edge_Guided_EGAB(in_channels=base_c * 4, out_channels=base_c * 4)


        self.sffm2 = Light_SFFM(in_channels=base_c * 4, out_channels_needed=base_c * 8)


        self.egab2 = Edge_Guided_EGAB(in_channels=base_c * 2, out_channels=base_c * 2)


        self.sffm3 = Light_SFFM(in_channels=base_c * 2, out_channels_needed=base_c * 4)


        self.egab3 = Edge_Guided_EGAB(in_channels=base_c, out_channels=base_c)


        self.sffm4 = Light_SFFM(in_channels=base_c, out_channels_needed=base_c * 2)


        self.egab4 = Edge_Guided_EGAB(in_channels=base_c, out_channels=base_c)


        self.out_conv = nn.Conv2d(base_c, n_classes, kernel_size=1)

    def forward(self, x):

        x_stem = self.pfc(x)
        x_dwt1 = self.dwt_stem(x_stem)
        x_l1 = self.down1(x_dwt1)
        x_dwt2 = self.dwt2(x_l1)
        x_l2 = self.down2(x_dwt2)
        x_dwt3 = self.dwt3(x_l2)
        x_l3 = self.down3(x_dwt3)
        x_dwt4 = self.dwt4(x_l3)
        x_l4 = self.down4(x_dwt4)


        x_fuse1 = self.sffm1(x_skip=x_l3, x_deep=x_l4)

        x_up1 = self.up1(x_fuse1)

        x_dec1 = self.egab1(x_up1)


        x_fuse2 = self.sffm2(x_skip=x_l2, x_deep=x_dec1)

        x_up2 = self.up2(x_fuse2)

        x_dec2 = self.egab2(x_up2)


        x_fuse3 = self.sffm3(x_skip=x_l1, x_deep=x_dec2)

        x_up3 = self.up3(x_fuse3)

        x_dec3 = self.egab3(x_up3)


        x_fuse4 = self.sffm4(x_skip=x_stem, x_deep=x_dec3)

        x_up4 = self.up4(x_fuse4)

        x_final = self.egab4(x_up4)

        out = self.out_conv(x_final)
        return out
