import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class Net(nn.Module):

    # The architecture was designed as a U-Net. The decoder upsamples the resolution and at each scale
    # concatenates the homologous features coming from the encoder (symmetric skips).
    # The encoder is ConvNeXt, and the decoder uses residual blocks (ResNet-style), i.e. an internal shortcut
    # in addition to the encoder skips.
    # Input:  [B, 3, 144, 256], Output: [B, 1, 144, 256]

    class UpSkip(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.act = nn.ReLU(inplace=True)
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False), # Bias disabled to reduce params; with BatchNorm the bias is redundant
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True), # Inplace to reduce memory usage
                nn.Dropout2d(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            # shortcut for residual sum
            self.conv1x1 = nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False)

        def forward(self, x, skip):
            # upsample to the skip resolution, concat and apply residual block
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear') # Upscale x to match skip (could use Upsample but interpolate ensures exact sizes)
                                                                    # Use bilinear because it preserves details better than nearest by interpolating over a neighborhood.
            z = torch.cat([x, skip], dim=1) # concatenate x and skip along channels
            y = self.conv(z) # apply convs to the concatenation and reduce channels to out_ch
            yv = self.conv1x1(z) # compute a shortcut via 1x1 conv to align dimensions
            return self.act(y + yv) # activation on the sum

    def __init__(self):
        super().__init__()
        
        # Encoder: ConvNeXt with 4 scales (1/4, 1/8, 1/16, 1/32)
        # ConvNeXt is a ConvNet proposed in "A ConvNet for the 2020s", showing strong performance vs ViTs.
        # Using timm as the implementation repository which provides pretrained models.
        # Two variants were tried: convnext_tiny and convnext_base
        self.convnext = timm.create_model(
                "convnext_base",
                pretrained=True,
                features_only=True, # Return only intermediate feature maps (useful for U-Net)
                out_indices=(0, 1, 2, 3), # Indices of stages whose feature maps we want (0=1/4, 1=1/8, 2=1/16, 3=1/32)
                drop_path_rate=0.1, # Stochastic depth for the encoder
            )
        # Actual channels from the backbone (Base: 128,256,512,1024; Tiny: 96,192,384,768)

        c1, c2, c3, c4 = self.convnext.feature_info.channels() # Extract channel counts of the 4 encoder stages

        # Since the exact output channels of different models were not always known, apply 1x1 convs to project channels to fixed sizes.
        # This also helps reduce parameter count and control overfitting.
        self.proj4 = nn.Conv2d(c4, 512, 1, bias=False)  # 1/32 and 512 ch
        self.proj3 = nn.Conv2d(c3, 256, 1, bias=False)  # 1/16 and 256 ch
        self.proj2 = nn.Conv2d(c2, 128, 1, bias=False)  # 1/8  and 128 ch
        self.proj1 = nn.Conv2d(c1,  64, 1, bias=False)  # 1/4  and  64 ch

        # Decoder uses a U-Net: at each scale it upsamples, concatenates the encoder skip and
        # applies a residual block (conv + shortcut) to merge information while preserving spatial detail
        self.up3 = Net.UpSkip(in_ch=512, skip_ch=256, out_ch=384)  # f4 (1/32) + f3 (1/16) -> 1/16
        self.up2 = Net.UpSkip(in_ch=384, skip_ch=128, out_ch=192)  # d3 (1/16) + f2 (1/8) -> 1/8
        self.up1 = Net.UpSkip(in_ch=192, skip_ch= 64, out_ch= 96)  # d2 (1/8) + f1 (1/4) -> 1/4

        # Once at 1/4 resolution use two simple upsample stages
        self.up_half = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.up_full = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        # Output head (from 96 channels to 1)
        self.out_head = nn.Conv2d(96, 1, 3, padding=1)

    def forward(self, x):
        # ConvNeXt returns features at 1/4, 1/8, 1/16, 1/32
        f1_4, f1_8, f1_16, f1_32 = self.convnext(x)

        # Channel projections
        f1 = self.proj1(f1_4)     # 1/4  and 64 ch
        f2 = self.proj2(f1_8)     # 1/8  and 128 ch
        f3 = self.proj3(f1_16)    # 1/16 and 256 ch
        f4 = self.proj4(f1_32)    # 1/32 and 512 ch

        # U-Net decoder
        d3 = self.up3(f4, f3)     # 1/16 and 384 ch
        d2 = self.up2(d3, f2)     # 1/8  and 192 ch
        d1 = self.up1(d2, f1)     # 1/4  and 96 ch

        # Upsample to 1/2 and then to full resolution
        d0 = self.up_half(d1)     # 1/2 and 96 ch
        d0 = self.up_full(d0)     # 1/1 and 96 ch

        # Output 
        y  = self.out_head(d0)    # [B,1,H,W]
        return y
