import torch
import torch.nn as nn
from torch.nn import init
from experimental.improved_diffusion.unet_adapted import UNetModelAdapted


class CIF_Module(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CIF_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x ):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        out = x * y

        return out


class MCCA(nn.Module):
    def __init__(self,  in_chans=2, out_chans=2, chans=16, num_res_blocks=1, channel_mult = (1, 2, 4, 8) ):
        super(MCCA, self).__init__()
        self.n_resblocks = num_res_blocks
        self.chans = chans
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.channel_mult = channel_mult

        self.net1 = UNetModelAdapted(in_channels=in_chans,model_channels=chans,out_channels=out_chans,num_res_blocks=num_res_blocks,attention_resolutions=(),
                              dropout=0,channel_mult=channel_mult,conv_resample=True, dims=2, num_classes=None,use_checkpoint=False,
                              num_heads=4,num_heads_upsample=-1,use_scale_shift_norm=False)
        self.net2 = UNetModelAdapted(in_channels=in_chans, model_channels=chans,out_channels=out_chans,num_res_blocks=num_res_blocks,attention_resolutions=(),
                              dropout=0,channel_mult=channel_mult,conv_resample=True, dims=2, num_classes=None,use_checkpoint=False,
                              num_heads=4,num_heads_upsample=-1,use_scale_shift_norm=False)

        self.fusion_input = nn.ModuleList()
        self.fusion_middle = nn.ModuleList()
        nfusion = 2
        for level, mult in enumerate(channel_mult):
            ch = mult * chans
            for _ in range(num_res_blocks):
                self.fusion_input.append(nn.Sequential(*[CIF_Module(channel=ch, reduction=16) for _ in range(nfusion)]))
            if level != len(channel_mult) - 1:
                self.fusion_input.append(nn.Sequential(*[CIF_Module(channel=ch, reduction=16) for _ in range(nfusion)]))

        for idx in range(len(self.net1.middle_block)):
            self.fusion_middle.append(nn.Sequential(*[CIF_Module(channel=ch, reduction=16) for _ in range(nfusion)]))

    def forward(self, aux_full, tar_sub):
        hsaux = []
        hstar = []
        aux_full = self.net1.input_blocks[0](aux_full)
        tar_sub = self.net2.input_blocks[0](tar_sub)
        hsaux.append(aux_full)
        hstar.append(tar_sub)

        for idx, (m1, m2, fusion_cm) in enumerate(zip(self.net1.input_blocks[1:]._modules.items(),
                                                      self.net2.input_blocks[1:]._modules.items(),
                                                      self.fusion_input)):
            name1, encodlayer1 = m1
            _, encodlayer2 = m2

            aux_full = encodlayer1(aux_full)
            tar_sub = encodlayer2(tar_sub)
            hsaux.append(aux_full)
            hstar.append(tar_sub)

            attn = fusion_cm(aux_full)
            tar_sub = tar_sub + attn * tar_sub

        for d1, d2, fusion_cm in zip(self.net1.middle_block._modules.items(),
                                    self.net2.middle_block._modules.items(),
                                    self.fusion_middle):
            name1, midlayer1 = d1
            _, midlayer2 = d2
            aux_full = midlayer1(aux_full)
            tar_sub = midlayer2(tar_sub)

            attn = fusion_cm(aux_full)
            tar_sub = tar_sub + attn * tar_sub

        total_layers=len(self.net1.output_blocks)
        for idx, (d1, d2) in enumerate(zip(self.net1.output_blocks._modules.items(), self.net2.output_blocks._modules.items())):
            name1, outlayer1 = d1
            _, outlayer2 = d2
            if idx not in [total_layers-1]:
                aux_full = torch.cat([aux_full, hsaux.pop()], dim=1)
                tar_sub  = torch.cat([tar_sub, hstar.pop()], dim=1)
            aux_full = outlayer1(aux_full)
            tar_sub = outlayer2(tar_sub)

        aux_rec = self.net1.out(aux_full)
        tar_rec = self.net2.out(tar_sub)

        return aux_rec, tar_rec
