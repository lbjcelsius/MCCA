"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import hashlib
from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from fastmri import MriModule

from experimental.improved_diffusion.MCCANet import MCCA
from fastmri.math import DataConsistency


class MCCA_Module(MriModule):
    def __init__(
            self,
            in_chans=2,
            out_chans=2,
            chans=32,
            n_resblocks=1,
            lr=0.001,
            lr_step_size=40,
            lr_gamma=0.1,
            weight_decay=0.0,
            **kwargs,
    ):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net
                model.
            chans (int): Number of channels of the middle convolution
                layer.
            num_pool_layers (int): Number of down-sampling and up-sampling
                layers.
            drop_prob (float): Dropout probability.
            lr (float): Learning rate.
            lr_step_size (int): Learning rate step size.
            lr_gamma (float): Learning rate gamma decay.
            weight_decay (float): Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)
        self.n_resblocks = n_resblocks
        self.chans = chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.rec_model = MCCA(
                              in_chans=in_chans,
                              out_chans=out_chans,
                              chans=self.chans,
                              num_res_blocks=self.n_resblocks,
                              )
        self.dc = DataConsistency()

    def forward(self, aux_full, tar_sub):
        aux_rec, tar_rec = self.rec_model(aux_full, tar_sub)

        return aux_rec, tar_rec

    def training_step(self, batch, batch_idx):
        tar_sub, tar_full, mean, std, fname, slice_num, k_tar_full, mask = batch['tar']
        _, aux_full, *_   = batch['aux']
        aux_rec, tar_rec = self(aux_full, tar_sub)
        tar_dc, k_pdfs_dc = self.dc(tar_rec, k_tar_full, mask)

        loss_tar_dc = F.l1_loss(tar_dc, tar_full)
        loss_tar_k = F.l1_loss(k_pdfs_dc, k_tar_full)
        loss_aux = F.l1_loss(aux_rec, aux_full)

        loss = 0.6 * loss_tar_dc + 0.4 * loss_aux + 0.01 * loss_tar_k

        self.log('loss', loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        tar_sub, tar_full, mean, std, fname, slice_num, k_tar_full, mask = batch['tar']
        _, aux_full, *_   = batch['aux']
        aux_rec, tar_rec = self(aux_full, tar_sub)
        tar_dc, k_pdfs_dc = self.dc(tar_rec, k_tar_full, mask)

        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        output = tar_dc
        # hash strings to int so pytorch can concat them
        fnumber = torch.zeros(len(fname), dtype=torch.long, device=output.device)
        for i, fn in enumerate(fname):
            fnumber[i] = (int(hashlib.sha256(fn.encode("utf-8")).hexdigest(), 16) % 10 ** 12)

        return {
            "fname": fnumber,
            "slice": slice_num,
            "output": (torch.sqrt((output**2).sum(dim=1))   ) * std + mean,
            "target": (torch.sqrt((tar_full**2).sum(dim=1)) ) * std + mean,
            "val_loss": F.l1_loss(output, tar_full),
        }

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument("--in_chans", default=2, type=int)
        parser.add_argument("--out_chans", default=2, type=int)
        parser.add_argument("--chans", default=64, type=int)
        parser.add_argument("--n_resblocks", default=1, type=int)
        parser.add_argument("--num_pool_layers", default=4, type=int)
        parser.add_argument("--drop_prob", default=0.0, type=float)

        # training params (opt)
        parser.add_argument("--lr", default=0.0001, type=float)
        parser.add_argument("--lr_step_size", default=40, type=int)
        parser.add_argument("--lr_gamma", default=0.1, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)

        return parser

