"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser
from collections import defaultdict
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler
from fastmri.data import transforms
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri import evaluate
from fastmri.data.fastmri_kneemri import SliceDataset
from fastmri.data.volume_sampler import VolumeSampler
from fastmri.evaluate import DistributedMetricSum


class MriModule(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - fastMRI data loaders
        - Evaluating reconstructions
        - Visualization
        - Saving test reconstructions

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - train_data_transform, val_data_transform, test_data_transform:
            Create and return data transformer objects for each data split
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(
        self,
        data_path,
        challenge,
        mask_type="random",
        center_fractions=[0.08],
        accelerations=[4],
        test_split="test",
        sample_rate=1.0,
        batch_size=1,
        num_workers=4,
        data_type=None,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.challenge = challenge

        self.mask_type = mask_type
        self.center_fractions = center_fractions
        self.accelerations = accelerations

        self.test_split = test_split
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_type = data_type

        self.NMSE = DistributedMetricSum(name="NMSE")
        self.SSIM = DistributedMetricSum(name="SSIM")
        self.PSNR = DistributedMetricSum(name="PSNR")
        self.ValLoss = DistributedMetricSum(name="ValLoss")
        self.TotExamples = DistributedMetricSum(name="TotExamples")

    def _create_data_loader(self, data_transform, data_partition, sample_rate=None):
        sample_rate = sample_rate or self.sample_rate
        dataset = SliceDataset(
            data_path=self.data_path / f"{self.challenge}_{data_partition}",
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=self.challenge,
            mode=data_partition
        )
        is_train = (data_partition == "train")
        sampler = None

        if self.trainer.accelerator_connector.use_ddp:
            if is_train:
                sampler = DistributedSampler(dataset)
            else:
                sampler = VolumeSampler(dataset, shuffle=False)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=is_train,
            sampler=sampler,
        )

        return dataloader

    def train_data_transform(self):
        maskfunc = create_mask_for_mask_type(self.mask_type, self.center_fractions, self.accelerations, )
        return transforms.DataTransform(self.challenge, maskfunc, use_seed=False)

    def val_data_transform(self):
        maskfunc = create_mask_for_mask_type(self.mask_type, self.center_fractions, self.accelerations, )
        return transforms.DataTransform(self.challenge, maskfunc, use_seed=True)

    def test_data_transform(self):
        maskfunc = create_mask_for_mask_type(self.mask_type, self.center_fractions, self.accelerations)
        return transforms.DataTransform(self.challenge, mask_func=maskfunc, use_seed=True)

    def train_dataloader(self):
        return self._create_data_loader(self.train_data_transform(), data_partition="train" )

    def val_dataloader(self):
        return self._create_data_loader(self.val_data_transform(), data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(self.test_data_transform(), data_partition='val')

    def validation_step_end(self, val_logs):
        device = val_logs["output"].device
        val_logs = {key: value.cpu() for key, value in val_logs.items()}
        val_logs["device"] = device

        return val_logs

    def validation_epoch_end(self, val_logs):
        device = val_logs[0]["device"]
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)

        for val_log in val_logs:
            losses.append(val_log["val_loss"])
            for i, (fname, slice_ind) in enumerate(zip(val_log["fname"], val_log["slice"])):
                if slice_ind not in [s for (s, _) in outputs[int(fname)]]:
                    outputs[int(fname)].append((int(slice_ind), val_log["output"][i]))
                    targets[int(fname)].append((int(slice_ind), val_log["target"][i]))

        metrics = dict(val_loss=0, nmse=0, ssim=0, psnr=0)
        for fname in outputs:
            output = torch.stack([out for _, out in sorted(outputs[fname])]).numpy()
            target = torch.stack([tgt for _, tgt in sorted(targets[fname])]).numpy()
            metrics["nmse"] = metrics["nmse"] + evaluate.nmse(target, output)
            metrics["ssim"] = metrics["ssim"] + evaluate.ssim(target, output)
            metrics["psnr"] = metrics["psnr"] + evaluate.psnr(target, output)

        metrics["nmse"] = self.NMSE(torch.tensor(metrics["nmse"]).to(device))
        metrics["ssim"] = self.SSIM(torch.tensor(metrics["ssim"]).to(device))
        metrics["psnr"] = self.PSNR(torch.tensor(metrics["psnr"]).to(device))
        metrics["val_loss"] = self.ValLoss(torch.sum(torch.stack(losses)).to(device))

        num_examples = torch.tensor(len(outputs)).to(device)
        tot_examples = self.TotExamples(num_examples)

        log_metrics = {f"metrics/{metric}": values / tot_examples for metric, values in metrics.items()}
        metrics = {metric: values / tot_examples for metric, values in metrics.items()}

        self.log_dict(log_metrics)

        return dict(log=log_metrics, **metrics)

    def test_epoch_end(self, test_logs, is_vis=False):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--mode', choices=['train', 'test'], default='train', type=str,)

        # data arguments
        parser.add_argument(
            "--data_path", type=pathlib.Path,
                default = pathlib.Path("/data/user/datasets/FastMRI/knee_singlecoil")
        )
        parser.add_argument(
            "--challenge",
            choices=["singlecoil", "multicoil"],
            default="singlecoil",
            type=str,
        )

        parser.add_argument(
            "--mask_type", choices=["random", "equispaced"], default="random", type=str
        )
        parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float)
        parser.add_argument("--accelerations", nargs="+", default=[4], type=int)

        parser.add_argument(
            "--sample_rate", default=0.01, type=float,
        )
        parser.add_argument(
            "--batch_size", default=4, type=int,
        )
        parser.add_argument(
            "--num_workers", default=1, type=float,
        )
        parser.add_argument(
            "--seed", default=42, type=int,
        )

        # logging params
        parser.add_argument(
            "--exp_dir", default=pathlib.Path(".."), type=pathlib.Path
        )
        parser.add_argument(
            "--exp_name", default="my_experiment", type=str,
        )
        parser.add_argument(
            "--test_split", default="test", type=str,
        )

        return parser

