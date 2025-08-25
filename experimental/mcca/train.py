"""

The code is bulid on fastMRI https://github.com/facebookresearch/fastMRI

"""
import os
import sys
import time
import numpy as np
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from experimental.mcca.module_MCCA import MCCA_Module


def build_args():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    parent_parser = ArgumentParser(add_help=False)
    parser = MCCA_Module.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    num_gpus = 1
    backend = 'ddp' if num_gpus > 1 else None

    config = dict(
        chans=64,
        num_pool_layers=4,
        data_type="fastmriknee",
        mask_type="random",
        n_resgroups=6,
        n_resblocks=1,
        weight_decay=0.0,
        batch_size=2,
    )
    parser.set_defaults(**config)

    # trainer config
    parser.set_defaults(
        gpus=num_gpus,
        max_epochs=80,
        replace_sampler_ddp=(backend != "ddp"),
        accelerator=backend,
        seed=42,
        deterministic=True,
        checkpoint_path=None,
    )

    args = parser.parse_args()

    return args


def main():
    args = build_args()

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    seed_everything(args.seed)
    model = MCCA_Module(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    model_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=None,
        filename=None,
        verbose=True,
        monitor='metrics/psnr',
        mode='max',
        save_top_k=10,
    )

    trainer = Trainer.from_argparse_args(args, callbacks=model_ckpt)

    # ------------------------
    # 3 START TRAINING OR TEST
    # ------------------------
    if args.mode == "train":
        trainer.fit(model)
    elif args.mode == "test":
        model = model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, **vars(args))
        trainer.test(model)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


if __name__ == "__main__":
    main()
