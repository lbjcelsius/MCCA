
import csv
import os
import random
import xml.etree.ElementTree as etree
from typing import Sequence, Tuple, Union
from warnings import warn
import pathlib
import h5py
import numpy as np
import yaml
from torch.utils.data import Dataset


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class SliceDataset(Dataset):
    def __init__(
            self,
            data_path,
            transform,
            challenge,
            sample_rate=1,
            mode='train',
    ):
        self.mode = mode

        #challenge
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.transform = transform
        self.examples = []
        self.data_path = data_path

        if self.mode == 'train':
            self.csv_file = os.path.join(self.data_path.parent, "singlecoil_train.csv")
        elif self.mode == 'val':
            self.csv_file = os.path.join(self.data_path.parent, "singlecoil_val.csv")

        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            file_names = list(reader)
            file_names = file_names[1:1+189] if mode == 'train' else file_names[1:1+40]
            file_names = [k[:2] for k in file_names]

        for idx, row in enumerate(file_names):
            pdfs = row[1]
            pd = row[0]
            pdfs = pdfs+'.h5' if not (pdfs.endswith('.h5')) else pdfs
            pd = pd+'.h5' if not (pd.endswith('.h5')) else pd
            pd_metadata, pd_num_slices = self._retrieve_metadata(os.path.join(self.data_path, pd))
            pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(os.path.join(self.data_path, pdfs))
            for slice_id in range(min(pd_num_slices, pdfs_num_slices)):
                self.examples.append((os.path.join(self.data_path, pd), os.path.join(self.data_path, pdfs), slice_id, pd_metadata, pdfs_metadata))

        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        pd_fname, pdfs_fname, slice, pd_metadata, pdfs_metadata = self.examples[i]
        with h5py.File(pd_fname, "r") as hf:
            pd_kspace = hf["kspace"][slice]
            pd_mask = np.asarray(hf["mask"]) if "mask" in hf else None
            pd_target = hf[self.recons_key][slice] if self.recons_key in hf else None
            attrs = dict(hf.attrs)
            attrs.update(pd_metadata)

        if self.transform is None:
            pd_sample = (pd_kspace, pd_mask, pd_target, attrs, pd_fname, slice)
        else:
            pd_sample = self.transform(pd_kspace, pd_mask, pd_target, attrs, pd_fname, slice)

        with h5py.File(pdfs_fname, "r") as hf:
            pdfs_kspace = hf["kspace"][slice]
            pdfs_mask = np.asarray(hf["mask"]) if "mask" in hf else None
            pdfs_target = hf[self.recons_key][slice] if self.recons_key in hf else None
            attrs = dict(hf.attrs)
            attrs.update(pdfs_metadata)

        if self.transform is None:
            pdfs_sample = (pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)
        else:
            pdfs_sample = self.transform(pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)

        return {'tar':pdfs_sample, 'aux':pd_sample}

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices
