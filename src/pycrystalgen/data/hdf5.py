import torch
from torch.utils.data import Dataset, DataLoader

from collections.abc import Iterable
import os


class HDF5Dataset(Dataset):

    def __init__(self, hdf5_file: str | os.PathLike):
        self.data_file = h5py.File(hdf5_file, "r")

    def __len__(self):
        # TODO: implement
        pass

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # TODO: implement

        if self.transform:
            sample = self.transform(sample)

        return sample

    @classmethod
    def from_structures(
        cls, hdf5_file: std | os.PathLike, structures: Iterable[Structures]
    ):
        # TODO: implement
        pass
