import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

from .batching import Structures

import dataclasses
from collections.abc import Iterable
import os
from typing import Union


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
    def get_tensor_type(cls, annotated_type:str)->Union[torch.dtype,None]:
        types = {"LongTensor":np.int64,"FloatTensor":np.float32}

        for (annot, dtype) in types.items():
            if annot in annotated_type:
                return dtype

        return None

    @classmethod
    def from_structures(
        cls, hdf5_file: str | os.PathLike, structures: Structures | Iterable[Structures]
    ):
        if isinstance(structures, Structures):
            iterator = iter((structures,))
        else:
            iterator = iter(structures)

        first = next(iterator)

        data_file = h5py.File(hdf5_file, "w")

        datasets = {}
        for field in dataclasses.fields(Structures):
            if field.name.startswith("_"):
                continue

            dtype = cls.get_tensor_type(field.type)

            if dtype is None:
                continue

            tensor = getattr(first,field.name).cpu()

            dataset = data_file.create_dataset(field.name, tensor.shape, maxshape=(None,*tensor.shape[1:]), dtype=dtype)
            dataset[...]=getattr(first,field.name)

            offset = tensor.shape[0]
            datasets[field.name]=(dataset,offset)
        
        for struct in iterator:
            for name,(dataset,offset) in datasets.items():
                tensor=getattr(struct,name).cpu()

                new_size = offset+tensor.shape[0]
                dataset.resize(new_size,axis=0)
                dataset[offset:new_size,...]=tensor

                datasets[name]=(dataset,new_size)

        data_file.close()


