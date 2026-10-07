#!/usr/bin/env python3
"""Update spacegroups.h5 with correct symmetry operations from spglib."""

import json

import h5py
import numpy as np


def main() -> None:
    from pymatgen.symmetry.groups import SpaceGroup

    output_path = "src/torchms/data/constants/spacegroups.h5"

    with h5py.File(output_path, "w") as h5:

        for int_number in range(1, 231):
            space_group = SpaceGroup.from_int_number(int_number)

            ops = np.array(
                list(map(lambda x: x.affine_matrix, space_group.symmetry_ops))
            )
            print(int_number, ops.shape)
            h5.create_dataset(
                str(int_number),
                data=ops,
                dtype=np.float32,
            )


if __name__ == "__main__":
    main()
