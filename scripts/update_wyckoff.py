"""Write the Wyckoff positions of the 230 space groups to wyckoff.npz.

The positions are the ones of pyxtal (MIT license), in its settings and its
order: for each space group, from the general position to the position "a",
and for each position its operations in the order pyxtal gives them. pyxtal
takes them from the Bilbao Crystallographic Server. torchms does not depend on
pyxtal: run this script once, outside of the environment of the project.

    uv run --isolated --no-project --python 3.12 --with pyxtal==0.6.7 --with "setuptools<81" \
        python scripts/update_wyckoff.py

The file is read by torchms.data.constants.get_wyckoff_positions. An
operation is the 3 x 4 matrix [R|t] of an affine map of fractional
coordinates: R is written as integers and t as integers over DENOMINATOR, which
is checked to be exact.
"""

import numpy as np

OUTPUT = "src/torchms/data/constants/wyckoff.npz"

# every translation of the table is a multiple of 1/24
DENOMINATOR = 24


def main() -> None:
    from pyxtal import __version__
    from pyxtal.symmetry import Group

    letters, multiplicities, operations = [], [], []
    num_positions = []
    for number in range(1, 231):
        positions = Group(number).Wyckoff_positions
        num_positions.append(len(positions))
        for position in positions:
            matrices = np.array([op.affine_matrix[:3] for op in position.ops])
            assert len(matrices) == position.multiplicity
            exact = np.round(matrices * DENOMINATOR)
            assert np.abs(exact - matrices * DENOMINATOR).max() < 1e-6
            rotations = exact[:, :, :3] / DENOMINATOR
            assert np.array_equal(rotations, np.round(rotations))

            letters.append(position.letter)
            multiplicities.append(position.multiplicity)
            operations.append(exact.astype(np.int16))

    np.savez_compressed(
        OUTPUT,
        num_positions=np.array(num_positions, dtype=np.int64),
        letters=np.array(letters),
        multiplicities=np.array(multiplicities, dtype=np.int64),
        operations=np.concatenate(operations),
        denominator=np.array(DENOMINATOR),
        pyxtal_version=np.array(__version__),
    )
    print(OUTPUT, len(letters), "positions", sum(multiplicities), "operations")


if __name__ == "__main__":
    main()
