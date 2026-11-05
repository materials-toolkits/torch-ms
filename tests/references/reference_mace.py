"""Reference energies of MACE, to check the rewrite of torchms.

Run once, outside of the environment of the project, which does not depend on
mace:

    uv run --isolated --no-project --python 3.11 --with mace-torch==0.3.16 \
        python tests/references/reference_mace.py

The structures are the ones of tests/data/orb_reference.npz. The result is
written to tests/data/mace_reference.npz and read by tests/test_mace.py.
"""

import hashlib
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from ase import Atoms
from mace.calculators import MACECalculator

DATA = Path(__file__).resolve().parent.parent / "data"
URL = (
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
    "2023-12-10-mace-128-L0_energy_epoch-249.model"
)
SHA256 = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"


def load_structures() -> list[Atoms]:
    data = np.load(DATA / "orb_reference.npz")
    ptr = np.concatenate([[0], np.cumsum(data["num_atoms"])])

    return [
        Atoms(
            numbers=data["z"][ptr[i] : ptr[i + 1]],
            scaled_positions=data["x"][ptr[i] : ptr[i + 1]].astype(np.float64),
            cell=data["lattice"][i].astype(np.float64),
            pbc=True,
        )
        for i in range(len(data["num_atoms"]))
    ]


def main():
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "mace-mp-0-small.model"
        urllib.request.urlretrieve(URL, source)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == SHA256

        calculator = MACECalculator(
            model_paths=str(source), device="cpu", default_dtype="float64"
        )

    energy, num_edges = [], []
    for atoms in load_structures():
        atoms.calc = calculator
        energy.append(atoms.get_potential_energy())
        batch = calculator._atoms_to_batch(atoms)  # pylint: disable=protected-access
        num_edges.append(batch["edge_index"].shape[1])

    np.savez(
        DATA / "mace_reference.npz",
        energy=np.array(energy, dtype=np.float64),
        num_edges=np.array(num_edges),
    )
    print("edges", num_edges)
    print("energy", np.array(energy).round(3))


if __name__ == "__main__":
    main()
