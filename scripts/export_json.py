import glob
import json
import os

from ase.io import read

def export_json(filename):

    structure = read(filename)

    cell_parameters = {
        p: v
        for p, v in zip(
            ["a", "b", "c", "alpha", "beta", "gamma"], structure.cell.cellpar()
        )
    }
    num_atoms = len(structure)
    atoms = []
    for x, ch in zip(structure.get_scaled_positions(), structure.symbols):
        atoms.append(
            {
                "coords": x.tolist(),
                "element": ch,
            }
        )

    structure = {
        "cell_parameters": cell_parameters,
        "num_atoms": num_atoms,
        "atoms": atoms,
    }

    with open(os.path.splitext(filename)[0] + ".json", "w") as fp:
        json.dump(structure, fp, indent=4)

for filename in glob.glob("tests/cif/*.cif"):
    export_json(filename)

for filename in glob.glob("tests/cif/pymatgen/*.cif"):
    try:
        export_json(filename)
    except Exception as e:
        print(e)
        print("file:",filename)

