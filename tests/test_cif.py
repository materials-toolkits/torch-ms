"""Tests for CIF file parsing and preprocessing."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from torchms.data import parse_cif
from torchms.data.batching import Structures
from torchms.data.constants import (
    get_spacegroup,
    get_symmetry_operations,
    symbol_to_number,
)

from torchms.data.compare import (
    compare_atoms,
    compare_structures,
    compute_cell_parameters,
    compare_atoms,
    compare_cell_parameters,
)


def load_expected(cif_path: Path) -> (dict[str, float], torch.Tensor, torch.Tensor):
    """Load expected values from JSON file with same name as CIF."""
    json_path = cif_path.with_suffix(".json")
    with open(json_path) as f:
        data = json.load(f)

    coords = torch.tensor([atom["coords"] for atom in data["atoms"]])
    atomic_numbers = torch.tensor(
        [symbol_to_number(atom["element"]) for atom in data["atoms"]]
    )

    return (data["cell_parameters"], coords, atomic_numbers)


@pytest.fixture(scope="session")
def triclinic1_cif() -> Path:
    """Path to triclinic CeP5O14 CIF file (P1, space group 1)."""
    return Path(__file__).parent / "cif" / "triclinic1.cif"


@pytest.fixture(scope="session")
def triclinic2_cif() -> Path:
    """Path to triclinic AlCl3 CIF file (P1, space group 1)."""
    return Path(__file__).parent / "cif" / "triclinic2.cif"


@pytest.fixture(scope="session")
def triclinic3_cif() -> Path:
    """Path to triclinic Np3F12·H2O CIF file (P1, space group 1)."""
    return Path(__file__).parent / "cif" / "triclinic3.cif"


class TestTriclinic:
    def test_triclinic1(self, triclinic1_cif):
        expected_cell, expected_x, expected_z = load_expected(triclinic1_cif)
        cif_string = triclinic1_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)

    def test_triclinic2(self, triclinic2_cif):
        expected_cell, expected_x, expected_z = load_expected(triclinic2_cif)
        cif_string = triclinic2_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)

    def test_triclinic3(self, triclinic3_cif):
        expected_cell, expected_x, expected_z = load_expected(triclinic3_cif)
        cif_string = triclinic3_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)


class TestMultipleFiles:
    def test_parse_multiple_cif_files(
        self, triclinic1_cif, triclinic2_cif, triclinic3_cif
    ):
        import random

        expected = [
            load_expected(triclinic1_cif),
            load_expected(triclinic2_cif),
            load_expected(triclinic3_cif),
        ]
        cif = [
            triclinic1_cif.read_text(),
            triclinic2_cif.read_text(),
            triclinic3_cif.read_text(),
        ]
        test_cases = [*zip(cif, expected)]

        cif, expected = zip(*[random.choice(test_cases) for _ in range(10)])
        cif = "".join(cif)

        structures = parse_cif(cif)
        separated = Structures.separate(structures)

        for struct, expect in zip(separated, expected):
            expected_cell, expected_x, expected_z = expect

            errors = compare_cell_parameters(
                expected_cell, compute_cell_parameters(struct.lattice[0])
            )
            assert not errors, "\n".join(errors)

            expected = Structures(struct.lattice, expected_x, expected_z)

            errors = compare_structures(struct, expected)
            assert not errors, "\n".join(errors)


class TestNonP1Symmetry:
    """Tests for CIF files with non-P1 spacegroups (symmetry expansion)."""

    @pytest.fixture(scope="class")
    def halite_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "halite.cif"

    @pytest.fixture(scope="class")
    def perovskite_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "perovskite.cif"

    @pytest.fixture(scope="class")
    def diamond_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "diamond.cif"

    @pytest.fixture(scope="class")
    def silicon_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "silicon.cif"

    @pytest.fixture(scope="class")
    def hexagonal_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "hexagonal_mp_542933.cif"

    @pytest.fixture(scope="class")
    def monoclinic_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "monoclinic_mp_1101139.cif"

    @pytest.fixture(scope="class")
    def orthorhombic_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "orthorhombic_mp_644325.cif"

    @pytest.fixture(scope="class")
    def tetragonal_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "tetragonal_mp_1207961.cif"

    @pytest.fixture(scope="class")
    def trigonal_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "trigonal_mp_1223466.cif"

    def test_perovskite(self, perovskite_cif):
        """Compare parsed halite structure with expected."""
        expected_cell, expected_x, expected_z = load_expected(perovskite_cif)
        cif_string = perovskite_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)

    def test_halite(self, halite_cif):
        """Compare parsed halite structure with expected."""
        expected_cell, expected_x, expected_z = load_expected(halite_cif)
        cif_string = halite_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)

    def test_diamond(self, diamond_cif):
        """Compare parsed diamond structure with expected."""
        expected_cell, expected_x, expected_z = load_expected(diamond_cif)
        cif_string = diamond_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)

    def test_cubic(self, silicon_cif):
        """Compare parsed silicon structure with expected."""
        expected_cell, expected_x, expected_z = load_expected(silicon_cif)
        cif_string = silicon_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)

    def test_hexagonal(self, hexagonal_cif):
        """Compare parsed silicon structure with expected."""
        expected_cell, expected_x, expected_z = load_expected(hexagonal_cif)
        cif_string = hexagonal_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)

    def test_monoclinic(self, monoclinic_cif):
        """Compare parsed silicon structure with expected."""
        expected_cell, expected_x, expected_z = load_expected(monoclinic_cif)
        cif_string = monoclinic_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)

    def test_orthorhombic(self, orthorhombic_cif):
        """Compare parsed silicon structure with expected."""
        expected_cell, expected_x, expected_z = load_expected(orthorhombic_cif)
        cif_string = orthorhombic_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)

    def test_tetragonal(self, tetragonal_cif):
        """Compare parsed silicon structure with expected."""
        expected_cell, expected_x, expected_z = load_expected(tetragonal_cif)
        cif_string = tetragonal_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)

    def test_trigonal(self, trigonal_cif):
        """Compare parsed silicon structure with expected."""
        expected_cell, expected_x, expected_z = load_expected(trigonal_cif)
        cif_string = trigonal_cif.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)

        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)


class TestProperties:
    """Tests for energy and band gap parsing in CIF files."""

    @pytest.fixture(scope="class")
    def silicon_with_props_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "silicon_with_properties.cif"

    @pytest.fixture(scope="class")
    def silicon_cif(self) -> Path:
        return Path(__file__).parent / "cif" / "silicon.cif"

    def test_parse_energy_and_band_gap(self, silicon_with_props_cif):
        """Test parsing of energy and band gap when present in CIF."""
        cif_string = silicon_with_props_cif.read_text()
        structures = parse_cif(cif_string)

        assert structures.total_energy is not None
        assert structures.energy_above_hull is not None
        assert structures.band_gap is not None

        assert torch.isclose(structures.total_energy[0], torch.tensor(-10.5))
        assert torch.isclose(structures.energy_above_hull[0], torch.tensor(0.0))
        assert torch.isclose(structures.band_gap[0], torch.tensor(1.1))

    def test_nan_fallback_missing_properties(self, silicon_cif):
        """Test that NaN is used when properties are missing from CIF."""
        cif_string = silicon_cif.read_text()
        structures = parse_cif(cif_string)

        assert structures.total_energy is not None
        assert structures.energy_above_hull is not None
        assert structures.band_gap is not None

        assert torch.isnan(structures.total_energy).all()
        assert torch.isnan(structures.energy_above_hull).all()
        assert torch.isnan(structures.band_gap).all()


class TestRegexOverrides:
    """Tests for custom regex override functionality in CifParser."""

    def test_custom_regex_total_energy(self):
        """Test overriding total_energy pattern with custom regex."""
        from torchms.data.cif import CifParser

        cif_content = """
data_test
_space_group_IT_number 1
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_cell_length_a 5.43
_cell_length_b 5.43
_cell_length_c 5.43
_my_custom_energy -15.5
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_type_symbol
Si 0.0 0.0 0.0 Si
"""
        parser = CifParser(
            regex_overrides={
                "total_energy": r"_my_custom_energy\s+([\d.-]+)",
            }
        )
        structures = parser.parse(cif_content)
        assert torch.isclose(structures.total_energy[0], torch.tensor(-15.5))

    def test_custom_regex_band_gap(self):
        """Test overriding band_gap pattern with custom regex."""
        from torchms.data.cif import CifParser

        cif_content = """
data_test
_space_group_IT_number 1
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_cell_length_a 5.43
_cell_length_b 5.43
_cell_length_c 5.43
_bandgap_value 2.5
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_type_symbol
Si 0.0 0.0 0.0 Si
"""
        parser = CifParser(
            regex_overrides={
                "band_gap": r"_bandgap_value\s+([\d.-]+)",
            }
        )
        structures = parser.parse(cif_content)
        assert torch.isclose(structures.band_gap[0], torch.tensor(2.5))

    def test_custom_regex_multiple_overrides(self):
        """Test multiple regex overrides at once."""
        from torchms.data.cif import CifParser

        cif_content = """
data_test
_spgrp_num 227
_cell_length_a 3.5
_cell_length_b 3.5
_cell_length_c 3.5
_angle_alpha 90.0
_angle_beta 90.0
_angle_gamma 90.0
_my_energy -20.0
_my_hull 0.5
_my_gap 1.2
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_type_symbol
Si 0.0 0.0 0.0 Si
"""
        parser = CifParser(
            regex_overrides={
                "space_group": r"_spgrp_num\s+(\d+)",
                "cell_angle": r"_angle_(alpha|beta|gamma)\s+([\d.]+)",
                "total_energy": r"_my_energy\s+([\d.-]+)",
                "energy_above_hull": r"_my_hull\s+([\d.-]+)",
                "band_gap": r"_my_gap\s+([\d.-]+)",
            }
        )
        structures = parser.parse(cif_content)
        assert torch.isclose(structures.total_energy[0], torch.tensor(-20.0))
        assert torch.isclose(structures.energy_above_hull[0], torch.tensor(0.5))
        assert torch.isclose(structures.band_gap[0], torch.tensor(1.2))

    def test_invalid_regex_raises_error(self):
        """Test that invalid regex raises re.error."""
        import re
        from torchms.data.cif import CifParser

        with pytest.raises(re.error):
            CifParser(regex_overrides={"band_gap": r"[invalid("})

    def test_empty_override_uses_default(self, silicon_cif_fixture):
        """Test that empty overrides use default patterns."""
        from torchms.data.cif import CifParser

        cif_string = silicon_cif_fixture.read_text()
        parser = CifParser(regex_overrides={})
        structures = parser.parse(cif_string)
        assert torch.isnan(structures.total_energy).all()

    @pytest.fixture(scope="class")
    def silicon_cif_fixture(self) -> Path:
        return Path(__file__).parent / "cif" / "silicon.cif"


class TestPymatgen:
    @pytest.mark.parametrize(
        "cif_name",
        [
            "AgO_kpath_test",
            "bad_occu",
            "bcc_1927",
            "btet_1915",
            "C26H16BeN2O2S2",
            "cif_finite_precision_frac_coord_error",
            "cif_implicit_hydrogens_cod_1011130",
            "Co8",
            "CoO19128",
            "CsI3Pb",
            "CsSnI3",
            "Cu7Te5",
            "CuCl",
            "Fe",
            "Fe3O4",
            "FePO4a",
            "garnet",
            "Graphite",
            "H6PbCI3N_mp-977013_symmetrized",
            "hex_1170",
            "La2CoO4",
            "La4Fe4O12",
            "Li",
            "Li10GeP2S12",
            "Li2O",
            "Li4Fe3Mn1(PO4)4",
            "Li8Fe2NiCoO8",
            "LiFePO4",
            "MgNiF6",
            "monoc_1028",
            "MultiStructure",
            "orac_632475",
            "orcc_1003",
            "orci_1010",
            "P24Ru4H252C296S24N16",
            "PF_sd_1002871",
            "PF_sd_1011081",
            "PF_sd_1500382",
            "PF_sd_1601634",
            "PF_sd_1615854",
            "PF_sd_1622133",
            "PF_sd_1704003",
            "PF_sd_1811457",
            "PF_sd_1908491",
            "PF_sd_1928405",
            "rhomb_1170",
            "rhomb_3478",
            "rhomb_3478_conv",
            "site_type_symbol_test",
            "V2O3",
        ],
    )
    def test_pymatgen_cif(self, cif_name: str):
        """Test parsing of pymatgen CIF files."""
        cif_path = Path(__file__).parent / "cif" / "pymatgen" / f"{cif_name}.cif"
        expected_cell, expected_x, expected_z = load_expected(cif_path)
        cif_string = cif_path.read_text()
        # the expected structure is the one of ase.io.read: the last block
        parsed = parse_cif(cif_string)
        structures = parsed.select(torch.tensor([len(parsed.num_atoms) - 1]))

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)
        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)


class TestFormationEnergy:
    """Tests for the formation energy tag, on a P1 structure."""

    @pytest.fixture(scope="class")
    def triclinic_cif(self) -> str:
        return (Path(__file__).parent / "cif" / "triclinic1.cif").read_text()

    @staticmethod
    def with_tags(cif: str, tags: str) -> str:
        cif, count = re.subn(r"^(data_\S+)$", rf"\1\n{tags}", cif, flags=re.MULTILINE)
        assert count == 1
        return cif

    def test_parse_formation_energy(self, triclinic_cif):
        cif = self.with_tags(
            triclinic_cif,
            "_formation_energy_per_atom -1.25\n_energy_above_hull 0.03",
        )
        structures = parse_cif(cif)

        assert torch.isclose(
            structures.formation_energy_per_atom[0], torch.tensor(-1.25)
        )
        assert torch.isclose(structures.energy_above_hull[0], torch.tensor(0.03))
        assert torch.isnan(structures.total_energy).all()

    def test_formation_energy_is_not_energy_above_hull(self, triclinic_cif):
        cif = self.with_tags(triclinic_cif, "_formation_energy -1.25")
        structures = parse_cif(cif)

        assert torch.isnan(structures.energy_above_hull).all()

    def test_nan_when_missing(self, triclinic_cif):
        structures = parse_cif(triclinic_cif)

        assert torch.isnan(structures.formation_energy_per_atom).all()


class TestCifDatasetLoader:
    """A DataLoader batches the structures of CIF files, and their graphs."""

    @staticmethod
    def make_dataset(tmp_path, transform=None):
        import shutil
        from pathlib import Path

        from torchms.data import CifDataset

        # the files in P1, the only ones read
        for name in ("triclinic1", "triclinic2", "triclinic3"):
            shutil.copy(Path("tests/cif") / f"{name}.cif", tmp_path / f"{name}.cif")

        return CifDataset(tmp_path, transform=transform)

    def test_batches_of_structures(self, tmp_path):
        from torch.utils.data import DataLoader

        from torchms.data import Structures

        dataset = self.make_dataset(tmp_path)
        batches = list(DataLoader(dataset, batch_size=2))

        assert [batch.num_atoms.tolist() for batch in batches] == [[40, 4], [32]]
        expected = Structures.collate([dataset[i] for i in range(2)])
        assert torch.equal(batches[0].x, expected.x)
        assert torch.equal(batches[0].ptr, expected.ptr)

    def test_graphs_built_per_structure_are_batched(self, tmp_path):
        from torch.utils.data import DataLoader

        from torchms.data import Structures
        from torchms.data.augmentation import Compose
        from torchms.graph.transforms import AddRadiusGraph, AddTriplets

        chain = Compose([AddRadiusGraph(3.2, geometry=True), AddTriplets(3.0, True)])
        plain = self.make_dataset(tmp_path)
        expected = chain(Structures.collate([plain[i] for i in range(len(plain))]))

        dataset = self.make_dataset(tmp_path, transform=chain)
        (batch,) = list(DataLoader(dataset, batch_size=4))

        assert torch.equal(batch.graph.num_edges, expected.graph.num_edges)
        assert torch.equal(batch.graph.edge_ptr, expected.graph.edge_ptr)
        assert torch.equal(batch.triplets.num_tuples, expected.triplets.num_tuples)
        # the indices of the batch point to the right atoms and edges
        assert torch.allclose(
            batch.graph.edge_distance, batch.graph.edges_distances(), atol=1e-5
        )
        assert torch.allclose(
            batch.graph.edge_distance.sort().values,
            expected.graph.edge_distance.sort().values,
            atol=1e-5,
        )
        assert torch.allclose(
            batch.triplets.angle.sort().values,
            expected.triplets.angle.sort().values,
            atol=1e-4,
        )
        sender = batch.graph.edge_index[0, batch.triplets.index]
        assert (sender[0] == sender[1]).all()


class TestSymmetry:
    """Symmetry operations, loops and mixed sites of a CIF block."""

    HEADER = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
"""

    @pytest.mark.parametrize(
        "operation, rotation, translation",
        [
            ("x, y, z", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0]),
            ("-x+1/2,y,-z", [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0.5, 0, 0]),
            ("x-y, x, z+1/2", [[1, -1, 0], [1, 0, 0], [0, 0, 1]], [0, 0, 0.5]),
            ("1/4+Y, 0.75-X, -Z", [[0, 1, 0], [-1, 0, 0], [0, 0, -1]], [0.25, 0.75, 0]),
        ],
    )
    def test_symmetry_operation(self, operation, rotation, translation):
        from torchms.data.cif import parse_symmetry_operation

        matrix = parse_symmetry_operation(operation)
        assert matrix[:3, :3].tolist() == rotation
        assert matrix[:3, 3].tolist() == translation
        assert matrix[3].tolist() == [0, 0, 0, 1]

    @pytest.mark.parametrize("operation", ["x, y", "x, y, w", "x, y, z+a"])
    def test_symmetry_operation_errors(self, operation):
        from torchms.data.cif import parse_symmetry_operation

        with pytest.raises(ValueError):
            parse_symmetry_operation(operation)

    def test_listed_operations_win_over_the_number(self):
        """The operations of the block are its setting: a shifted inversion here."""
        cif = self.HEADER + """_symmetry_Int_Tables_number 2
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'-x+1/2, -y, -z'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 0.1 0.2 0.3
"""
        structures = parse_cif(cif)
        assert torch.allclose(
            structures.x.double(),
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.8, 0.7]], dtype=torch.float64),
            atol=1e-6,
        )

    def test_images_merge_across_the_cell(self):
        """0.0 and 0.9999 are the same coordinate."""
        cif = self.HEADER + """loop_
_space_group_symop_operation_xyz
'x, y, z'
'-x, -y, -z'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 0.0001 0.5 0.5
"""
        assert parse_cif(cif).num_atoms.tolist() == [1]

    def test_a_loop_ends_at_the_next_loop(self):
        """The anisotropic loop after the atom sites is not read as atoms."""
        cif = self.HEADER + """_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
Na1 0.0 0.0 0.0 0.01(2)
Cl1 0.5 0.5 0.5 0.02
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
Na1 0.03009(17)
"""
        structures = parse_cif(cif)
        assert structures.z.tolist() == [11, 17]

    def test_mixed_site_is_its_most_occupied_element(self):
        cif = self.HEADER + """_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_occupancy
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
M1 Ge 0.3 0.5 0.5 0.5
M1 Sb 0.7 0.5 0.5 0.5
M2 '0.8Nb + 0.2Zr' 1 0.0 0.0 0.0
"""
        assert parse_cif(cif).z.tolist() == [51, 41]

    def test_blocks_without_structure_are_left_out(self):
        cif = (
            "data_global\n_journal_year 2014\n\n"
            + self.HEADER
            + """_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 0.0 0.0 0.0
"""
        )
        assert parse_cif(cif).num_atoms.tolist() == [1]
        with pytest.raises(ValueError, match="No structure"):
            parse_cif("data_global\n_journal_year 2014\n")


# cell lengths and angles of each crystal system, for the check of every group
CELLS = {
    "triclinic": (7.0, 8.0, 9.0, 80.0, 85.0, 95.0),
    "monoclinic": (7.0, 8.0, 9.0, 90.0, 100.0, 90.0),
    "orthorhombic": (7.0, 8.0, 9.0, 90.0, 90.0, 90.0),
    "tetragonal": (7.0, 7.0, 9.0, 90.0, 90.0, 90.0),
    "trigonal": (7.0, 7.0, 9.0, 90.0, 90.0, 120.0),
    "hexagonal": (7.0, 7.0, 9.0, 90.0, 90.0, 120.0),
    "cubic": (7.0, 7.0, 7.0, 90.0, 90.0, 90.0),
}


def operation_string(operation: torch.Tensor) -> str:
    """A symmetry operation of CIF from its affine matrix: "-x+y,-x,z+1/3"."""
    rows = []
    for rotation, translation in zip(operation[:3, :3].tolist(), operation[:3, 3]):
        row = "".join(
            f"{'+' if c > 0 else '-'}{axis}"
            for c, axis in zip(rotation, "xyz")
            if round(c) != 0
        )
        # translations of the space groups are multiples of 1/12
        twelfths = round(float(translation) % 1.0 * 12) % 12
        if twelfths:
            row += f"+{twelfths}/12"
        rows.append(row.lstrip("+"))

    return ",".join(rows)


@pytest.mark.parametrize("number", range(1, 231))
def test_every_space_group(number):
    """A general position has one image per operation, from the number or listed."""
    group = get_spacegroup(number)
    a, b, c, alpha, beta, gamma = CELLS[group.crystal_system]
    header = f"""data_group
_cell_length_a {a}
_cell_length_b {b}
_cell_length_c {c}
_cell_angle_alpha {alpha}
_cell_angle_beta {beta}
_cell_angle_gamma {gamma}
"""
    # a general position: on no symmetry element of any group, no coordinate a
    # multiple or a sum of another (y = 2x is on a mirror of hexagonal groups)
    sites = """loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.1234 0.3571 0.4189
"""
    operations = get_symmetry_operations(number)
    listed = "loop_\n_space_group_symop_operation_xyz\n" + "".join(
        f"'{operation_string(op)}'\n" for op in operations
    )

    by_number = parse_cif(header + f"_space_group_IT_number {number}\n" + sites)
    by_operations = parse_cif(header + listed + sites)

    assert by_number.num_atoms.tolist() == [len(operations)]
    assert by_operations.num_atoms.tolist() == [len(operations)]
    errors = compare_structures(by_operations, by_number)
    assert not errors, "\n".join(errors)
