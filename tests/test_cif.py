"""Tests for CIF file parsing and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pycrystalgen.data import parse_cif
from pycrystalgen.data.batching import Structures
from pycrystalgen.data.constants import symbol_to_number

from pycrystalgen.data.compare import (
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


@pytest.mark.skip(reason="Only P1 structures are supported")
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

@pytest.mark.skip(reason="Only P1 structures are supported")
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
        from pycrystalgen.data.cif import CifParser

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
        from pycrystalgen.data.cif import CifParser

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

    @pytest.mark.skip(reason="Only P1 structures are supported")
    def test_custom_regex_multiple_overrides(self):
        """Test multiple regex overrides at once."""
        from pycrystalgen.data.cif import CifParser

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
        from pycrystalgen.data.cif import CifParser

        with pytest.raises(re.error):
            CifParser(regex_overrides={"band_gap": r"[invalid("})

    @pytest.mark.skip(reason="Only P1 structures are supported")
    def test_empty_override_uses_default(self, silicon_cif_fixture):
        """Test that empty overrides use default patterns."""
        from pycrystalgen.data.cif import CifParser

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
            "bad_occo",
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

    @pytest.mark.skip(reason="Only P1 structures are supported")
    def test_pymatgen_cif(self, cif_name: str):
        """Test parsing of pymatgen CIF files."""
        cif_path = Path(__file__).parent / "cif" / "pymatgen" / f"{cif_name}.cif"
        expected_cell, expected_x, expected_z = load_expected(cif_path)
        cif_string = cif_path.read_text()
        structures = parse_cif(cif_string)

        errors = compare_cell_parameters(
            expected_cell, compute_cell_parameters(structures.lattice[0])
        )
        assert not errors, "\n".join(errors)

        expected = Structures(structures.lattice, expected_x, expected_z)
        errors = compare_structures(structures, expected)
        assert not errors, "\n".join(errors)
