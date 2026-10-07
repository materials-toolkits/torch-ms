import pytest
import torch
import numpy as np
from unittest.mock import patch

from torchms.data import Structures
from torchms.data.batching import InvalidStructure


@pytest.fixture
def single_structure_fields():
    return {
        "lattice": torch.randn(1, 3, 3),
        "x": torch.randn(4, 3),
        "z": torch.randint(1, 100, (4,)),
    }


@pytest.fixture
def multiple_structures_fields():
    return {
        "lattice": torch.randn(2, 3, 3),
        "x": torch.randn(4, 3),
        "z": torch.randint(1, 100, (4,)),
        "num_atoms": torch.tensor([2, 2]),
    }


class TestStructuresInitialization:
    @patch.object(Structures, "check_tensors")
    def test_init_with_required_fields_single_structure(
        self, mock_check, single_structure_fields
    ):
        struct = Structures(**single_structure_fields)

        mock_check.assert_called_once()
        assert struct.lattice.shape == (1, 3, 3)
        assert struct.x.shape == (4, 3)
        assert struct.z.shape == (4,)
        assert struct._validate is True

    @patch.object(Structures, "check_tensors")
    def test_init_with_required_fields_multiple_structures(
        self, mock_check, multiple_structures_fields
    ):
        struct = Structures(**multiple_structures_fields)

        mock_check.assert_called_once()
        assert struct.lattice.shape == (2, 3, 3)
        assert struct.x.shape == (4, 3)
        assert struct.z.shape == (4,)
        assert struct._validate is True

    @patch.object(Structures, "check_tensors")
    def test_init_with_all_fields(self, mock_check, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields.update(
            {
                "batch": torch.tensor([0, 0, 1, 1]),
                "total_energy": torch.tensor([-5.0, -3.0]),
                "energy_above_hull": torch.tensor([0.1, 0.2]),
                "band_gap": torch.tensor([1.5, 2.0]),
            }
        )
        struct = Structures(**fields)

        mock_check.assert_called_once()
        assert torch.equal(struct.num_atoms, torch.tensor([2, 2]))
        assert torch.equal(struct.batch, torch.tensor([0, 0, 1, 1]))
        assert torch.equal(struct.total_energy, torch.tensor([-5.0, -3.0]))
        assert torch.equal(struct.energy_above_hull, torch.tensor([0.1, 0.2]))
        assert torch.equal(struct.band_gap, torch.tensor([1.5, 2.0]))

    @patch.object(Structures, "check_tensors")
    def test_init_validate_flag_skips_validation(
        self, mock_check, single_structure_fields
    ):
        fields = single_structure_fields.copy()
        fields["_validate"] = True

        struct = Structures(**fields)

        mock_check.assert_not_called()
        assert struct._validate is True


class TestStructuresDefaults:
    def test_default_batch_with_equal_atoms(self, multiple_structures_fields):
        struct = Structures(**multiple_structures_fields)

        assert struct.batch is not None
        assert struct.batch.shape == (4,)
        assert torch.equal(struct.batch, torch.tensor([0, 0, 1, 1]))

    def test_default_batch_with_varying_atoms(self):
        struct = Structures(
            lattice=torch.randn(2, 3, 3),
            x=torch.randn(4, 3),
            z=torch.randint(1, 100, (4,)),
            num_atoms=torch.tensor([1, 3]),
        )

        assert struct.batch is not None
        assert struct.batch.shape == (4,)
        assert torch.equal(struct.batch, torch.tensor([0, 1, 1, 1]))

    def test_default_ptr_single_structure(self):
        struct = Structures(
            lattice=torch.randn(1, 3, 3),
            x=torch.randn(3, 3),
            z=torch.randint(1, 100, (3,)),
        )

        assert struct.ptr is not None
        assert struct.ptr.shape == (2,)
        assert torch.equal(struct.ptr, torch.tensor([0, 3]))

    def test_default_ptr_multiple_structures_equal_atoms(
        self, multiple_structures_fields
    ):
        struct = Structures(**multiple_structures_fields)

        assert struct.ptr is not None
        assert struct.ptr.shape == (3,)
        assert torch.equal(struct.ptr, torch.tensor([0, 2, 4]))

    def test_default_ptr_multiple_structures_varying_atoms(self):
        struct = Structures(
            lattice=torch.randn(2, 3, 3),
            x=torch.randn(4, 3),
            z=torch.randint(1, 100, (4,)),
            num_atoms=torch.tensor([1, 3]),
        )

        assert struct.ptr is not None
        assert torch.equal(struct.ptr, torch.tensor([0, 1, 4]))

    def test_default_energy_fields_nan(self, multiple_structures_fields):
        struct = Structures(**multiple_structures_fields)

        assert struct.total_energy is not None
        assert struct.total_energy.shape == (2,)
        assert torch.isnan(struct.total_energy).all()

        assert struct.energy_above_hull is not None
        assert struct.energy_above_hull.shape == (2,)
        assert torch.isnan(struct.energy_above_hull).all()

        assert struct.band_gap is not None
        assert struct.band_gap.shape == (2,)
        assert torch.isnan(struct.band_gap).all()


class TestStructuresValidation:
    def test_multiple_structures_without_num_atoms_raises(self):
        with pytest.raises(InvalidStructure, match="num_atoms"):
            Structures(
                lattice=torch.randn(2, 3, 3),
                x=torch.randn(4, 3),
                z=torch.randint(1, 100, (4,)),
            )

    def test_invalid_lattice_shape(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["lattice"] = torch.randn(2, 3)

        with pytest.raises(InvalidStructure, match="invalid shape"):
            Structures(**fields)

    def test_invalid_x_shape(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["x"] = torch.randn(4, 2)

        with pytest.raises(InvalidStructure, match="invalid shape"):
            Structures(**fields)

    def test_invalid_z_shape(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["z"] = torch.randint(1, 100, (3,))

        with pytest.raises(InvalidStructure, match="invalid shape"):
            Structures(**fields)

    def test_inconsistent_num_atoms_shape(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["num_atoms"] = torch.tensor([2, 2, 2])

        with pytest.raises(InvalidStructure, match="invalid shape"):
            Structures(**fields)

    def test_inconsistent_batch_shape(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["batch"] = torch.tensor([0, 1])

        with pytest.raises(InvalidStructure, match="invalid shape"):
            Structures(**fields)

    def test_inconsistent_ptr_shape(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["ptr"] = torch.tensor([0, 4])

        with pytest.raises(InvalidStructure, match="invalid shape"):
            Structures(**fields)

    def test_batch_not_ascending_raises(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["batch"] = torch.tensor([0, 1, 0, 1])

        with pytest.raises(InvalidStructure, match="ascending order"):
            Structures(**fields)

    def test_ptr_not_ascending_raises(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["ptr"] = torch.tensor([0, 4, 2])

        with pytest.raises(InvalidStructure, match="ascending order"):
            Structures(**fields)

    def test_inconsistent_total_energy_shape(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["total_energy"] = torch.tensor([1.0, 2.0, 3.0])

        with pytest.raises(InvalidStructure, match="invalid shape"):
            Structures(**fields)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_inconsistent_device(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["x"] = torch.randn(4, 3).cuda()

        with pytest.raises(InvalidStructure, match="inconsistent device"):
            Structures(**fields)


class TestStructuresConversion:
    def test_lists_to_tensors(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["lattice"] = fields["lattice"].tolist()
        fields["x"] = fields["x"].tolist()
        fields["z"] = fields["z"].tolist()

        struct = Structures(**fields)

        assert isinstance(struct.lattice, torch.Tensor)
        assert isinstance(struct.x, torch.Tensor)
        assert isinstance(struct.z, torch.Tensor)
        assert struct.lattice.shape == (2, 3, 3)
        assert struct.x.shape == (4, 3)
        assert struct.z.shape == (4,)

    def test_numpy_to_tensors(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["lattice"] = fields["lattice"].numpy()
        fields["x"] = fields["x"].numpy()
        fields["z"] = fields["z"].numpy().astype(np.int64)

        struct = Structures(**fields)

        assert isinstance(struct.lattice, torch.Tensor)
        assert isinstance(struct.x, torch.Tensor)
        assert isinstance(struct.z, torch.Tensor)

    def test_dtype_conversion(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["lattice"] = torch.randn(2, 3, 3).double()
        fields["x"] = torch.randn(4, 3).double()
        fields["z"] = torch.randint(1, 100, (4,))

        struct = Structures(**fields)

        assert struct.lattice.dtype == torch.float32
        assert struct.x.dtype == torch.float32
        assert struct.z.dtype == torch.int64


class TestStructuresDevice:
    def test_device_cpu(self, multiple_structures_fields):
        struct = Structures(**multiple_structures_fields)

        assert struct.device() == torch.device("cpu")
        assert struct.lattice.device == torch.device("cpu")
        assert struct.x.device == torch.device("cpu")
        assert struct.z.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["lattice"] = torch.randn(2, 3, 3).cuda()
        fields["x"] = torch.randn(4, 3).cuda()
        fields["z"] = torch.randint(1, 100, (4,)).cuda()
        fields["num_atoms"] = fields["num_atoms"].cuda()

        struct = Structures(**fields)

        assert struct.device() == torch.device("cuda:0")
        assert struct.num_atoms.device == torch.device("cuda:0")
        assert struct.lattice.device == torch.device("cuda:0")
        assert struct.x.device == torch.device("cuda:0")
        assert struct.z.device == torch.device("cuda:0")

    @staticmethod
    def tensors(struct: Structures) -> dict[str, torch.Tensor]:
        return {
            name: value
            for name, value in vars(struct).items()
            if isinstance(value, torch.Tensor)
        }

    @staticmethod
    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        """Same values, unknown labels (NaN) included."""
        return a.dtype == b.dtype and torch.allclose(
            a.double(), b.double(), rtol=0, atol=0, equal_nan=True
        )

    def test_to_moves_every_tensor(self, two_structures_with_energy):
        struct = Structures.collate(two_structures_with_energy)

        # the "meta" device has no data: it tells where a tensor is, without a GPU
        moved = struct.to("meta")

        assert moved is not struct
        assert moved.device() == torch.device("meta")
        assert len(self.tensors(moved)) == 10
        for name, tensor in self.tensors(moved).items():
            original = getattr(struct, name)
            assert tensor.device == torch.device("meta"), name
            assert tensor.shape == original.shape and tensor.dtype == original.dtype

        # the original is not moved
        assert struct.device() == torch.device("cpu")
        assert all(
            t.device == torch.device("cpu") for t in self.tensors(struct).values()
        )

    def test_to_same_device(self, two_structures_with_energy):
        struct = Structures.collate(two_structures_with_energy)
        moved = struct.to("cpu")

        assert moved is not struct
        for name, tensor in self.tensors(moved).items():
            assert self.same(tensor, getattr(struct, name)), name
        moved.check_tensors()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda_and_back(self, two_structures_with_energy):
        struct = Structures.collate(two_structures_with_energy)
        moved = struct.to("cuda")

        assert moved.device() == torch.device("cuda:0")
        assert all(
            t.device == torch.device("cuda:0") for t in self.tensors(moved).values()
        )
        moved.check_tensors()

        back = moved.to("cpu")
        for name, tensor in self.tensors(back).items():
            assert self.same(tensor, getattr(struct, name)), name


class TestStructuresBatching:
    def test_contiguous_batching(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["batch"] = torch.tensor([0, 0, 1, 1])

        struct = Structures(**fields)

        assert torch.equal(struct.batch, torch.tensor([0, 0, 1, 1]))
        assert torch.equal(struct.ptr, torch.tensor([0, 2, 4]))


class TestStructuresEdgeCases:
    def test_zero_atoms_single_structure(self):
        with pytest.raises(
            InvalidStructure, match="must contain at least one atom per crystal"
        ):
            Structures(
                lattice=torch.randn(1, 3, 3),
                x=torch.empty(0, 3),
                z=torch.tensor([], dtype=torch.long),
            )

    def test_mixed_zero_nonzero_atoms(self):
        with pytest.raises(
            InvalidStructure, match="must contain at least one atom per crystal"
        ):
            Structures(
                lattice=torch.randn(2, 3, 3),
                x=torch.randn(3, 3),
                z=torch.randint(1, 100, (3,)),
                num_atoms=torch.tensor([0, 3]),
            )

    def test_zero_structures(self):
        with pytest.raises(InvalidStructure, match="must contain at least one crystal"):
            Structures(
                lattice=torch.empty(0, 3, 3),
                x=torch.empty(0, 3),
                z=torch.tensor([], dtype=torch.long),
            )


@pytest.fixture
def two_structures():
    return [
        Structures(
            lattice=torch.randn(1, 3, 3),
            x=torch.randn(2, 3),
            z=torch.tensor([1, 6]),
        ),
        Structures(
            lattice=torch.randn(1, 3, 3),
            x=torch.randn(2, 3),
            z=torch.tensor([8, 1]),
        ),
    ]


@pytest.fixture
def two_structures_with_energy():
    return [
        Structures(
            lattice=torch.randn(1, 3, 3),
            x=torch.randn(2, 3),
            z=torch.tensor([1, 6]),
            total_energy=torch.tensor([-5.0]),
            energy_above_hull=torch.tensor([0.1]),
            band_gap=torch.tensor([1.5]),
        ),
        Structures(
            lattice=torch.randn(1, 3, 3),
            x=torch.randn(3, 3),
            z=torch.tensor([8, 1, 6]),
            total_energy=torch.tensor([-3.0]),
            energy_above_hull=torch.tensor([0.2]),
            band_gap=torch.tensor([2.0]),
        ),
    ]


class TestStructuresCollate:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Structures.collate([])

    def test_single_structure(self):
        s = Structures(
            lattice=torch.randn(1, 3, 3),
            x=torch.randn(2, 3),
            z=torch.tensor([1, 6]),
        )

        batched = Structures.collate([s])

        assert batched.lattice.shape == (1, 3, 3)
        assert batched.x.shape == (2, 3)
        assert batched.z.shape == (2,)
        assert torch.equal(batched.batch, torch.tensor([0, 0]))
        assert torch.equal(batched.num_atoms, torch.tensor([2]))
        assert torch.equal(batched.ptr, torch.tensor([0, 2]))

    def test_two_structures_equal_atoms(self, two_structures):
        s1, s2 = two_structures
        batched = Structures.collate(two_structures)

        assert torch.equal(batched.num_atoms, torch.tensor([2, 2]))
        assert torch.equal(batched.batch, torch.tensor([0, 0, 1, 1]))
        assert torch.equal(batched.ptr, torch.tensor([0, 2, 4]))
        assert torch.equal(batched.x, torch.cat([s1.x, s2.x], dim=0))
        assert torch.equal(batched.z, torch.cat([s1.z, s2.z], dim=0))
        assert torch.equal(batched.lattice, torch.cat([s1.lattice, s2.lattice], dim=0))

    def test_two_structures_different_atoms(self):
        s1 = Structures(
            lattice=torch.randn(1, 3, 3),
            x=torch.randn(1, 3),
            z=torch.tensor([1]),
        )
        s2 = Structures(
            lattice=torch.randn(1, 3, 3),
            x=torch.randn(3, 3),
            z=torch.tensor([8, 1, 6]),
        )

        batched = Structures.collate([s1, s2])

        assert torch.equal(batched.num_atoms, torch.tensor([1, 3]))
        assert torch.equal(batched.batch, torch.tensor([0, 1, 1, 1]))
        assert torch.equal(batched.ptr, torch.tensor([0, 1, 4]))
        assert torch.equal(batched.x, torch.cat([s1.x, s2.x], dim=0))
        assert torch.equal(batched.z, torch.cat([s1.z, s2.z], dim=0))

    def test_ptr_computation(self, two_structures):
        batched = Structures.collate(two_structures)

        assert torch.equal(batched.ptr, torch.tensor([0, 2, 4]))

    def test_all_structures_have_optionals(self, two_structures_with_energy):
        batched = Structures.collate(two_structures_with_energy)

        assert batched.total_energy.shape == (2,)
        assert torch.equal(batched.total_energy, torch.tensor([-5.0, -3.0]))
        assert torch.equal(batched.energy_above_hull, torch.tensor([0.1, 0.2]))
        assert torch.equal(batched.band_gap, torch.tensor([1.5, 2.0]))

    def test_no_structures_have_optionals(self, two_structures):
        batched = Structures.collate(two_structures)

        assert batched.total_energy.shape == (2,)
        assert batched.energy_above_hull.shape == (2,)
        assert batched.band_gap.shape == (2,)
        assert torch.isnan(batched.total_energy).all()
        assert torch.isnan(batched.energy_above_hull).all()
        assert torch.isnan(batched.band_gap).all()

    def test_collate_batches(self, two_structures_with_energy):
        s1, s2 = two_structures_with_energy
        expected = Structures.collate([s1, s2, s1, s2, s2])

        # batches of two, two and one structures
        batched = Structures.collate(
            [Structures.collate([s1, s2]), Structures.collate([s1, s2]), s2]
        )

        batched.check_tensors()
        assert torch.equal(batched.num_atoms, torch.tensor([2, 3, 2, 3, 3]))
        assert torch.equal(
            batched.batch, torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
        )
        assert torch.equal(batched.ptr, torch.tensor([0, 2, 5, 7, 10, 13]))
        assert torch.equal(batched.batch, expected.batch)
        assert torch.equal(batched.x, expected.x)
        assert torch.equal(batched.z, expected.z)
        assert torch.equal(batched.lattice, expected.lattice)
        assert torch.equal(batched.total_energy, expected.total_energy)


class TestStructuresSeparate:
    def test_single_structure(self):
        x = torch.randn(2, 3)
        z = torch.tensor([1, 6])
        lattice = torch.randn(1, 3, 3)
        s = Structures(
            lattice=lattice,
            x=x,
            z=z,
        )

        separated = Structures.separate(s)

        assert len(separated) == 1
        assert torch.equal(separated[0].x, x)
        assert torch.equal(separated[0].z, z)
        assert torch.equal(separated[0].num_atoms, torch.tensor([2]))
        assert torch.equal(separated[0].ptr, torch.tensor([0, 2]))
        assert torch.equal(separated[0].lattice, lattice)

    def test_multiple_structures(self, two_structures):
        s1, s2 = two_structures
        batched = Structures.collate(two_structures)
        separated = Structures.separate(batched)

        assert len(separated) == 2
        assert torch.equal(separated[0].num_atoms, s1.num_atoms)
        assert torch.equal(separated[1].num_atoms, s2.num_atoms)
        assert torch.equal(separated[0].x, s1.x)
        assert torch.equal(separated[1].x, s2.x)
        assert torch.equal(separated[0].z, s1.z)
        assert torch.equal(separated[1].z, s2.z)
        assert torch.equal(separated[0].lattice, s1.lattice)
        assert torch.equal(separated[1].lattice, s2.lattice)

    def test_preserves_optionals(self, two_structures_with_energy):
        batched = Structures.collate(two_structures_with_energy)
        separated = Structures.separate(batched)

        assert torch.equal(separated[0].total_energy, torch.tensor([-5.0]))
        assert torch.equal(separated[1].total_energy, torch.tensor([-3.0]))
        assert torch.equal(separated[0].energy_above_hull, torch.tensor([0.1]))
        assert torch.equal(separated[1].energy_above_hull, torch.tensor([0.2]))

    def test_separate_resets_batch(self, two_structures):
        batched = Structures.collate(two_structures)
        separated = Structures.separate(batched)

        for orig, sep in zip(two_structures, separated):
            assert torch.equal(sep.batch, torch.zeros_like(orig.batch))
            assert torch.equal(sep.batch, orig.batch)
            # a valid structure on its own
            sep.check_tensors()
            assert int(sep.batch.max()) < len(sep.num_atoms)


class TestStructuresRoundTrip:
    def test_roundtrip_data_integrity(self, two_structures):
        batched = Structures.collate(two_structures)
        separated = Structures.separate(batched)

        for orig, sep in zip(two_structures, separated):
            assert torch.allclose(orig.lattice, sep.lattice)
            assert torch.allclose(orig.x, sep.x)
            assert torch.equal(orig.z, sep.z)
            assert torch.equal(orig.num_atoms, sep.num_atoms)

    def test_roundtrip_optionals_mixed(self, two_structures_with_energy):
        batched = Structures.collate(two_structures_with_energy)
        separated = Structures.separate(batched)

        for orig, sep in zip(two_structures_with_energy, separated):
            assert torch.equal(orig.total_energy, sep.total_energy)
            assert torch.equal(orig.energy_above_hull, sep.energy_above_hull)
            assert torch.equal(orig.band_gap, sep.band_gap)


class TestFormationEnergy:
    def test_default_is_nan(self, multiple_structures_fields):
        struct = Structures(**multiple_structures_fields)

        assert struct.formation_energy_per_atom.shape == (2,)
        assert struct.formation_energy_per_atom.dtype == torch.float32
        assert torch.isnan(struct.formation_energy_per_atom).all()

    def test_invalid_shape(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["formation_energy_per_atom"] = torch.tensor([-1.0])

        with pytest.raises(InvalidStructure, match="invalid shape"):
            Structures(**fields)

    def test_collate_and_separate(self, two_structures):
        two_structures[0].formation_energy_per_atom = torch.tensor([-1.5])
        two_structures[1].formation_energy_per_atom = torch.tensor([0.25])

        batched = Structures.collate(two_structures)
        assert torch.equal(
            batched.formation_energy_per_atom, torch.tensor([-1.5, 0.25])
        )

        separated = Structures.separate(batched)
        assert torch.equal(separated[0].formation_energy_per_atom, torch.tensor([-1.5]))
        assert torch.equal(separated[1].formation_energy_per_atom, torch.tensor([0.25]))
