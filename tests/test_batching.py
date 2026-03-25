import pytest
import torch
import numpy as np

from pycrystalgen.data import Structures
from pycrystalgen.data.batching import InvalidStructure


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
    def test_init_with_required_fields_single_structure(self, single_structure_fields):
        struct = Structures(**single_structure_fields)

        assert struct.lattice.shape == (1, 3, 3)
        assert struct.x.shape == (4, 3)
        assert struct.z.shape == (4,)
        assert struct._validate is True

    def test_init_with_required_fields_multiple_structures(
        self, multiple_structures_fields
    ):
        struct = Structures(**multiple_structures_fields)

        assert struct.lattice.shape == (2, 3, 3)
        assert struct.x.shape == (4, 3)
        assert struct.z.shape == (4,)
        assert struct._validate is True

    def test_init_with_all_fields(self, multiple_structures_fields):
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

        assert struct.num_atoms is not None
        assert struct.batch is not None
        assert struct.ptr is not None
        assert struct.total_energy is not None
        assert struct.energy_above_hull is not None
        assert struct.band_gap is not None
        assert torch.equal(struct.num_atoms, torch.tensor([2, 2]))
        assert torch.equal(struct.batch, torch.tensor([0, 0, 1, 1]))
        assert torch.equal(struct.ptr, torch.tensor([0, 2, 4]))

    def test_init_single_structure(self):
        struct = Structures(
            lattice=torch.randn(1, 3, 3),
            x=torch.randn(3, 3),
            z=torch.randint(1, 100, (3,)),
        )

        assert struct.lattice.shape == (1, 3, 3)
        assert struct.x.shape == (3, 3)
        assert struct.z.shape == (3,)

    def test_init_validate_flag_skips_validation(self, single_structure_fields):
        fields = single_structure_fields.copy()
        fields["_validate"] = True

        struct = Structures(**fields)

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
        assert struct.ptr.shape == (3,)
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

    def test_inconsistent_total_energy_shape(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["total_energy"] = torch.tensor([1.0, 2.0, 3.0])

        with pytest.raises(InvalidStructure, match="invalid shape"):
            Structures(**fields)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_inconsistent_device(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["x"] = torch.randn(4, 3).cuda()

        with pytest.raises(InvalidStructure, match="inconsistend device"):
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

        struct = Structures(**fields)

        assert struct.device() == torch.device("cuda:0")
        assert struct.lattice.device == torch.device("cuda:0")
        assert struct.x.device == torch.device("cuda:0")
        assert struct.z.device == torch.device("cuda:0")


class TestStructuresBatching:
    def test_contiguous_batching(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["batch"] = torch.tensor([0, 0, 1, 1])

        struct = Structures(**fields)

        assert torch.equal(struct.batch, torch.tensor([0, 0, 1, 1]))
        assert torch.equal(struct.ptr, torch.tensor([0, 2, 4]))

    def test_batch_matches_num_atoms(self, multiple_structures_fields):
        fields = multiple_structures_fields.copy()
        fields["batch"] = torch.tensor([0, 1, 0, 1])

        struct = Structures(**fields)

        assert struct.num_atoms[0] == 2
        assert struct.num_atoms[1] == 2


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
