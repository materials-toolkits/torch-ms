"""Tests for the rewrite of the energy prediction of MACE."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from pycrystalgen.data import Structures, parse_cif
from pycrystalgen.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    LatticeBasisChange,
    LatticeRotation,
    make_supercell,
    random_orthogonal,
)
from pycrystalgen.estimator.mace import InvalidWeights, MaceEnergyModel, weights
from pycrystalgen.estimator.mace.model import (
    DEGREES,
    EXPECTED,
    NUM_HARMONICS,
    PARAMETERS,
    RADIUS,
)
from pycrystalgen.estimator.mace.reader import Placeholder, read_model
from pycrystalgen.estimator.orb.model import spherical_harmonics

DATA = Path(__file__).parent / "data"
NAME = "mace-mp-0-small"

# products of three environments of 128 channels: slower than the default timeout
pytestmark = pytest.mark.timeout(60)


def make_structures() -> list[Structures]:
    with open("tests/cif/triclinic1.cif", "r", encoding="utf-8") as fp:
        triclinic = parse_cif(fp.read())

    return [Structures.zincblende(), triclinic]


def make_batch() -> Structures:
    return Structures.collate(make_structures())


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def mace_state_dict(model: MaceEnergyModel) -> dict[str, torch.Tensor]:
    """Parameters of a model as mace stores them: flat weights and its own names."""
    own = model.state_dict()
    state_dict = {
        name: own[target].flatten() for name, (target, _) in PARAMETERS.items()
    }
    state_dict.update({name: torch.tensor(value) for name, value in EXPECTED.items()})
    return state_dict


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS) == {NAME}
    checkpoint = weights.CHECKPOINTS[NAME]
    assert checkpoint.url.startswith("https://")
    assert len(checkpoint.sha256) == 64
    assert weights.checkpoint_path(NAME, tmp_path) == (
        tmp_path / "mace" / "2023-12-10-mace-128-L0_energy_epoch-249.model"
    )

    with pytest.raises(KeyError, match="unknown model"):
        weights.checkpoint_path("mace-9")


# reading of a pickled model without its classes


class Block(torch.nn.Module):
    """Module of a pickled model, as the ones of mace."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(4))])
        self.register_buffer("scale", torch.tensor(2.5, dtype=torch.float64))
        self.register_buffer("temporary", torch.zeros(2), persistent=False)
        self.avg_num_neighbors = np.float64(12.5)
        self.activation = torch.nn.functional.silu


class Pickled(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(), Block()])
        self.register_buffer("atomic_numbers", torch.tensor([1, 6, 8]))


def test_read_model_gives_the_state_dict(tmp_path):
    model = Pickled()
    torch.save(model, tmp_path / "model.pt")

    tensors, modules = read_model(tmp_path / "model.pt")

    expected = model.state_dict()
    assert set(tensors) == set(expected)
    assert "blocks.0.temporary" not in tensors
    for name, tensor in expected.items():
        assert torch.equal(tensors[name], tensor)
        assert tensors[name].dtype == tensor.dtype

    # the classes of the model are not imported, their attributes are kept
    assert all(isinstance(module, Placeholder) for module in modules.values())
    assert not any(isinstance(module, torch.nn.Module) for module in modules.values())
    assert modules[""].path.endswith(".Pickled")
    assert modules["blocks.1"].path.endswith(".Block")
    assert modules["blocks.1"].attributes()["avg_num_neighbors"] == 12.5


class Malicious:
    """Object whose unpickling runs a function, as a file from the internet could."""

    def __reduce__(self):
        return (exec, ("raise SystemExit('executed')",))


def test_read_model_does_not_execute_the_file(tmp_path):
    torch.save(Malicious(), tmp_path / "malicious.pt")

    # unpickled as usual, the file runs its function
    with pytest.raises(SystemExit, match="executed"):
        torch.load(tmp_path / "malicious.pt", weights_only=False)

    # here the function is a placeholder which is never called
    with pytest.raises(pickle.UnpicklingError, match="not a pickled torch module"):
        read_model(tmp_path / "malicious.pt")


def test_read_model_rejects_a_state_dict(tmp_path):
    torch.save(Pickled().state_dict(), tmp_path / "state.pt")

    with pytest.raises(pickle.UnpicklingError, match="not a pickled torch module"):
        read_model(tmp_path / "state.pt")


def test_degrees():
    assert len(DEGREES) == NUM_HARMONICS == 16
    assert [DEGREES.count(degree) for degree in range(4)] == [1, 3, 5, 7]


def test_model_with_random_weights():
    torch.manual_seed(0)
    model = MaceEnergyModel().eval().requires_grad_(False)
    energy = model(make_batch())

    assert energy.shape == (2,)
    assert energy.dtype == torch.float64
    assert energy.isfinite().all()


def test_model_unknown_element():
    model = MaceEnergyModel().eval().requires_grad_(False)
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[100]
    )

    with pytest.raises(ValueError, match="not supported"):
        model(structures)


def test_load_pretrained_checks_the_parameters():
    torch.manual_seed(0)
    model = MaceEnergyModel().eval().requires_grad_(False)
    for product in model.products:
        for order in (1, 2, 3):
            getattr(product, f"U_matrix_{order}").normal_()
    state_dict = mace_state_dict(model)

    other = MaceEnergyModel()
    other.load_pretrained(state_dict, 30.0, {"a": 1.5, "b": 1.5})
    other.eval()
    assert other.avg_num_neighbors == 30.0 and other.activation == 1.5

    model.avg_num_neighbors, model.activation = 30.0, 1.5
    with torch.no_grad():
        assert torch.allclose(other(make_batch()), model(make_batch()))

    with pytest.raises(InvalidWeights, match="different activation"):
        other.load_pretrained(state_dict, 30.0, {"a": 1.5, "b": 1.6})

    with pytest.raises(InvalidWeights, match="r_max must be"):
        other.load_pretrained({**state_dict, "r_max": torch.tensor(5.0)}, 30.0)

    with pytest.raises(InvalidWeights, match="expected"):
        other.load_pretrained({**state_dict, "scale_shift.scale": torch.ones(2)}, 30.0)

    del state_dict["products.1.linear.weight"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state_dict, 30.0)


# the pretrained model against the real mace


@pytest.fixture(scope="module")
def pretrained() -> MaceEnergyModel:
    if not weights.checkpoint_path(NAME).exists():
        pytest.skip(f"weights of {NAME} are not in the cache")

    return MaceEnergyModel.from_pretrained(NAME, device="cpu")


def reference_structures() -> tuple[Structures, dict[str, np.ndarray]]:
    data = dict(np.load(DATA / "orb_reference.npz"))
    structures = Structures(
        lattice=torch.from_numpy(data["lattice"]),
        x=torch.from_numpy(data["x"]),
        z=torch.from_numpy(data["z"]),
        num_atoms=torch.from_numpy(data["num_atoms"]),
    )
    return structures, dict(np.load(DATA / "mace_reference.npz"))


def test_pretrained_matches_mace(pretrained):
    """Energies of mace 0.3.16, computed once outside of the project."""
    structures, data = reference_structures()
    graph, vectors = pretrained.edges(structures)

    assert graph.num_edges.tolist() == data["num_edges"].tolist()
    assert vectors.dtype == torch.float64
    assert (vectors.norm(dim=-1) <= RADIUS + 1e-5).all()

    energy = pretrained(structures).numpy()
    error = np.abs(energy - data["energy"]) / structures.num_atoms.numpy()

    assert error.max() < 1e-9


def test_pretrained_coupling_tensors_are_invariant(pretrained):
    # the products of the harmonics of a direction do not depend on the
    # orientation: the coupling tensors give invariants
    direction = torch.randn(5, 3, dtype=torch.float64, generator=generator())
    rotation = random_orthogonal(1, torch.device("cpu"), generator())[0].double()
    product = pretrained.products[0]

    def invariants(vectors: torch.Tensor) -> list[torch.Tensor]:
        y = spherical_harmonics(vectors)
        return [
            torch.einsum("wk,bw->bk", product.U_matrix_1, y),
            torch.einsum("wxk,bw,bx->bk", product.U_matrix_2, y, y),
            torch.einsum("wxik,bw,bx,bi->bk", product.U_matrix_3, y, y, y),
        ]

    for before, after in zip(invariants(direction), invariants(direction @ rotation.T)):
        assert before.abs().max() > 0.1
        assert torch.allclose(before, after, atol=1e-10)


def test_pretrained_batch_equals_single(pretrained):
    alone = torch.cat([pretrained(structure) for structure in make_structures()])

    assert torch.allclose(pretrained(make_batch()), alone, rtol=1e-10)


@pytest.mark.parametrize(
    "transform",
    [AtomPermutation, LatticeRotation, AtomTranslation, LatticeBasisChange],
)
def test_pretrained_invariances(pretrained, transform):
    batch = make_batch()
    augmented = transform(generator=generator())(batch)

    # the structures are in single precision
    assert torch.allclose(pretrained(augmented), pretrained(batch), rtol=1e-5)


def test_pretrained_energy_of_a_supercell(pretrained):
    batch = make_batch()
    repeat = torch.tensor([[2, 1, 1], [1, 1, 1]])
    supercell = make_supercell(batch, repeat)

    assert torch.allclose(
        pretrained(supercell), pretrained(batch) * repeat.prod(dim=1), rtol=1e-5
    )
