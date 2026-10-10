"""Tests for the rewrite of the two models of DiffCSP."""

from pathlib import Path

import numpy as np
import pytest
import torch

from torchms.data import Structures, parse_cif
from torchms.data.augmentation import (
    AtomTranslation,
    LatticeRotation,
    actions,
    sampling,
)
from torchms.graph.transforms import AddRadiusGraph
from torchms.models.diffcsp import (
    DiffCspGenerationModel,
    DiffCspModel,
    GenerationNoise,
    InvalidWeights,
)
from torchms.models.diffcsp import model as diffcsp_model
from torchms.models.diffcsp.model import (
    GENERATION_UPSTREAM_ARGS,
    NUM_ATOMS_DISTRIBUTION,
    NUM_ELEMENTS,
    SIGMA_BEGIN,
    SIGMA_END,
    TIMESTEPS,
    UPSTREAM_ARGS,
)
from torchms.models.estimator import weights
from torchms.models.layers import periodic_fourier, sinusoidal_embedding

DATA = Path(__file__).parent / "data"
NAME = "diffcsp-mp20"
GENERATION_NAME = "diffcsp-gen-mp20"

# a network of 512 features on every pair of atoms: slower than the default timeout
pytestmark = pytest.mark.timeout(60)


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_structures() -> list[Structures]:
    with open("tests/cif/triclinic1.cif", "r", encoding="utf-8") as fp:
        triclinic = parse_cif(fp.read())

    return [Structures.zincblende(), triclinic]


def make_batch() -> Structures:
    return Structures.collate(make_structures())


@pytest.fixture(scope="module")
def model() -> DiffCspModel:
    """Model with random weights, and scores scaled as the ones of small noise."""
    torch.manual_seed(0)
    model = DiffCspModel().eval().requires_grad_(False)
    sigmas = model.sigma_scheduler.sigmas
    model.sigma_scheduler.sigmas_norm[1:] = 1.0 / sigmas[1:] ** 2

    return model


TIMES = torch.tensor([3, 700])


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS["diffcsp"]) == {NAME, GENERATION_NAME}
    assert DiffCspModel.DEFAULT_WEIGHTS == NAME
    assert DiffCspGenerationModel.DEFAULT_WEIGHTS == GENERATION_NAME

    for name in (NAME, GENERATION_NAME):
        for checkpoint in weights.CHECKPOINTS["diffcsp"][name]:
            # a file of Google Drive by its identifier, not a folder or a name
            assert checkpoint.url.startswith(
                "https://drive.usercontent.google.com/download?id="
            )
            assert len(checkpoint.sha256) == 64

        paths = weights.checkpoint_paths("diffcsp", name, tmp_path)
        assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
            f"diffcsp/{name}/last.ckpt",
            f"diffcsp/{name}/hparams.yaml",
        ]

    with pytest.raises(KeyError, match="unknown model"):
        weights.checkpoint_paths("diffcsp", "diffcsp-9")


# the bases


def test_periodic_fourier():
    x = torch.rand(7, 3, generator=generator())
    basis = periodic_fourier(x, 5)

    assert basis.shape == (7, 30)
    # sines then cosines, by coordinate then by frequency
    assert torch.allclose(basis[:, 15::5], torch.ones(7, 3))
    assert torch.allclose(basis[:, 6], torch.sin(2 * torch.pi * x[:, 1]), atol=1e-6)
    assert torch.allclose(basis[:, 27], torch.cos(4 * torch.pi * x[:, 2]), atol=1e-6)

    # of period 1
    shift = torch.tensor([[1.0, -2.0, 3.0]])
    assert torch.allclose(periodic_fourier(x + shift, 5), basis, atol=1e-4)

    assert periodic_fourier(x.double(), 5).dtype == torch.float64


def test_sinusoidal_embedding():
    time = torch.tensor([0, 1, 500])
    embedding = sinusoidal_embedding(time, 16, 10000.0)

    assert embedding.shape == (3, 16)
    assert torch.equal(embedding[0], torch.tensor([0.0] * 8 + [1.0] * 8))
    # frequencies from 1 to 1 / max_period
    assert torch.allclose(embedding[1, [0, 7]], torch.tensor([1.0, 1e-4]).sin())
    assert torch.allclose(embedding[:, :8] ** 2 + embedding[:, 8:] ** 2, torch.ones(8))
    assert torch.equal(sinusoidal_embedding(time.float(), 16, 10000.0), embedding)


# the schedules of the noise


def test_schedules(model):
    beta, sigma = model.beta_scheduler, model.sigma_scheduler

    for values in (beta.betas, beta.alphas, beta.alphas_cumprod, beta.sigmas):
        assert values.shape == (TIMESTEPS + 1,)
    assert sigma.sigmas.shape == sigma.sigmas_norm.shape == (TIMESTEPS + 1,)

    # the time 0 has no noise
    assert beta.betas[0] == 0 and beta.alphas_cumprod[0] == 1 and beta.sigmas[0] == 0
    assert sigma.sigmas[0] == 0

    # the lattice is only noise at the last time
    assert (beta.alphas_cumprod[1:] < beta.alphas_cumprod[:-1]).all()
    assert beta.alphas_cumprod[-1] < 1e-8
    assert ((beta.betas[1:] >= 1e-4) & (beta.betas[1:] <= 0.9999)).all()

    assert torch.allclose(sigma.sigmas[[1, -1]], torch.tensor([SIGMA_BEGIN, SIGMA_END]))
    assert (sigma.sigmas[1:] > sigma.sigmas[:-1]).all()
    ratio = sigma.sigmas[2:] / sigma.sigmas[1:-1]
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-5)


# the network with random weights


def test_output(model):
    structures = model.transform()(make_batch())
    lattice_noise, score = model(structures, TIMES)

    assert lattice_noise.shape == (2, 3, 3)
    assert score.shape == (len(structures.z), 3)
    assert lattice_noise.dtype == score.dtype == torch.float32
    assert lattice_noise.isfinite().all() and score.isfinite().all()
    assert score.abs().max() > 0


def test_graph_is_complete(model):
    structures = model.transform()(make_batch())
    graph = structures.graph

    assert torch.equal(graph.num_edges, structures.num_atoms**2)
    assert (graph.edge_shift == 0).all()
    assert (graph.senders == graph.receivers).sum() == len(structures.z)


def test_depends_on_the_time(model):
    structures = model.transform()(make_batch())
    lattice_noise, score = model(structures, TIMES)
    other_noise, other_score = model(structures, torch.tensor([3, 701]))

    first = structures.batch == 0
    assert torch.equal(other_noise[0], lattice_noise[0])
    assert torch.equal(other_score[first], score[first])
    assert not torch.allclose(other_noise[1], lattice_noise[1])
    assert not torch.allclose(other_score[~first], score[~first])


def test_batch_equals_single(model):
    transform = model.transform()
    noises, scores = zip(
        *[
            model(transform(structure), TIMES[i : i + 1])
            for i, structure in enumerate(make_structures())
        ]
    )
    lattice_noise, score = model(transform(make_batch()), TIMES)

    assert torch.allclose(lattice_noise, torch.cat(noises), atol=1e-5)
    assert torch.allclose(score, torch.cat(scores), atol=1e-5)


def test_follows_a_rotation(model):
    batch = make_batch()
    rotated = LatticeRotation(generator=generator())(batch)

    lattice_noise, score = model(model.transform()(batch), TIMES)
    rotated_noise, rotated_score = model(model.transform()(rotated), TIMES)

    # the scores are in fractional coordinates, the noise of the lattice is
    # rotated with it
    assert torch.allclose(rotated_score, score, atol=1e-4)
    rotation = torch.linalg.solve(batch.lattice, rotated.lattice)
    assert torch.allclose(rotated_noise, lattice_noise @ rotation, atol=1e-4)


def test_translation_invariance(model):
    batch = make_batch()
    translated = AtomTranslation(generator=generator())(batch)
    assert not torch.allclose(translated.x, batch.x)

    lattice_noise, score = model(model.transform()(batch), TIMES)
    translated_noise, translated_score = model(model.transform()(translated), TIMES)

    assert torch.allclose(translated_noise, lattice_noise, atol=1e-4)
    assert torch.allclose(translated_score, score, atol=1e-4)


def test_follows_a_permutation(model):
    batch = make_batch()
    permutation = sampling.random_permutation(batch.batch, generator())
    permuted = actions.permute_atoms(batch, permutation)
    assert torch.equal(permuted.z, batch.z[permutation])
    assert not torch.equal(permuted.z, batch.z)

    lattice_noise, score = model(model.transform()(batch), TIMES)
    permuted_noise, permuted_score = model(model.transform()(permuted), TIMES)

    assert torch.allclose(permuted_noise, lattice_noise, atol=1e-4)
    assert torch.allclose(permuted_score, score[permutation], atol=1e-4)


def test_coordinates_outside_of_the_cell(model):
    batch = make_batch()
    shift = torch.randint(-2, 3, batch.x.shape, generator=generator()).float()
    moved = Structures(
        lattice=batch.lattice, x=batch.x + shift, z=batch.z, num_atoms=batch.num_atoms
    )

    lattice_noise, score = model(model.transform()(batch), TIMES)
    moved_noise, moved_score = model(model.transform()(moved), TIMES)

    assert torch.allclose(moved_noise, lattice_noise, atol=1e-3)
    assert torch.allclose(moved_score, score, atol=1e-3)


def test_needs_its_graph(model):
    with pytest.raises(ValueError, match=r"apply DiffCspModel.transform\(\)"):
        model(make_batch(), TIMES)


def test_another_graph_gives_another_output(model):
    # the graph attached is read: the model does not build its own
    batch = make_batch()
    _, score = model(model.transform()(batch), TIMES)
    _, other = model(AddRadiusGraph(3.0)(batch), TIMES)

    assert not torch.allclose(other, score, atol=1e-4)


def test_unknown_element(model):
    # mendelevium has no row in the embedding
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[101]
    )
    assert NUM_ELEMENTS == 100

    with pytest.raises(ValueError, match="not supported"):
        model(model.transform()(structures), torch.tensor([1]))

    with pytest.raises(ValueError, match="not supported"):
        model.sample(structures.z, structures.num_atoms)


# sampling with random weights


def step_noise(batch: Structures, seed: int = 0):
    random = generator(seed)

    return (
        torch.randn(batch.x.shape, generator=random),
        torch.randn(batch.lattice.shape, generator=random),
        torch.randn(batch.x.shape, generator=random),
    )


def test_sample_step(model):
    batch = make_batch()
    time = 400
    corrector, lattice, predictor = step_noise(batch)
    zero_x, zero_l = torch.zeros_like(batch.x), torch.zeros_like(batch.lattice)

    moved = model.sample_step(batch, time, corrector, lattice, predictor)
    assert moved.graph is None
    assert torch.equal(moved.z, batch.z)
    assert torch.equal(moved.num_atoms, batch.num_atoms)
    assert (moved.x >= 0).all() and (moved.x < 1).all()
    assert not torch.allclose(moved.lattice, batch.lattice)

    # the structures given are not modified, and can have a graph
    again = model.sample_step(
        model.transform()(batch), time, corrector, lattice, predictor
    )
    assert torch.equal(again.x, moved.x) and torch.equal(again.lattice, moved.lattice)

    # without noise, the lattice follows the noise the network finds in it
    quiet = model.sample_step(batch, time, zero_x, zero_l, zero_x, step_size=0.0)
    beta = model.beta_scheduler
    found, _ = model(model.transform()(batch), torch.full((2,), time))
    scale = (1 - beta.alphas[time]) / (1 - beta.alphas_cumprod[time]).sqrt()
    expected = (batch.lattice - scale * found) / beta.alphas[time].sqrt()
    assert torch.allclose(quiet.lattice, expected, atol=1e-5)

    # the noise of the lattice is added with the deviation of the time, and
    # does not move the atoms
    kept = model.sample_step(batch, time, corrector, zero_l, predictor)
    assert torch.allclose(
        moved.lattice - kept.lattice, beta.sigmas[time] * lattice, atol=1e-5
    )
    assert torch.equal(kept.x, moved.x)

    # a corrector without step adds no noise
    noisy = model.sample_step(batch, time, corrector, zero_l, zero_x, step_size=0.0)
    assert torch.equal(noisy.x, quiet.x)
    assert not torch.equal(kept.x, quiet.x)


def test_sample_step_moves_the_atoms_along_the_score(model):
    batch = make_batch()
    time = 400
    zero_x, zero_l = torch.zeros_like(batch.x), torch.zeros_like(batch.lattice)
    sigmas = model.sigma_scheduler

    # without correction and without noise: one step of the predictor
    moved = model.sample_step(batch, time, zero_x, zero_l, zero_x, step_size=0.0)

    _, score = model(model.transform()(batch), torch.full((2,), time))
    step = sigmas.sigmas[time] ** 2 - sigmas.sigmas[time - 1] ** 2
    expected = batch.x - step * score * sigmas.sigmas_norm[time].sqrt()
    difference = (moved.x - expected + 0.5) % 1.0 - 0.5
    assert difference.abs().max() < 1e-5
    assert (step * score * sigmas.sigmas_norm[time].sqrt()).abs().max() > 1e-4


def test_sample(model, monkeypatch):
    # the last steps of the diffusion only: all of them take seconds
    monkeypatch.setattr(diffcsp_model, "TIMESTEPS", 20)
    z = torch.tensor([11, 17, 8])
    num_atoms = torch.tensor([2, 1])

    structures = model.sample(z, num_atoms, generator=generator())
    assert structures.graph is None
    assert structures.lattice.shape == (2, 3, 3)
    assert torch.equal(structures.z, z)
    assert torch.equal(structures.num_atoms, num_atoms)
    assert (structures.x >= 0).all() and (structures.x < 1).all()

    # the same draws give the same structures
    again = model.sample(z, num_atoms, generator=generator())
    assert torch.equal(again.x, structures.x)
    assert torch.equal(again.lattice, structures.lattice)
    other = model.sample(z, num_atoms, generator=generator(1))
    assert not torch.equal(other.x, structures.x)


# the weights


def upstream_hparams(
    arguments: dict[str, dict[str, object]] = UPSTREAM_ARGS,
) -> dict[str, object]:
    """Configuration of the training of DiffCSP a rewrite is written for."""
    return {
        "model": {
            **arguments["model"],
            "decoder": dict(arguments["model.decoder"]),
            "beta_scheduler": dict(arguments["model.beta_scheduler"]),
            "sigma_scheduler": dict(arguments["model.sigma_scheduler"]),
        },
    }


def test_load_pretrained_checks_the_parameters(model):
    state = dict(model.state_dict())
    assert "elements" not in state

    other = DiffCspModel()
    other.load_pretrained(state, upstream_hparams())
    other.eval()
    with torch.no_grad():
        structures = model.transform()(make_batch())
        assert torch.equal(other(structures, TIMES)[1], model(structures, TIMES)[1])
    assert torch.equal(
        other.sigma_scheduler.sigmas_norm, model.sigma_scheduler.sigmas_norm
    )

    # the model which also generates the elements, with another network
    hparams = upstream_hparams()
    hparams["model"]["_target_"] = "diffcsp.pl_modules.diffusion_w_type.CSPDiffusion"
    hparams["model"]["decoder"]["num_layers"] = 4
    with pytest.raises(InvalidWeights, match="another configuration") as error:
        other.load_pretrained(state, hparams)
    assert "model._target_" in str(error.value)
    assert "model.decoder.num_layers" in str(error.value)

    with pytest.raises(InvalidWeights, match="no model.sigma_scheduler"):
        other.load_pretrained(state, {"model": {"decoder": {}, "beta_scheduler": {}}})

    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state, "decoder.type_out.bias": torch.zeros(100)})

    del state["sigma_scheduler.sigmas_norm"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state)


# the pretrained model against the real DiffCSP


def reference() -> tuple[Structures, dict[str, torch.Tensor]]:
    """Structures of the tests and what DiffCSP computed for them."""
    data = dict(np.load(DATA / "orb_reference.npz"))
    expected = {
        name: torch.from_numpy(values)
        for name, values in np.load(DATA / "diffcsp_reference.npz").items()
    }
    structures = Structures(
        lattice=torch.from_numpy(data["lattice"]),
        x=torch.from_numpy(data["x"]),
        z=torch.from_numpy(data["z"]),
        num_atoms=torch.from_numpy(data["num_atoms"]),
    )

    return structures, expected


def close(found: torch.Tensor, expected: torch.Tensor) -> bool:
    """Whether values are the expected ones up to the rounding of float32.

    The difference is under 1e-5 of the largest expected value, or 1e-5.
    """
    scale = max(1.0, float(expected.abs().max()))

    return float((found - expected).abs().max()) < 1e-5 * scale


def close_in_the_cell(found: torch.Tensor, expected: torch.Tensor) -> bool:
    """Whether fractional coordinates are the expected ones, modulo the cell."""
    difference = (found - expected + 0.5) % 1.0 - 0.5

    return float(difference.abs().max()) < 1e-5


@pytest.fixture(scope="module")
def pretrained() -> DiffCspModel:
    if not all(path.exists() for path in weights.checkpoint_paths("diffcsp", NAME)):
        pytest.skip(f"weights of {NAME} are not in the cache")

    return DiffCspModel.from_pretrained(NAME, device="cpu")


def test_pretrained_schedules_are_the_ones_of_the_configuration(pretrained):
    """The schedules of the weights are the ones computed from config.yaml."""
    computed = DiffCspModel()

    for name in ("betas", "alphas", "alphas_cumprod", "sigmas"):
        found = getattr(pretrained.beta_scheduler, name)
        assert torch.allclose(found, getattr(computed.beta_scheduler, name), atol=1e-7)
    assert torch.equal(
        pretrained.sigma_scheduler.sigmas, computed.sigma_scheduler.sigmas
    )

    # the mean squared score of a small noise is the inverse of its variance,
    # up to the random draws of DiffCSP
    sigmas = pretrained.sigma_scheduler.sigmas[1:200]
    norm = pretrained.sigma_scheduler.sigmas_norm[1:200]
    assert pretrained.sigma_scheduler.sigmas_norm[0] == 1
    assert torch.allclose(norm * sigmas**2, torch.ones_like(norm), atol=0.1)


def test_pretrained_network_matches_diffcsp(pretrained):
    """CSPNet of DiffCSP 7121d15, computed once outside of the project."""
    structures, expected = reference()
    times = expected["times"]
    assert times[0] == 1 and times[-1] == TIMESTEPS

    lattice_noise, score = pretrained(pretrained.transform()(structures), times)
    assert close(lattice_noise, expected["pred_l"])
    assert close(score, expected["pred_x"])
    assert (lattice_noise - expected["pred_l"]).abs().mean() < 1e-6
    assert (score - expected["pred_x"]).abs().mean() < 1e-6


def test_pretrained_network_matches_diffcsp_on_noisy_structures(pretrained):
    """The same with the noise of the diffusion at the time of each structure."""
    structures, expected = reference()
    noisy = Structures(
        lattice=expected["noisy_l"],
        x=expected["noisy_x"],
        z=structures.z,
        num_atoms=structures.num_atoms,
    )

    lattice_noise, score = pretrained(pretrained.transform()(noisy), expected["times"])
    assert close(lattice_noise, expected["noisy_pred_l"])
    assert close(score, expected["noisy_pred_x"])
    assert (score - expected["noisy_pred_x"]).abs().mean() < 1e-6


def dynamics(expected: dict[str, torch.Tensor], structures: Structures, step: int):
    """Structures of the sampling of DiffCSP before one of its saved steps."""
    num_atoms = structures.num_atoms[: expected["dynamics_l"].shape[1]]

    return Structures(
        lattice=expected["dynamics_l"][step],
        x=expected["dynamics_x"][step],
        z=structures.z[: int(num_atoms.sum())],
        num_atoms=num_atoms,
    )


def test_pretrained_steps_match_diffcsp(pretrained):
    """Steps of the sampling of DiffCSP, each from its structures and its noise."""
    structures, expected = reference()
    times = expected["dynamics_times"].tolist()
    assert times[0] == TIMESTEPS and times[-1] == 1

    for step, time in enumerate(times):
        moved = pretrained.sample_step(
            dynamics(expected, structures, step),
            time,
            expected["corrector_noise_x"][step],
            expected["predictor_noise_l"][step],
            expected["predictor_noise_x"][step],
        )

        assert close(moved.lattice, expected["dynamics_next_l"][step]), time
        assert close_in_the_cell(moved.x, expected["dynamics_next_x"][step]), time

    # the last step gives the structures DiffCSP returns
    assert torch.equal(expected["dynamics_next_l"][-1], expected["final_l"])
    assert torch.equal(expected["dynamics_next_x"][-1], expected["final_x"])


def test_pretrained_dynamics_match_diffcsp(pretrained):
    """The first steps of the sampling of DiffCSP, one after the other."""
    structures, expected = reference()
    times = expected["dynamics_times"].tolist()
    assert times[:3] == [TIMESTEPS, TIMESTEPS - 1, TIMESTEPS - 2]

    current = dynamics(expected, structures, 0)
    for step in range(3):
        current = pretrained.sample_step(
            current,
            times[step],
            expected["corrector_noise_x"][step],
            expected["predictor_noise_l"][step],
            expected["predictor_noise_x"][step],
        )

    assert close(current.lattice, expected["dynamics_next_l"][2])
    assert close_in_the_cell(current.x, expected["dynamics_next_x"][2])


# the model which also generates the elements, with random weights


@pytest.fixture(scope="module")
def generation() -> DiffCspGenerationModel:
    """Model with random weights, and scores scaled as the ones of small noise."""
    torch.manual_seed(1)
    model = DiffCspGenerationModel().eval().requires_grad_(False)
    sigmas = model.sigma_scheduler.sigmas
    model.sigma_scheduler.sigmas_norm[1:] = 1.0 / sigmas[1:] ** 2

    return model


def noisy_elements(model: DiffCspGenerationModel, batch: Structures) -> torch.Tensor:
    """Vectors of the elements of structures, with some noise."""
    noise = torch.randn(len(batch.z), NUM_ELEMENTS, generator=generator(2))

    return model.element_vectors(batch.z) + 0.3 * noise


def test_element_vectors(generation):
    z = torch.tensor([1, 8, 100, 8])
    vectors = generation.element_vectors(z)

    assert vectors.shape == (4, NUM_ELEMENTS) and vectors.dtype == torch.float32
    assert torch.equal(vectors.sum(dim=1), torch.ones(4))
    assert vectors[0, 0] == 1 and vectors[1, 7] == 1 and vectors[2, 99] == 1
    assert torch.equal(generation.atomic_numbers(vectors), z)

    # the largest value, whatever the other ones
    assert torch.equal(generation.atomic_numbers(vectors * 0.2 - 3.0), z)

    for unknown in (0, 101):
        with pytest.raises(ValueError, match="not supported"):
            generation.element_vectors(torch.tensor([6, unknown]))


def test_generation_output(generation):
    structures = generation.transform()(make_batch())
    elements = noisy_elements(generation, structures)
    lattice_noise, score, element_noise = generation(structures, elements, TIMES)

    assert lattice_noise.shape == (2, 3, 3)
    assert score.shape == (len(structures.z), 3)
    assert element_noise.shape == (len(structures.z), NUM_ELEMENTS)
    for values in (lattice_noise, score, element_noise):
        assert values.dtype == torch.float32 and values.isfinite().all()
    assert element_noise.abs().max() > 0


def test_generation_does_not_read_the_atomic_numbers(generation):
    batch = make_batch()
    elements = noisy_elements(generation, batch)
    other = Structures(
        lattice=batch.lattice,
        x=batch.x,
        z=torch.ones_like(batch.z),
        num_atoms=batch.num_atoms,
    )

    expected = generation(generation.transform()(batch), elements, TIMES)
    found = generation(generation.transform()(other), elements, TIMES)
    assert all(torch.equal(a, b) for a, b in zip(found, expected))

    # the vectors are read
    changed = generation(generation.transform()(batch), elements.flip(1), TIMES)
    assert not torch.allclose(changed[2], expected[2], atol=1e-4)


def test_generation_batch_equals_single(generation):
    transform = generation.transform()
    batch = make_batch()
    elements = noisy_elements(generation, batch)

    alone = [
        generation(transform(structure), vectors, TIMES[i : i + 1])
        for i, (structure, vectors) in enumerate(
            zip(make_structures(), elements.split(batch.num_atoms.tolist()))
        )
    ]
    together = generation(transform(batch), elements, TIMES)

    for found, parts in zip(together, zip(*alone)):
        assert torch.allclose(found, torch.cat(parts), atol=1e-5)


def test_generation_follows_a_permutation_and_a_translation(generation):
    batch = make_batch()
    elements = noisy_elements(generation, batch)
    permutation = sampling.random_permutation(batch.batch, generator())
    moved = AtomTranslation(generator=generator())(
        actions.permute_atoms(batch, permutation)
    )

    lattice_noise, score, element_noise = generation(
        generation.transform()(batch), elements, TIMES
    )
    moved_noise, moved_score, moved_elements = generation(
        generation.transform()(moved), elements[permutation], TIMES
    )

    assert torch.allclose(moved_noise, lattice_noise, atol=1e-4)
    assert torch.allclose(moved_score, score[permutation], atol=1e-4)
    assert torch.allclose(moved_elements, element_noise[permutation], atol=1e-4)


def test_generation_needs_its_graph(generation):
    batch = make_batch()

    with pytest.raises(ValueError, match=r"apply DiffCspGenerationModel.transform\(\)"):
        generation(batch, noisy_elements(generation, batch), TIMES)


# sampling with the elements, with random weights


def generation_noise(batch: Structures, seed: int = 0) -> GenerationNoise:
    random = generator(seed)

    return GenerationNoise(
        corrector=torch.randn(batch.x.shape, generator=random),
        lattice=torch.randn(batch.lattice.shape, generator=random),
        elements=torch.randn(len(batch.z), NUM_ELEMENTS, generator=random),
        predictor=torch.randn(batch.x.shape, generator=random),
    )


def test_generation_sample_step(generation):
    batch = make_batch()
    elements = noisy_elements(generation, batch)
    time = 400
    noise = generation_noise(batch)
    quiet = GenerationNoise(
        *(torch.zeros_like(values) for values in vars(noise).values())
    )
    beta = generation.beta_scheduler

    moved, moved_elements = generation.sample_step(batch, elements, time, noise)
    assert moved.graph is None
    assert torch.equal(moved.num_atoms, batch.num_atoms)
    assert (moved.x >= 0).all() and (moved.x < 1).all()
    assert moved_elements.shape == elements.shape

    # the atomic numbers returned are read from the vectors returned
    assert torch.equal(moved.z, generation.atomic_numbers(moved_elements))

    # without correction and without noise, the lattice and the elements
    # follow the noise the network finds in them
    still, still_elements = generation.sample_step(
        batch, elements, time, quiet, step_size=0.0
    )
    found_lattice, _, found_elements = generation(
        generation.transform()(batch), elements, torch.full((2,), time)
    )
    scale = (1 - beta.alphas[time]) / (1 - beta.alphas_cumprod[time]).sqrt()
    assert torch.allclose(
        still.lattice,
        (batch.lattice - scale * found_lattice) / beta.alphas[time].sqrt(),
        atol=1e-5,
    )
    assert torch.allclose(
        still_elements,
        (elements - scale * found_elements) / beta.alphas[time].sqrt(),
        atol=1e-5,
    )

    # the noise of the elements is added with the deviation of the time, and
    # moves nothing else
    only_elements = GenerationNoise(
        quiet.corrector, quiet.lattice, noise.elements, quiet.predictor
    )
    other, other_elements = generation.sample_step(
        batch, elements, time, only_elements, step_size=0.0
    )
    assert torch.allclose(
        other_elements - still_elements, beta.sigmas[time] * noise.elements, atol=1e-5
    )
    assert torch.equal(other.x, still.x) and torch.equal(other.lattice, still.lattice)


def test_sample_num_atoms(generation):
    distribution = torch.tensor(NUM_ATOMS_DISTRIBUTION)
    assert len(distribution) == 21 and distribution[0] == 0
    assert torch.isclose(distribution.sum(), torch.tensor(1.0), atol=1e-6)

    num_atoms = generation.sample_num_atoms(20000, generator())
    assert num_atoms.shape == (20000,) and num_atoms.dtype == torch.int64
    assert num_atoms.min() >= 1 and num_atoms.max() <= 20

    frequency = torch.bincount(num_atoms, minlength=21) / 20000
    assert (frequency - distribution).abs().max() < 0.01

    again = generation.sample_num_atoms(20000, generator())
    assert torch.equal(again, num_atoms)


def test_generation_sample(generation, monkeypatch):
    # the last steps of the diffusion only: all of them take seconds
    monkeypatch.setattr(diffcsp_model, "TIMESTEPS", 20)
    num_atoms = torch.tensor([2, 1, 4])

    structures = generation.sample(num_atoms, generator=generator())
    assert structures.graph is None
    assert structures.lattice.shape == (3, 3, 3)
    assert structures.x.shape == (7, 3) and structures.z.shape == (7,)
    assert torch.equal(structures.num_atoms, num_atoms)
    assert (structures.x >= 0).all() and (structures.x < 1).all()
    assert ((structures.z >= 1) & (structures.z <= NUM_ELEMENTS)).all()

    # the same draws give the same structures
    again = generation.sample(num_atoms, generator=generator())
    assert torch.equal(again.x, structures.x) and torch.equal(again.z, structures.z)
    other = generation.sample(num_atoms, generator=generator(1))
    assert not torch.equal(other.x, structures.x)


# the weights of the two models


def test_generation_load_pretrained_checks_the_parameters(generation, model):
    state = dict(generation.state_dict())
    assert "num_atoms_distribution" not in state
    hparams = upstream_hparams(GENERATION_UPSTREAM_ARGS)

    # the checkpoint has a schedule the model does not have
    state["type_sigma_scheduler.sigmas"] = torch.zeros(TIMESTEPS + 1)
    state["type_sigma_scheduler.sigmas_norm"] = torch.ones(TIMESTEPS + 1)

    other = DiffCspGenerationModel()
    other.load_pretrained(state, hparams)
    other.eval()
    with torch.no_grad():
        structures = generation.transform()(make_batch())
        elements = noisy_elements(generation, structures)
        found = other(structures, elements, TIMES)
        expected = generation(structures, elements, TIMES)
    assert all(torch.equal(a, b) for a, b in zip(found, expected))

    # each model refuses the weights of the other one, by their configuration
    with pytest.raises(InvalidWeights, match="another configuration") as error:
        other.load_pretrained(state, upstream_hparams())
    assert "model._target_" in str(error.value)
    with pytest.raises(InvalidWeights, match="another configuration"):
        DiffCspModel().load_pretrained(dict(model.state_dict()), hparams)

    # and by their parameters
    with pytest.raises(InvalidWeights, match="type_out|node_embedding"):
        other.load_pretrained(dict(model.state_dict()))
    with pytest.raises(InvalidWeights, match="type_out|node_embedding"):
        DiffCspModel().load_pretrained(state)

    del state["decoder.type_out.weight"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state)


# the pretrained model of the generation against the real DiffCSP


def generation_reference() -> tuple[Structures, dict[str, torch.Tensor]]:
    """Structures of the tests and what the generation of DiffCSP computed."""
    structures, _ = reference()
    expected = {
        name: torch.from_numpy(values)
        for name, values in np.load(DATA / "diffcsp_generation_reference.npz").items()
    }

    return structures, expected


def close_after(found: torch.Tensor, expected: torch.Tensor, scale: float) -> bool:
    """Whether values are the expected ones after a step which scales the rounding.

    A step of the sampling divides what it predicts by the square root of the
    alpha of its time, 0.01 at the last time: the difference is under 1e-5 of
    the largest expected value before this division.
    """
    largest = max(1.0, float(expected.abs().max()))

    return float((found - expected).abs().max()) < 1e-5 * largest * max(1.0, scale)


@pytest.fixture(scope="module")
def pretrained_generation() -> DiffCspGenerationModel:
    paths = weights.checkpoint_paths("diffcsp", GENERATION_NAME)
    if not all(path.exists() for path in paths):
        pytest.skip(f"weights of {GENERATION_NAME} are not in the cache")

    return DiffCspGenerationModel.from_pretrained(GENERATION_NAME, device="cpu")


def test_pretrained_generation_schedules(pretrained_generation, pretrained):
    """The two models have the same schedules, but for the random draws."""
    ours, other = pretrained_generation, pretrained

    assert torch.equal(ours.beta_scheduler.betas, other.beta_scheduler.betas)
    assert torch.equal(ours.beta_scheduler.sigmas, other.beta_scheduler.sigmas)
    assert torch.equal(ours.sigma_scheduler.sigmas, other.sigma_scheduler.sigmas)

    norm, other_norm = (
        ours.sigma_scheduler.sigmas_norm,
        other.sigma_scheduler.sigmas_norm,
    )
    assert not torch.equal(norm, other_norm)
    assert torch.allclose(norm, other_norm, rtol=0.1)


def test_pretrained_generation_network_matches_diffcsp(pretrained_generation):
    """CSPNet of the generation of DiffCSP 7121d15, on noisy structures."""
    structures, expected = generation_reference()
    noisy = Structures(
        lattice=expected["noisy_l"],
        x=expected["noisy_x"],
        z=structures.z,
        num_atoms=structures.num_atoms,
    )

    lattice_noise, score, element_noise = pretrained_generation(
        pretrained_generation.transform()(noisy), expected["noisy_t"], expected["times"]
    )
    assert close(lattice_noise, expected["noisy_pred_l"])
    assert close(score, expected["noisy_pred_x"])
    assert close(element_noise, expected["noisy_pred_t"])
    assert (element_noise - expected["noisy_pred_t"]).abs().mean() < 1e-6
    assert (score - expected["noisy_pred_x"]).abs().mean() < 1e-6


def generation_dynamics(
    expected: dict[str, torch.Tensor], structures: Structures, step: int
) -> tuple[Structures, torch.Tensor, GenerationNoise]:
    """What the sampling of DiffCSP had before one of its saved steps, and its noise."""
    num_atoms = structures.num_atoms[expected["dynamics_structures"]]
    elements = expected["dynamics_t"][step]
    start = Structures(
        lattice=expected["dynamics_l"][step],
        x=expected["dynamics_x"][step],
        z=elements.argmax(dim=-1) + 1,
        num_atoms=num_atoms,
    )
    noise = GenerationNoise(
        corrector=expected["corrector_noise_x"][step],
        lattice=expected["predictor_noise_l"][step],
        elements=expected["predictor_noise_t"][step],
        predictor=expected["predictor_noise_x"][step],
    )

    return start, elements, noise


def test_pretrained_generation_steps_match_diffcsp(pretrained_generation):
    """Steps of the generation of DiffCSP, each from its structures and its noise."""
    structures, expected = generation_reference()
    times = expected["dynamics_times"].tolist()
    assert times[0] == TIMESTEPS and times[-1] == 1
    alphas = pretrained_generation.beta_scheduler.alphas

    for step, time in enumerate(times):
        start, elements, noise = generation_dynamics(expected, structures, step)
        moved, elements = pretrained_generation.sample_step(
            start, elements, time, noise
        )

        # the first step divides by 0.01: see close_after
        scale = float(1 / alphas[time].sqrt())
        assert close_after(
            moved.lattice, expected["dynamics_next_l"][step], scale
        ), time
        assert close_after(elements, expected["dynamics_next_t"][step], scale), time
        assert close_in_the_cell(moved.x, expected["dynamics_next_x"][step]), time
        if time < TIMESTEPS:
            assert scale < 2.1

    # the last step gives the structures and the elements DiffCSP returns
    assert torch.equal(moved.z, expected["final_z"])
    assert torch.equal(expected["dynamics_next_l"][-1], expected["final_l"])
    assert torch.equal(expected["dynamics_next_x"][-1], expected["final_x"])


def test_pretrained_generation_dynamics_match_diffcsp(pretrained_generation):
    """Two steps of the generation of DiffCSP, one after the other."""
    structures, expected = generation_reference()
    times = expected["dynamics_times"].tolist()
    assert times[1:3] == [TIMESTEPS - 1, TIMESTEPS - 2]

    current, elements, _ = generation_dynamics(expected, structures, 1)
    for step in (1, 2):
        _, _, noise = generation_dynamics(expected, structures, step)
        current, elements = pretrained_generation.sample_step(
            current, elements, times[step], noise
        )

    assert close(current.lattice, expected["dynamics_next_l"][2])
    assert close(elements, expected["dynamics_next_t"][2])
    assert close_in_the_cell(current.x, expected["dynamics_next_x"][2])
