"""Tests for the rewrite of the base model of MatterGen."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from helpers.compare import (
    as_near_as_original,
    assert_close,
    assert_close_in_the_cell,
    rounding_spread,
    within_rounding,
)
from helpers.graphs import neighbours_at_the_same_distance as ties
from helpers.references import load_reference, reference_structures
from helpers.structures import generator, noisy
from helpers.structures import make_batch as make_batch_of
from helpers.structures import make_structures as make_structures_of
from helpers.weights import pretrained as load_pretrained
from torchms.data import Structures
from torchms.data.augmentation import (
    AtomTranslation,
    LatticeRotation,
    actions,
    sampling,
)
from torchms.graph.geometry import lattice_heights, symmetric_lattice
from torchms.graph.line_graph import reverse_edges
from torchms.graph.sparse import count_by
from torchms.graph.transforms import (
    AddRadiusGraph,
)
from torchms.models.estimator import weights
from torchms.models.layers import sinusoidal_frequencies
from torchms.models.layers import gemnet
from torchms.models.mattergen import (
    Config,
    Denoised,
    Guidance,
    InvalidWeights,
    MatterGen,
    MatterGenBandGap,
    MatterGenBulkModulus,
    MatterGenChemicalSystem,
    MatterGenChemicalSystemEnergyAboveHull,
    MatterGenMagneticDensity,
    MatterGenMagneticDensityHHIScore,
    MatterGenSpaceGroup,
    StepNoise,
)
from torchms.models.mattergen.diffusion import guided_scores, symmetric_noise
from torchms.models.mattergen.properties import NoiseLevelEncoding, StandardScaler

NAME = "mattergen-base"
MP_20_NAME = "mattergen-mp-20-base"
CHEMICAL_SYSTEM_NAME = "mattergen-chemical-system"
SPACE_GROUP_NAME = "mattergen-space-group"
BAND_GAP_NAME = "mattergen-dft-band-gap"
MAGNETIC_DENSITY_NAME = "mattergen-dft-mag-density"
ENERGY_ABOVE_HULL_NAME = "mattergen-chemical-system-energy-above-hull"
BULK_MODULUS_NAME = "mattergen-ml-bulk-modulus"
HHI_SCORE_NAME = "mattergen-dft-mag-density-hhi-score"

# a network of 512 features on 50 neighbours per atom and their triplets:
# slower than the default timeout
pytestmark = pytest.mark.timeout(120)


def make_structures() -> list[Structures]:
    return make_structures_of(cif="triclinic2.cif")


def make_batch() -> Structures:
    return make_batch_of(cif="triclinic2.cif")


def narrow(config: Config | None = None) -> Config:
    """Configuration of the same network with fewer features and blocks."""
    config = config or Config.load()

    return replace(
        config,
        model=replace(config.model, hidden_dim=64),
        decoder={**config.decoder, "num_blocks": 2},
    )


def relatively_close(found: torch.Tensor, expected: torch.Tensor, rtol: float) -> bool:
    """Whether values are near the expected ones, relatively to the largest one.

    The scores of the lattice of random weights are of order 1e-6.
    """
    return float((found - expected).abs().max()) <= rtol * float(expected.abs().max())


@pytest.fixture(scope="module")
def model() -> MatterGen:
    """Model with random weights, narrower than the pretrained one: as fast as possible."""
    torch.manual_seed(0)
    return MatterGen(narrow()).eval().requires_grad_(False)


TIMES = torch.tensor([0.05, 0.7])


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS["mattergen"]) == {
        NAME,
        MP_20_NAME,
        CHEMICAL_SYSTEM_NAME,
        SPACE_GROUP_NAME,
        BAND_GAP_NAME,
        MAGNETIC_DENSITY_NAME,
        ENERGY_ABOVE_HULL_NAME,
        BULK_MODULUS_NAME,
        HHI_SCORE_NAME,
    }
    assert Config.load().default_weights == NAME
    assert Config.load().chemical_system.default_weights == CHEMICAL_SYSTEM_NAME
    assert MatterGenChemicalSystem().default_weights() == CHEMICAL_SYSTEM_NAME
    assert MatterGenSpaceGroup().default_weights() == SPACE_GROUP_NAME
    assert MatterGenBandGap().default_weights() == BAND_GAP_NAME
    assert MatterGenMagneticDensity().default_weights() == MAGNETIC_DENSITY_NAME
    joint = MatterGenChemicalSystemEnergyAboveHull()
    assert joint.default_weights() == ENERGY_ABOVE_HULL_NAME
    assert MatterGenBulkModulus().default_weights() == BULK_MODULUS_NAME
    assert MatterGenMagneticDensityHHIScore().default_weights() == HHI_SCORE_NAME

    for name, directory in (
        (NAME, "mattergen_base"),
        (MP_20_NAME, "mp_20_base"),
        (CHEMICAL_SYSTEM_NAME, "chemical_system"),
        (SPACE_GROUP_NAME, "space_group"),
        (BAND_GAP_NAME, "dft_band_gap"),
        (MAGNETIC_DENSITY_NAME, "dft_mag_density"),
        (ENERGY_ABOVE_HULL_NAME, "chemical_system_energy_above_hull"),
        (BULK_MODULUS_NAME, "ml_bulk_modulus"),
        (HHI_SCORE_NAME, "dft_mag_density_hhi_score"),
    ):
        for checkpoint in weights.CHECKPOINTS["mattergen"][name]:
            # a commit of the repository of the authors, not a branch
            assert checkpoint.url.startswith(
                "https://huggingface.co/microsoft/mattergen/resolve/"
                f"5244495dd9a979ff71abc7548a0b14b9deb0069a/checkpoints/{directory}/"
            )
            assert len(checkpoint.sha256) == 64

        paths = weights.checkpoint_paths("mattergen", name, tmp_path)
        assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
            f"mattergen/{name}/last.ckpt",
            f"mattergen/{name}/config.yaml",
        ]


def test_configuration():
    config = Config.load()
    network = config.model
    sections = config.upstream.sections

    # the classes of the diffusion of the elements are the elements and the mask
    d3pm = "lightning_module.diffusion_module.corruption.discrete_corruptions"
    assert sections[f"{d3pm}.atomic_numbers.d3pm"]["dim"] == network.num_classes
    assert network.mask == network.num_elements + 1
    assert config.elements.num_steps == config.sampling.num_steps

    # elements up to bismuth, without the noble gases, Tc and Pm
    allowed = set(network.allowed_elements)
    assert len(allowed) == len(network.allowed_elements) == 76
    assert max(allowed) == 83 and not allowed & {2, 10, 18, 36, 43, 54, 61}

    assert len(config.sampling.num_atoms_distribution) == 21
    assert config.sampling.num_atoms_distribution[0] == 0
    assert sum(config.sampling.num_atoms_distribution) == pytest.approx(1.0)


def test_output(model):
    structures = model.transform()(make_batch())
    denoised = model(structures, TIMES)

    assert denoised.x.shape == (len(structures.z), 3)
    assert denoised.lattice.shape == (2, 3, 3)
    assert denoised.logits.shape == (len(structures.z), 101)
    for values in (denoised.x, denoised.lattice, denoised.logits):
        assert values.dtype == torch.float32 and values.isfinite().all()

    # the score of the lattice is symmetric, the elements not allowed and the
    # mask are lowered
    assert torch.equal(denoised.lattice, denoised.lattice.transpose(1, 2))
    assert denoised.lattice.abs().max() > 0
    allowed = torch.tensor(Config.load().model.allowed_elements) - 1
    lowered = torch.ones(101, dtype=torch.bool)
    lowered[allowed] = False
    assert lowered[-1]
    assert (denoised.logits[:, lowered] < -1e9).all()
    assert (denoised.logits[:, ~lowered].abs() < 1e3).all()


def test_graph():
    config = Config.load()
    structures = MatterGen.transform()(noisy(make_batch()))
    graph = structures.graph

    assert (graph.edges_distances() <= config.graph.radius).all()
    assert graph.edge_shift.abs().max() <= config.graph.max_image_shift
    assert (reverse_edges(graph) >= 0).all()
    # made symmetric from 50 neighbours per atom: about as many
    degree = count_by(graph.senders, len(structures.z)).float()
    assert (
        0.5 * config.graph.max_neighbors
        < degree.mean()
        < 1.5 * config.graph.max_neighbors
    )

    bonds, triplets = structures.bonds.index, structures.triplets.index
    assert sorted(bonds.flatten().tolist()) == list(range(int(graph.num_edges.sum())))
    assert (graph.senders[triplets[0]] == graph.senders[triplets[1]]).all()
    assert (triplets[0] != triplets[1]).all()


def test_graph_of_a_flat_cell(model):
    # the sampling meets flat cells: their images are bounded before they are
    # listed, as in MatterGen, and the network still gives numbers
    flat = Structures(
        lattice=torch.diag(torch.tensor([4.0, 4.0, 1e-3]))[None],
        x=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        z=[8, 26],
    )
    structures = model.transform()(flat)

    assert structures.graph.edge_shift.abs().max() <= 5
    assert model(structures, TIMES[:1]).logits.isfinite().all()


def test_depends_on_the_time(model):
    structures = model.transform()(make_batch())
    denoised = model(structures, TIMES)
    other = model(structures, torch.tensor([0.05, 0.2]))

    first = structures.batch == 0
    assert torch.equal(other.x[first], denoised.x[first])
    assert torch.equal(other.lattice[0], denoised.lattice[0])
    assert not torch.allclose(other.x[~first], denoised.x[~first])
    assert not relatively_close(other.lattice[1], denoised.lattice[1], 1e-3)


def test_batch_equals_single(model):
    transform = model.transform()
    singles = [
        model(transform(structure), TIMES[i : i + 1])
        for i, structure in enumerate(make_structures())
    ]
    denoised = model(transform(make_batch()), TIMES)

    for name in ("x", "logits"):
        expected = torch.cat([getattr(single, name) for single in singles])
        assert torch.allclose(getattr(denoised, name), expected, atol=1e-4), name
    expected = torch.cat([single.lattice for single in singles])
    assert relatively_close(denoised.lattice, expected, 1e-4)


def test_follows_a_rotation(model):
    batch = noisy(make_batch())
    rotated = LatticeRotation(generator=generator())(batch)

    denoised = model(model.transform()(batch), TIMES)
    turned = model(model.transform()(rotated), TIMES)

    # the scores of the coordinates are fractional and the elements do not
    # depend on the orientation; the score of the lattice turns with it
    assert torch.allclose(turned.x, denoised.x, atol=1e-4)
    assert torch.allclose(turned.logits, denoised.logits, atol=1e-3)
    rotation = torch.linalg.solve(batch.lattice, rotated.lattice)
    expected = rotation.transpose(1, 2) @ denoised.lattice @ rotation
    assert relatively_close(turned.lattice, expected, 1e-3)


def test_translation_invariance(model):
    batch = noisy(make_batch())
    translated = AtomTranslation(generator=generator())(batch)
    assert not torch.allclose(translated.x, batch.x)

    denoised = model(model.transform()(batch), TIMES)
    moved = model(model.transform()(translated), TIMES)

    assert torch.allclose(moved.x, denoised.x, atol=1e-3)
    assert torch.allclose(moved.logits, denoised.logits, atol=1e-3)
    assert relatively_close(moved.lattice, denoised.lattice, 1e-3)


def test_follows_a_permutation(model):
    batch = noisy(make_batch())
    permutation = sampling.random_permutation(batch.batch, generator())
    permuted = actions.permute_atoms(batch, permutation)
    assert not torch.equal(permuted.z, batch.z)

    # on a graph where no neighbour is left out, the network follows the order
    config = Config.load()
    config = replace(config, graph=replace(config.graph, max_neighbors=1000))
    transform = model.transform(config)
    denoised = model(transform(batch), TIMES)
    other = model(transform(permuted), TIMES)

    assert relatively_close(other.x, denoised.x[permutation], 1e-3)
    assert relatively_close(other.lattice, denoised.lattice, 1e-3)
    assert torch.allclose(other.logits, denoised.logits[permutation], atol=1e-3)

    # with 50 neighbours, the graph is made symmetric from the edges towards
    # atoms of lower index, as in MatterGen: it depends on the order
    graph = model.transform()(batch).graph
    other_graph = model.transform()(permuted).graph
    assert not torch.equal(graph.num_edges, other_graph.num_edges)


def test_reads_the_mask(model):
    batch = make_batch()
    masked = replace(batch, z=torch.where(batch.batch == 1, model.mask, batch.z))

    denoised = model(model.transform()(batch), TIMES)
    other = model(model.transform()(masked), TIMES)
    assert not torch.allclose(other.logits, denoised.logits)


def test_needs_its_graphs(model):
    with pytest.raises(ValueError, match=r"apply MatterGen.transform\(\)"):
        model(make_batch(), TIMES)

    with pytest.raises(ValueError, match="bonds"):
        model(AddRadiusGraph(3.0)(make_batch()), TIMES)


def test_unknown_element(model):
    # the mask is the atomic number 101, there is nothing after it
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[102]
    )

    with pytest.raises(ValueError, match="not supported"):
        model(model.transform()(structures), TIMES[:1])


# the diffusions


def test_embedding_of_the_time(model):
    # as MatterGen: a sine then a cosine for each frequency
    embedding = model.embed_time(torch.tensor([0.0, 0.5]))
    frequencies = sinusoidal_frequencies(64, 10000.0, endpoint=False)

    assert embedding.shape == (2, 64)
    assert torch.equal(embedding[0, 0::2], torch.zeros(32))
    assert torch.equal(embedding[0, 1::2], torch.ones(32))
    assert torch.allclose(embedding[1, 0::2], torch.sin(0.5 * frequencies))
    assert torch.equal(model.time_frequencies, frequencies)


def test_diffusion_of_the_coordinates(model):
    num_atoms = torch.tensor([1, 8])
    start = model.coordinate_deviation(torch.zeros(2), num_atoms)
    end = model.coordinate_deviation(torch.ones(2), num_atoms)

    assert torch.allclose(start, torch.tensor([0.01, 0.005]))
    assert torch.allclose(end, torch.tensor([5.0, 2.5]))


def test_diffusion_of_the_lattice(model):
    config = Config.load().lattice
    num_atoms = torch.tensor([1, 8])
    times = torch.tensor([0.0, 1.0])

    coefficient = model.lattice_mean_coefficient(times)
    expected = torch.exp(-torch.tensor([0.0, 0.25 * 19.9 + 0.05]))
    assert torch.allclose(coefficient, expected)

    # a cubic cell of the mean density, with a deviation which grows as the
    # cube root of the number of atoms
    mean, variance = model.lattice_limit(num_atoms)
    volume = torch.linalg.det(mean)
    assert torch.allclose(volume, num_atoms / config.limit_density, rtol=1e-5)
    assert torch.allclose(variance[:, 0, 1], 0.25 * num_atoms ** (2 / 3))

    deviation = model.lattice_deviation(times, num_atoms)
    assert torch.equal(deviation[0], torch.zeros(3, 3))
    assert torch.allclose(deviation[1], (variance[1] * (1 - expected[1] ** 2)).sqrt())


def test_diffusion_of_the_elements(model):
    # an atom is masked with a probability t / N
    steps = Config.load().elements.num_steps
    expected = 1 - torch.arange(steps + 1, dtype=torch.float64) / steps

    assert model.mask_kept.shape == (steps + 1,)
    assert torch.allclose(model.mask_kept.double(), expected, atol=1e-6)


def test_times(model):
    times, step = model.times(), model.time_step()

    assert times.shape == (1000,)
    assert times[0] == 1.0 and times[-1] == pytest.approx(1e-3)
    assert torch.allclose(times[1:] - times[:-1], step.expand(999), atol=1e-6)
    # after the last step, the time 0
    assert times[-1] + step <= 0


def test_symmetric_noise():
    noise = torch.randn(1000, 3, 3, generator=generator())
    symmetric = symmetric_noise(noise)

    assert torch.equal(symmetric, symmetric.transpose(1, 2))
    # with the variance of the noise
    assert symmetric.var(dim=0).sub(1).abs().max() < 0.2


def test_symmetric_lattice():
    batch = make_batch()
    symmetric = symmetric_lattice(batch.lattice)

    assert torch.allclose(symmetric, symmetric.transpose(1, 2), atol=1e-5)
    # the same cell, rotated
    gram = batch.lattice @ batch.lattice.transpose(1, 2)
    assert torch.allclose(symmetric @ symmetric.transpose(1, 2), gram, atol=1e-3)
    assert torch.allclose(lattice_heights(symmetric), lattice_heights(batch.lattice))


def test_element_posterior(model):
    mask = model.mask
    logits = torch.randn(3, 101, generator=generator())
    logits[:, -1] = -1e10
    probabilities = torch.softmax(logits, dim=-1)
    z = torch.tensor([mask, 26, mask])
    steps = torch.tensor([500, 500, 0])

    posterior = torch.softmax(model.element_posterior(z, logits, steps), dim=-1)

    # an element stays, a masked atom can stay masked or take an element in
    # proportion of its probability at the time 0
    assert posterior[1, 25] == pytest.approx(1.0)
    masked = posterior[0, :-1] / posterior[0, :-1].sum()
    assert torch.allclose(masked, probabilities[0, :-1], atol=1e-6)
    assert posterior[0, -1] == pytest.approx(500 / 501, abs=1e-4)

    # at the first step, a masked atom takes an element
    assert posterior[2, -1] < 1e-6
    assert torch.allclose(posterior[2, :-1], probabilities[2, :-1], atol=1e-5)


# sampling with random weights


def step_noise(batch: Structures, elements: bool = True, seed: int = 0) -> StepNoise:
    random = generator(seed)
    n_struct, n_atoms = len(batch.num_atoms), len(batch.z)

    return StepNoise(
        corrector_x=torch.randn((n_atoms, 3), generator=random),
        corrector_lattice=torch.randn((n_struct, 3, 3), generator=random),
        predictor_x=torch.randn((n_atoms, 3), generator=random),
        predictor_lattice=torch.randn((n_struct,), generator=random),
        elements=(
            -torch.log(-torch.log(torch.rand((n_atoms, 101), generator=random)))
            if elements
            else None
        ),
    )


def test_correct(model):
    batch = noisy(make_batch())
    corrected = model.correct(batch, 500, step_noise(batch))

    assert corrected.graph is None
    assert torch.equal(corrected.z, batch.z)
    assert torch.allclose(corrected.lattice, corrected.lattice.transpose(1, 2))
    assert (corrected.x >= 0).all() and (corrected.x < 1).all()
    assert not torch.allclose(corrected.x, batch.x)


def test_corrector_couples_the_crystals(model):
    # the step of the Langevin corrector is computed from the whole batch, as
    # in MatterGen: a crystal moves differently with other crystals
    structures = [noisy(structure) for structure in make_structures()]
    batch = Structures.collate(structures)
    noise = step_noise(batch)
    first = batch.batch == 0
    alone = StepNoise(
        corrector_x=noise.corrector_x[first],
        corrector_lattice=noise.corrector_lattice[:1],
        predictor_x=noise.predictor_x[first],
        predictor_lattice=noise.predictor_lattice[:1],
        elements=None,
    )

    together = model.correct(batch, 500, noise)
    single = model.correct(structures[0], 500, alone)
    assert not torch.allclose(together.x[first], single.x, atol=1e-4)


def test_predict(model):
    batch = noisy(make_batch())
    masked = replace(batch, z=torch.where(batch.batch == 1, model.mask, batch.z))
    predicted = model.predict(masked, 500, step_noise(masked))

    assert predicted.graph is None
    assert (predicted.x >= 0).all() and (predicted.x < 1).all()
    assert predicted.lattice.isfinite().all()
    # an atom with an element keeps it
    assert torch.equal(predicted.z[batch.batch == 0], batch.z[batch.batch == 0])

    # after the last step: no noise, and every atom has an allowed element
    last = model.predict(masked, 999, step_noise(masked, elements=False))
    again = model.predict(
        masked, 999, replace(step_noise(masked, seed=1), elements=None)
    )
    assert torch.equal(last.x, again.x)
    assert torch.equal(last.lattice, again.lattice)
    assert torch.equal(last.z, again.z)
    allowed = torch.tensor(Config.load().model.allowed_elements)
    assert torch.isin(last.z, allowed).all()


def short_diffusion(steps: int = 4) -> Config:
    """Configuration of a diffusion of a few steps: all of them take seconds."""
    config = narrow()

    return replace(
        config,
        elements=replace(config.elements, num_steps=steps),
        sampling=replace(config.sampling, num_steps=steps),
    )


def test_a_model_follows_its_configuration():
    config = short_diffusion()
    short = MatterGen(config)

    assert short.config is config
    assert short.times().shape == (4,) and short.mask_kept.shape == (5,)
    assert MatterGen.transform(config) == MatterGen.transform()

    # without argument, the one of config.yaml
    assert MatterGen().config == Config.load()

    # the weights of the model of the file do not fit
    with pytest.raises(InvalidWeights, match="size mismatch"):
        short.load_pretrained(upstream_state(MatterGen()))


def test_sample():
    torch.manual_seed(0)
    model = MatterGen(short_diffusion()).eval().requires_grad_(False)
    num_atoms = torch.tensor([3, 1])

    structures = model.sample(num_atoms, generator=generator())
    assert structures.graph is None
    assert torch.equal(structures.num_atoms, num_atoms)
    assert (structures.x >= 0).all() and (structures.x < 1).all()
    assert structures.lattice.isfinite().all()
    allowed = torch.tensor(Config.load().model.allowed_elements)
    assert torch.isin(structures.z, allowed).all()

    # the same draws give the same crystals
    again = model.sample(num_atoms, generator=generator())
    assert torch.equal(again.x, structures.x)
    assert torch.equal(again.lattice, structures.lattice)
    assert torch.equal(again.z, structures.z)
    other = model.sample(num_atoms, generator=generator(1))
    assert not torch.equal(other.x, structures.x)


def test_sample_prior(model):
    num_atoms = torch.tensor([2, 8])
    structures = model.sample_prior(num_atoms, generator=generator())

    assert (structures.z == model.mask).all()
    assert (structures.x >= 0).all() and (structures.x < 1).all()
    assert torch.equal(structures.lattice, structures.lattice.transpose(1, 2))


def test_sample_num_atoms(model):
    num_atoms = model.sample_num_atoms(2000, generator=generator())

    assert num_atoms.shape == (2000,)
    assert num_atoms.min() >= 1 and num_atoms.max() <= 20
    # 4 and 12 are the most frequent, 1 and 2 are rare
    counts = torch.bincount(num_atoms, minlength=21)
    assert counts[4] > 200 and counts[12] > 200 and counts[1] + counts[2] < 30


# the weights


def nested(sections: dict[str, dict[str, object]]) -> dict[str, object]:
    """Configuration of a training with these sections, named with dots."""
    configuration: dict[str, object] = {}
    for name, arguments in sections.items():
        section = configuration
        for key in name.split("."):
            section = section.setdefault(key, {})  # type: ignore[assignment]
        section.update(arguments)  # type: ignore[attr-defined]

    return configuration


def upstream_state(model: MatterGen) -> dict[str, torch.Tensor]:
    """Parameters of a model under the names of MatterGen."""
    upstream = Config.load().upstream
    # the renames of the property of the model: two properties share names
    renamed = {new: old for old, new in upstream.renamed_parameters.items()}
    state = {}
    for name, values in model.state_dict().items():
        for new, old in renamed.items():
            if name.endswith(new):
                name = name[: -len(new)] + old
        state[upstream.prefix + name] = values

    return state


def test_load_pretrained_checks_the_parameters(model):
    state = upstream_state(model)
    configuration = nested(Config.load().upstream.sections)
    assert "diffusion_module.model.noise_level_encoding.div_term" in state
    assert "diffusion_module.model.gemnet.radial_basis.rbf.offset" in state

    other = MatterGen(model.config)
    other.load_pretrained(state, configuration)
    other.eval()
    with torch.no_grad():
        structures = model.transform()(make_batch())
        assert torch.equal(other(structures, TIMES).x, model(structures, TIMES).x)

    # a model conditioned on a property
    sections = dict(Config.load().upstream.sections)
    sections["lightning_module.diffusion_module.model"] = {
        **sections["lightning_module.diffusion_module.model"],
        "property_embeddings": {"dft_band_gap": {}},
    }
    with pytest.raises(InvalidWeights, match="another configuration") as error:
        other.load_pretrained(state, nested(sections))
    assert "property_embeddings" in str(error.value)

    with pytest.raises(InvalidWeights, match="no lightning_module"):
        other.load_pretrained(state, {})

    with pytest.raises(InvalidWeights, match="outside of"):
        other.load_pretrained({**state, "ema.weight": torch.zeros(1)})

    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained(
            {**state, "diffusion_module.model.property_embeddings.w": torch.zeros(1)}
        )

    del state["diffusion_module.model.fc_atom.bias"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state)


# the model conditioned on a chemical system, with random weights


@pytest.fixture(scope="module")
def conditional() -> MatterGenChemicalSystem:
    torch.manual_seed(0)
    return MatterGenChemicalSystem(narrow()).eval().requires_grad_(False)


def test_guidance(conditional):
    guidance = conditional.guidance(["Zn-S", "Li-Fe-O"], scale=1.5)

    (systems,) = guidance.values
    assert systems.shape == (2, 101)
    assert systems.nonzero().tolist() == [
        [0, 16],
        [0, 30],
        [1, 3],
        [1, 8],
        [1, 26],
    ]
    assert guidance.scale == 1.5
    assert conditional.guidance(["O"]).scale == 2.0

    for unknown in ("Li-Xx", "Li-Og", "li"):
        with pytest.raises(ValueError, match="not supported"):
            conditional.guidance([unknown])


def test_conditional_network(conditional):
    structures = conditional.transform()(make_batch())
    systems = conditional.guidance(["Zn-S", "Zn-S"]).values

    free = conditional(structures, TIMES)
    mixed = conditional(structures, TIMES, systems, torch.tensor([True, False]))
    first = structures.batch == 0

    # the second crystal is not conditioned: its scores are the ones without
    # chemical system; the first one is, and only keeps zinc and sulfur
    assert torch.equal(mixed.x[~first], free.x[~first])
    assert torch.equal(mixed.lattice[1], free.lattice[1])
    assert torch.equal(mixed.logits[~first], free.logits[~first])
    assert not torch.allclose(mixed.x[first], free.x[first])
    kept = (mixed.logits[first] > -1e9).nonzero()[:, 1].unique() + 1
    assert kept.tolist() == [16, 30]
    assert (free.logits[first] > -1e9).sum(dim=1).eq(76).all()


def test_the_base_model_reads_no_chemical_system(model):
    structures = model.transform()(make_batch())
    systems = torch.zeros(2, 101)

    with pytest.raises(ValueError, match="not conditioned"):
        model(structures, TIMES, (systems,))

    with pytest.raises(ValueError, match="one Condition each"):
        model.gemnet(
            model.embed_time(TIMES),
            model.element_rows(structures.z),
            structures.graph,
            structures.triplets,
            structures.bonds,
            structures.graph.edges_vectors(),
            torch.zeros(structures.triplets.index.shape[1]),
            structures.lattice,
            (gemnet.Condition(torch.zeros(2, 64), torch.ones(2, dtype=torch.bool)),),
        )


def test_guided_scores(conditional):
    structures = conditional.transform()(noisy(make_batch()))
    systems = conditional.guidance(["Zn-S", "Si"]).values

    free = conditional.scores(structures, TIMES)
    with_system = conditional.unguided_scores(structures, TIMES, systems)
    assert not torch.allclose(with_system.x, free.x)

    # 0 is the model without condition, 1 the model with it
    for scale, expected in ((0.0, free), (1.0, with_system)):
        found = conditional.scores(structures, TIMES, Guidance(systems, scale))
        assert torch.equal(found.x, expected.x)
        assert torch.equal(found.logits, expected.logits)

    guided = conditional.scores(structures, TIMES, Guidance(systems, 2.0))
    assert torch.allclose(guided.x, 2 * with_system.x - free.x, atol=1e-5)
    assert torch.allclose(guided.lattice, 2 * with_system.lattice - free.lattice)
    # the elements outside the system stay lowered
    first = structures.batch == 0
    kept = (guided.logits[first] > -1e9).nonzero()[:, 1].unique() + 1
    assert kept.tolist() == [16, 30]


def test_conditional_sample():
    torch.manual_seed(0)
    model = MatterGenChemicalSystem(short_diffusion()).eval().requires_grad_(False)
    num_atoms = torch.tensor([3, 2])
    guidance = model.guidance(["Li-O", "Fe-Ni"])

    structures = model.sample(num_atoms, generator=generator(), guidance=guidance)
    first = structures.batch == 0
    assert set(structures.z[first].tolist()) <= {3, 8}
    assert set(structures.z[~first].tolist()) <= {26, 28}

    # without guidance, the conditioned model samples without condition
    free = model.sample(num_atoms, generator=generator())
    assert not torch.equal(free.x, structures.x)

    with pytest.raises(ValueError, match="not conditioned"):
        MatterGen(short_diffusion()).sample(num_atoms, guidance=guidance)


def test_conditional_load_pretrained_checks_the_parameters(conditional, model):
    state = upstream_state(conditional)
    assert (
        "diffusion_module.model.property_embeddings_adapt.chemical_system."
        "conditional_embedding_module.embedding.weight"
    ) in state
    assert (
        "diffusion_module.model.gemnet.cond_mixin_layers.chemical_system.0.weight"
        in state
    )
    configuration = nested(Config.load().upstream_sections("chemical_system"))

    other = MatterGenChemicalSystem(conditional.config)
    other.load_pretrained(state, configuration)

    # each model refuses the weights and the configuration of the other one
    with pytest.raises(InvalidWeights, match="another configuration"):
        MatterGen(model.config).load_pretrained(state, configuration)
    with pytest.raises(InvalidWeights, match="Unexpected"):
        MatterGen(model.config).load_pretrained(state)
    with pytest.raises(InvalidWeights, match="no lightning_module"):
        other.load_pretrained(
            upstream_state(model), nested(Config.load().upstream.sections)
        )
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(upstream_state(model))


# the model conditioned on a space group, with random weights


@pytest.fixture(scope="module")
def space_group_model() -> MatterGenSpaceGroup:
    torch.manual_seed(0)
    return MatterGenSpaceGroup(narrow()).eval().requires_grad_(False)


def test_space_group_guidance(space_group_model):
    guidance = space_group_model.guidance([225, 1, 230], scale=0.5)

    assert torch.equal(guidance.values[0], torch.tensor([225, 1, 230]))
    assert guidance.scale == 0.5
    assert space_group_model.guidance(torch.tensor([194])).scale == 2.0

    for wrong in (0, 231):
        with pytest.raises(ValueError, match="no space group"):
            space_group_model.guidance([wrong])


def test_space_group_network(space_group_model):
    structures = space_group_model.transform()(make_batch())
    groups = space_group_model.guidance([216, 216]).values

    free = space_group_model(structures, TIMES)
    mixed = space_group_model(structures, TIMES, groups, torch.tensor([True, False]))
    other = space_group_model(structures, TIMES, (torch.tensor([225, 216]),))
    first = structures.batch == 0

    # only the conditioned crystal changes, with its space group, and no
    # element is lowered by a space group
    assert torch.equal(mixed.x[~first], free.x[~first])
    assert torch.equal(mixed.logits[~first], free.logits[~first])
    assert not torch.allclose(mixed.x[first], free.x[first])
    assert not torch.allclose(other.x[first], mixed.x[first])
    assert torch.equal(mixed.logits > -1e9, free.logits > -1e9)


def test_space_group_sample():
    torch.manual_seed(0)
    model = MatterGenSpaceGroup(short_diffusion()).eval().requires_grad_(False)
    num_atoms = torch.tensor([3, 2])

    guided = model.sample(
        num_atoms, generator=generator(), guidance=model.guidance([225, 194])
    )
    free = model.sample(num_atoms, generator=generator())
    assert guided.lattice.isfinite().all()
    assert not torch.equal(guided.x, free.x)


def test_each_fine_tuned_model_refuses_the_weights_of_the_other_one(
    conditional, space_group_model
):
    chemical, space = upstream_state(conditional), upstream_state(space_group_model)
    assert (
        "diffusion_module.model.property_embeddings_adapt.space_group."
        "conditional_embedding_module.embedding.weight"
    ) in space

    other = MatterGenSpaceGroup(space_group_model.config)
    other.load_pretrained(space, nested(Config.load().upstream_sections("space_group")))

    with pytest.raises(
        InvalidWeights, match="no .*property_embeddings_adapt.space_group"
    ):
        other.load_pretrained(
            space, nested(Config.load().upstream_sections("chemical_system"))
        )
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(chemical)
    with pytest.raises(InvalidWeights, match="Missing"):
        MatterGenChemicalSystem(conditional.config).load_pretrained(space)


# the model conditioned on a band gap, with random weights


@pytest.fixture(scope="module")
def band_gap_model() -> MatterGenBandGap:
    torch.manual_seed(0)
    model = MatterGenBandGap(narrow()).eval().requires_grad_(False)
    scaler = model.property_embeddings_adapt["dft_band_gap"].scaler
    scaler.means.fill_(0.8)
    scaler.stds.fill_(1.4)

    return model


def test_band_gap_guidance(band_gap_model):
    guidance = band_gap_model.guidance([0.0, 3.5], scale=1.0)

    assert torch.equal(guidance.values[0], torch.tensor([0.0, 3.5]))
    assert guidance.scale == 1.0
    assert band_gap_model.guidance([1.0]).scale == 2.0

    for wrong in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            band_gap_model.guidance([wrong])


def test_band_gap_embedding(band_gap_model):
    # the gap standardized, then a sine and a cosine per frequency, as the time
    property_embedding = band_gap_model.property_embeddings_adapt["dft_band_gap"]
    embedding = property_embedding(torch.tensor([0.8, 2.2]))
    frequencies = property_embedding.conditional_embedding_module.div_term

    assert embedding.shape == (2, 64)
    assert torch.equal(embedding[0], band_gap_model.embed_time(torch.zeros(1))[0])
    assert torch.allclose(embedding[1, 0::2], torch.sin(frequencies))
    assert torch.allclose(embedding[1, 1::2], torch.cos(frequencies))


def test_band_gap_network(band_gap_model):
    structures = band_gap_model.transform()(make_batch())

    free = band_gap_model(structures, TIMES)
    mixed = band_gap_model(
        structures, TIMES, (torch.tensor([3.0, 3.0]),), torch.tensor([True, False])
    )
    other = band_gap_model(structures, TIMES, (torch.tensor([0.0, 3.0]),))
    first = structures.batch == 0

    assert torch.equal(mixed.x[~first], free.x[~first])
    assert not torch.allclose(mixed.x[first], free.x[first])
    assert not torch.allclose(other.x[first], mixed.x[first])
    assert torch.equal(mixed.logits > -1e9, free.logits > -1e9)


def test_band_gap_load_pretrained(band_gap_model, space_group_model):
    state = upstream_state(band_gap_model)
    prefix = "diffusion_module.model.property_embeddings_adapt.dft_band_gap."
    assert torch.equal(state[prefix + "scaler.means"], torch.tensor([0.8]))
    assert prefix + "conditional_embedding_module.div_term" in state

    other = MatterGenBandGap(band_gap_model.config)
    other.load_pretrained(
        state, nested(Config.load().upstream_sections("dft_band_gap"))
    )
    scaler = other.property_embeddings_adapt["dft_band_gap"].scaler
    assert torch.equal(scaler.stds, torch.tensor([1.4]))

    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(upstream_state(space_group_model))
    with pytest.raises(InvalidWeights, match="Missing"):
        MatterGenSpaceGroup(space_group_model.config).load_pretrained(state)


def test_magnetic_density_shares_the_model_of_a_number(band_gap_model):
    # the same network as the band gap, its own property and its own weights
    model = MatterGenMagneticDensity(narrow())
    assert model.CONDITIONS == ("dft_mag_density",)
    assert model.gemnet.config.conditions == ("dft_mag_density",)
    assert model.guidance([0.15]).scale == 2.0

    state = upstream_state(model)
    prefix = "diffusion_module.model.property_embeddings_adapt.dft_mag_density."
    assert prefix + "scaler.stds" in state
    model.load_pretrained(
        state, nested(Config.load().upstream_sections("dft_mag_density"))
    )

    with pytest.raises(InvalidWeights, match="Missing"):
        model.load_pretrained(upstream_state(band_gap_model))
    with pytest.raises(InvalidWeights, match="no .*dft_mag_density"):
        model.load_pretrained(
            state, nested(Config.load().upstream_sections("dft_band_gap"))
        )


def test_bulk_modulus_is_standardized_after_its_logarithm(band_gap_model):
    # the network of the other numbers, the logarithm of the bulk modulus
    model = MatterGenBulkModulus(narrow()).eval().requires_grad_(False)
    property_embedding = model.property_embeddings_adapt["ml_bulk_modulus"]
    scaler = property_embedding.scaler
    encoding = property_embedding.conditional_embedding_module
    assert isinstance(scaler, StandardScaler) and scaler.log10
    assert isinstance(encoding, NoiseLevelEncoding)
    scaler.means.fill_(1.7)
    scaler.stds.fill_(0.4)
    assert model.CONDITIONS == ("ml_bulk_modulus",)
    band_gap_scaler = band_gap_model.property_embeddings_adapt["dft_band_gap"].scaler
    assert isinstance(band_gap_scaler, StandardScaler) and not band_gap_scaler.log10

    # 10^1.7 GPa is the mean: the embedding of 0, as the time 0
    embedding = property_embedding(torch.tensor([10**1.7, 10**2.1]))
    assert torch.allclose(embedding[0], model.embed_time(torch.zeros(1))[0], atol=1e-6)
    assert torch.allclose(embedding[1, 0::2], torch.sin(encoding.div_term), atol=1e-5)

    values = model.guidance([400.0]).values[0]
    assert values is not None and torch.equal(values, torch.tensor([400.0]))
    for wrong in (0.0, -50.0, float("nan")):
        with pytest.raises(ValueError, match="positive|finite"):
            model.guidance([wrong])

    state = upstream_state(model)
    model.load_pretrained(
        state, nested(Config.load().upstream_sections("ml_bulk_modulus"))
    )
    # a training without the logarithm is refused
    sections = Config.load().upstream_sections("ml_bulk_modulus")
    name = (
        "lightning_module.diffusion_module.model.property_embeddings_adapt."
        "ml_bulk_modulus.scaler"
    )
    configuration = nested(
        {**sections, name: {**sections[name], "log10_transform": False}}
    )
    with pytest.raises(InvalidWeights, match="log10_transform"):
        model.load_pretrained(state, configuration)


# the model conditioned on a chemical system and the energy above the hull


@pytest.fixture(scope="module")
def joint_model() -> MatterGenChemicalSystemEnergyAboveHull:
    torch.manual_seed(0)
    return MatterGenChemicalSystemEnergyAboveHull(narrow()).eval().requires_grad_(False)


def test_joint_guidance(joint_model):
    guidance = joint_model.guidance(["Zn-S", "Fe-O"], [0.0, 0.05], scale=1.5)
    systems, energies = guidance.values

    assert systems is not None and energies is not None
    assert systems.nonzero().tolist() == [[0, 16], [0, 30], [1, 8], [1, 26]]
    assert torch.equal(energies, torch.tensor([0.0, 0.05]))
    assert guidance.scale == 1.5
    assert joint_model.guidance(None, [0.0]).values[0] is None

    with pytest.raises(ValueError, match="give values for each"):
        joint_model.guidance(["Li-O"])
    with pytest.raises(ValueError, match="give values for each"):
        joint_model.guidance(None, None)
    with pytest.raises(ValueError, match="finite"):
        joint_model.guidance(["Li-O"], [float("nan")])
    with pytest.raises(ValueError, match="not conditioned"):
        MatterGen(narrow()).guidance(["Li-O"])


def test_joint_network(joint_model):
    structures = joint_model.transform()(make_batch())
    guidance = joint_model.guidance(["Zn-S", "Zn-S"], [0.0, 0.0])
    systems, energies = guidance.values

    free = joint_model(structures, TIMES)
    both = joint_model(structures, TIMES, guidance.values)
    system_only = joint_model(structures, TIMES, (systems, None))
    energy_only = joint_model(structures, TIMES, (None, energies))
    first = structures.batch == 0

    # each property changes the scores; only the chemical system lowers elements
    assert not torch.allclose(both.x[first], system_only.x[first])
    assert not torch.allclose(energy_only.x[first], free.x[first])
    assert torch.equal(energy_only.logits > -1e9, free.logits > -1e9)
    kept = (system_only.logits[first] > -1e9).nonzero()[:, 1].unique() + 1
    assert kept.tolist() == [16, 30]

    with pytest.raises(ValueError, match="one value each"):
        joint_model(structures, TIMES, (systems,))


def test_joint_load_pretrained(joint_model, conditional):
    state = upstream_state(joint_model)
    prefix = "diffusion_module.model.property_embeddings_adapt."
    assert prefix + "energy_above_hull.scaler.means" in state
    assert (
        prefix + "chemical_system.conditional_embedding_module.embedding.weight"
        in state
    )

    other = MatterGenChemicalSystemEnergyAboveHull(joint_model.config)
    sections = Config.load().upstream_sections("chemical_system_energy_above_hull")
    other.load_pretrained(state, nested(sections))

    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(upstream_state(conditional))
    with pytest.raises(InvalidWeights, match="Unexpected"):
        MatterGenChemicalSystem(conditional.config).load_pretrained(state)


def test_two_numbers(band_gap_model):
    # the magnetic density and the HHI score: two numbers, each its own adapter
    model = MatterGenMagneticDensityHHIScore(narrow()).eval().requires_grad_(False)
    assert model.CONDITIONS == ("dft_mag_density", "hhi_score")
    assert model.gemnet.config.conditions == ("dft_mag_density", "hhi_score")

    guidance = model.guidance([0.15, 0.0], [1000.0, 6000.0])
    densities, scores = guidance.values
    assert densities is not None and scores is not None
    assert torch.equal(scores, torch.tensor([1000.0, 6000.0]))
    assert guidance.scale == 2.0
    with pytest.raises(ValueError, match="finite"):
        model.guidance([0.15], [float("inf")])

    structures = model.transform()(make_batch())
    free = model(structures, TIMES)
    both = model(structures, TIMES, guidance.values)
    density_only = model(structures, TIMES, (densities, None))
    score_only = model(structures, TIMES, (None, scores))
    first = structures.batch == 0
    assert not torch.allclose(both.x[first], density_only.x[first])
    assert not torch.allclose(score_only.x[first], free.x[first])
    assert torch.equal(both.logits > -1e9, free.logits > -1e9)

    state = upstream_state(model)
    prefix = "diffusion_module.model.property_embeddings_adapt."
    assert prefix + "hhi_score.scaler.stds" in state
    model.load_pretrained(
        state, nested(Config.load().upstream_sections("dft_mag_density_hhi_score"))
    )
    with pytest.raises(InvalidWeights, match="Unexpected"):
        MatterGenMagneticDensity(narrow()).load_pretrained(state)
    with pytest.raises(InvalidWeights, match="Missing"):
        model.load_pretrained(upstream_state(band_gap_model))


# the pretrained model against the real MatterGen


def reference(
    name: str = "mattergen_reference.npz",
) -> tuple[Structures, dict[str, torch.Tensor]]:
    """Structures of the tests and what MatterGen computed for them."""
    return reference_structures(), dict(load_reference(name).arrays)


def neighbours_at_the_same_distance(structures: Structures) -> torch.Tensor:
    """Structures where the 50 nearest neighbours of GemNet-T are not defined.

    In a crystal several neighbours are at the same distance, and the two
    images of an atom on each side of it always are: MatterGen keeps one or the
    other from the rounding of their distances and the order of its sort.
    """
    config = Config.load().graph
    near = AddRadiusGraph(
        config.radius, self_images=True, max_image_shift=config.max_image_shift
    )

    return ties(structures, near, config.max_neighbors)


# the weights of the models without condition, with what MatterGen computed
BASE_REFERENCES = {
    NAME: "mattergen_reference.npz",
    MP_20_NAME: "mattergen_mp_20_base_reference.npz",
}


@pytest.fixture(scope="module", params=sorted(BASE_REFERENCES))
def base_weights(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def pretrained(base_weights) -> MatterGen:
    return load_pretrained(MatterGen, "mattergen", base_weights)


@pytest.fixture(scope="module")
def base_reference(base_weights) -> tuple[Structures, dict[str, torch.Tensor]]:
    return reference(BASE_REFERENCES[base_weights])


def test_pretrained_frequencies_are_the_ones_of_the_configuration(pretrained):
    """The weights have their frequencies of the time, computed on another device."""
    computed = MatterGen().time_frequencies
    found = pretrained.time_frequencies

    assert torch.allclose(found, computed, rtol=1e-6, atol=0)
    assert not torch.equal(found, computed)


def test_pretrained_graph_matches_mattergen(pretrained, base_reference):
    """The graph GemNet-T of MatterGen 9242366 builds, where its neighbours are defined."""
    structures, expected = base_reference
    defined = ~neighbours_at_the_same_distance(structures)
    assert defined.sum() >= 15

    graphs = pretrained.transform()(structures)
    graph = graphs.graph

    assert torch.equal(graph.num_edges[defined], expected["graph_num_edges"][defined])
    assert torch.equal(
        graphs.triplets.num_tuples[defined], expected["graph_num_triplets"][defined]
    )
    distance_sum = graph.aggregate_edges_to_structures(graph.edges_distances().double())
    assert torch.allclose(
        distance_sum[defined], expected["graph_sum_distances"][defined], rtol=1e-6
    )


def condition_values(
    pretrained: MatterGen, expected: dict[str, torch.Tensor], prefix: str
) -> tuple[torch.Tensor, ...] | None:
    """Properties of each structure of a reference, as the fine-tuned model reads them.

    Args:
        prefix: "" for the structures of the tests, "sampling_" for the
            crystals of the sampling.
    """
    if not pretrained.CONDITIONS:
        return None

    return tuple(
        (
            expected[f"{prefix}{name}s"].float()
            if name == "chemical_system"
            else expected[f"{prefix}{name}s"]
        )
        for name in pretrained.CONDITIONS
    )


def compare_network(
    pretrained: MatterGen,
    structures: Structures,
    expected: dict[str, torch.Tensor],
    prefix: str,
) -> None:
    defined = ~neighbours_at_the_same_distance(structures)
    assert defined.sum() >= 15
    atoms = defined[structures.batch]

    values = condition_values(pretrained, expected, "")
    conditioned = expected.get("conditioned")

    def evaluate(crystals: Structures) -> tuple[torch.Tensor, ...]:
        denoised = pretrained(
            pretrained.transform()(crystals),
            expected[f"{prefix}_times"],
            values,
            conditioned,
        )
        return denoised.x, denoised.lattice, denoised.logits

    denoised = Denoised(*evaluate(structures))
    spread = dict(zip(("x", "l", "logits"), rounding_spread(evaluate, structures)))
    logits = expected[f"{prefix}_pred_logits"]
    allowed = logits[atoms] > -1e9
    assert torch.equal(denoised.logits[atoms] > -1e9, allowed)

    for found, name, selected in (
        (denoised.x[atoms], "x", atoms),
        (denoised.lattice[defined], "l", defined),
        (denoised.logits[atoms][allowed], "logits", atoms),
    ):
        single = expected[f"{prefix}_pred_{name}"][selected]
        double = expected[f"{prefix}_double_{name}"][selected]
        moved = spread[name][selected]
        if name == "logits":
            single, double, moved = single[allowed], double[allowed], moved[allowed]

        # as near as MatterGen to MatterGen in double precision, or as near to
        # MatterGen as a change of the last bits of the inputs moves the rewrite
        assert as_near_as_original(found, single, double) or within_rounding(
            found, single, moved
        ), name

    # elsewhere the graphs differ by a few edges: the outputs stay near
    assert (denoised.x - expected[f"{prefix}_pred_x"]).abs().mean() < 0.05


def test_pretrained_network_matches_mattergen(pretrained, base_reference):
    """The denoiser of MatterGen 9242366, on the structures at times from 0 to 1."""
    structures, expected = base_reference
    compare_network(pretrained, structures, expected, "clean")


def test_pretrained_network_matches_mattergen_on_noisy_structures(
    pretrained, base_reference
):
    """The same with the noise of the diffusion at the time of each structure."""
    structures, expected = base_reference
    noisy_structures = Structures(
        lattice=expected["noisy_l"],
        x=expected["noisy_x"],
        z=expected["noisy_z"],
        num_atoms=structures.num_atoms,
    )
    assert (noisy_structures.z == pretrained.mask).sum() > 20

    compare_network(pretrained, noisy_structures, expected, "noisy")


def state(expected: dict[str, torch.Tensor], name: str, step: int) -> Structures:
    """Crystals of the sampling of MatterGen at one of its saved steps, with their graph."""
    return MatterGen.transform()(
        Structures(
            lattice=expected[f"{name}_l"][step],
            x=expected[f"{name}_x"][step],
            z=expected[f"{name}_z"][step],
            num_atoms=expected["sampling_num_atoms"],
        )
    )


def recorded_scores(
    expected: dict[str, torch.Tensor], name: str, step: int
) -> Denoised:
    """Scores MatterGen computed for its corrector or its predictor at a saved step."""
    logits = expected["predictor_score_logits"][step]

    return Denoised(
        x=expected[f"{name}_score_x"][step],
        lattice=expected[f"{name}_score_l"][step],
        logits=logits if name == "predictor" else torch.zeros_like(logits),
    )


def recorded_noise(expected: dict[str, torch.Tensor], step: int) -> StepNoise:
    """Noise MatterGen drew at a saved step, and Gumbel noise which selects its elements."""
    chosen = F.one_hot(expected["after_z"][step] - 1, 101).float()

    return StepNoise(
        corrector_x=expected["corrector_noise_x"][step],
        corrector_lattice=expected["corrector_noise_l"][step],
        predictor_x=expected["predictor_noise_x"][step],
        predictor_lattice=expected["predictor_noise_l"][step],
        elements=(chosen - 1) * 1e30,
    )


def test_pretrained_step_matches_mattergen(pretrained, base_reference):
    """A step of the sampling of MatterGen: the network, the corrector and the predictor."""
    _, expected = base_reference
    check_one_step(pretrained, expected, ("score",))


@pytest.mark.full
def test_pretrained_steps_match_mattergen(pretrained, base_reference):
    """Steps of the sampling of MatterGen, each from its crystals, its scores and its noise."""
    _, expected = base_reference
    check_steps(pretrained, expected)


def check_step(
    pretrained: MatterGen, expected: dict[str, torch.Tensor], i: int
) -> None:
    """A recorded step of the sampling of MatterGen, given its scores and its noise."""
    step = int(expected["sampling_steps"][i])
    noise = recorded_noise(expected, i)

    # the corrector moves the coordinates and the lattice: the step of each
    # crystal depends on the other ones
    corrected = pretrained.correct_given_scores(
        state(expected, "before", i),
        step,
        recorded_scores(expected, "corrector", i),
        noise,
    )
    assert_close_in_the_cell(
        corrected.x, expected["corrected_x"][i], name=f"corrected x at {step}"
    )
    assert_close(
        corrected.lattice, expected["corrected_l"][i], name=f"corrected L at {step}"
    )
    assert torch.equal(corrected.z, expected["corrected_z"][i])

    # the predictor, from the crystals MatterGen corrected
    given = state(expected, "corrected", i)
    assert given.graph is not None
    times, _ = pretrained.step_times(given, step)
    element_steps = given.graph.unpool_structures(pretrained.element_steps(times))
    posterior = pretrained.element_posterior(
        given.z, expected["predictor_score_logits"][i], element_steps
    )
    assert_close(
        posterior, expected["posterior_logits"][i], name=f"posterior at {step}"
    )

    scores = recorded_scores(expected, "predictor", i)
    predicted = pretrained.predict_given_scores(given, step, scores, noise)
    assert_close_in_the_cell(predicted.x, expected["after_x"][i], name=f"x at {step}")
    assert_close(predicted.lattice, expected["after_l"][i], name=f"L at {step}")


def check_steps(pretrained: MatterGen, expected: dict[str, torch.Tensor]) -> None:
    """Every saved step of the sampling of MatterGen, given its scores and its noise."""
    steps = expected["sampling_steps"].tolist()
    assert torch.allclose(pretrained.times()[steps], expected["sampling_times"])

    for i in range(len(steps)):
        check_step(pretrained, expected, i)

    # the last step gives the crystals MatterGen returns: without noise, and
    # each atom with its most probable element
    last = len(steps) - 1
    noise = replace(recorded_noise(expected, last), elements=None)
    scores = recorded_scores(expected, "predictor", last)
    final = pretrained.predict_given_scores(
        state(expected, "corrected", last), steps[-1], scores, noise
    )
    assert torch.equal(expected["after_x"][-1], expected["mean_x"])
    assert torch.equal(expected["after_l"][-1], expected["mean_l"])
    assert torch.equal(final.z, expected["mean_z"])
    assert_close_in_the_cell(final.x, expected["mean_x"], name="final x")
    assert_close(final.lattice, expected["mean_l"], name="final L")


def step_with_a_defined_state(expected: dict[str, torch.Tensor]) -> int:
    """The first recorded step with a state whose 50 neighbours are defined.

    Before the corrector or before the predictor: there, the network can be
    compared with the scores MatterGen computed. No step of the reference of
    the base model has both states defined.
    """
    for i in range(len(expected["sampling_steps"])):
        if not all(
            neighbours_at_the_same_distance(state(expected, name, i)).any()
            for name in ("before", "corrected")
        ):
            return i
    raise AssertionError("no recorded step without ties of the neighbours")


def check_one_step(
    pretrained: MatterGen, expected: dict[str, torch.Tensor], kinds: tuple[str, ...]
) -> None:
    """A step of the sampling of MatterGen, end to end.

    The network on the states MatterGen reached where their graphs are
    defined, within its rounding, then the corrector and the predictor from
    MatterGen's scores and noise.
    """
    i = step_with_a_defined_state(expected)
    for kind in kinds:
        assert compare_scores_at(pretrained, expected, kind, i) >= 1
    check_step(pretrained, expected, i)


@pytest.mark.full
def test_pretrained_scores_of_the_sampling_match_mattergen(pretrained, base_reference):
    """Scores of the crystals of the sampling of MatterGen, where their graphs are defined."""
    _, expected = base_reference
    assert compare_sampling_scores(pretrained, expected, "score") >= 2


def compare_scores_at(
    pretrained: MatterGen, expected: dict[str, torch.Tensor], kind: str, i: int
) -> int:
    """Compare the scores of the network on the crystals of a recorded step.

    Args:
        kind: "score" for the scores of the steps, "unconditional" or
            "conditional" for the ones the guidance mixes.
        i: Index of the recorded step.

    Returns:
        Number of calls of the network compared, of 2: the ones whose crystals
        have no tie of their neighbours.
    """
    step = int(expected["sampling_steps"][i])
    values = None
    if kind == "conditional":
        values = condition_values(pretrained, expected, "sampling_")

    compared = 0
    for name, call in (("before", "corrector"), ("corrected", "predictor")):
        crystals = state(expected, name, i)
        if neighbours_at_the_same_distance(crystals).any():
            continue

        times, _ = pretrained.step_times(crystals, step)

        def evaluate(given: Structures) -> tuple[torch.Tensor, ...]:
            scores = pretrained.unguided_scores(
                pretrained.transform()(given),
                times,
                values,  # pylint: disable=cell-var-from-loop
            )
            return scores.x, scores.lattice

        found = evaluate(crystals.without_graphs())
        spread = rounding_spread(evaluate, crystals.without_graphs())
        for scores, moved, field in zip(found, spread, ("x", "l")):
            wanted = expected[f"{call}_{kind}_{field}"][i]
            assert within_rounding(scores, wanted, moved), (step, name, field)
        compared += 1

    return compared


def compare_sampling_scores(
    pretrained: MatterGen, expected: dict[str, torch.Tensor], kind: str
) -> int:
    """Compare the scores of the network on the crystals of every recorded step.

    Returns:
        Number of calls of the network compared.
    """
    return sum(
        compare_scores_at(pretrained, expected, kind, i)
        for i in range(len(expected["sampling_steps"]))
    )


# the models fine-tuned on a property against the real MatterGen

FINE_TUNED = {
    "chemical_system": (MatterGenChemicalSystem, CHEMICAL_SYSTEM_NAME),
    "space_group": (MatterGenSpaceGroup, SPACE_GROUP_NAME),
    "dft_band_gap": (MatterGenBandGap, BAND_GAP_NAME),
    "dft_mag_density": (MatterGenMagneticDensity, MAGNETIC_DENSITY_NAME),
    "chemical_system_energy_above_hull": (
        MatterGenChemicalSystemEnergyAboveHull,
        ENERGY_ABOVE_HULL_NAME,
    ),
    "ml_bulk_modulus": (MatterGenBulkModulus, BULK_MODULUS_NAME),
    "dft_mag_density_hhi_score": (MatterGenMagneticDensityHHIScore, HHI_SCORE_NAME),
}


@pytest.fixture(scope="module", params=sorted(FINE_TUNED))
def fine_tuned(request) -> MatterGen:
    model, name = FINE_TUNED[request.param]

    return load_pretrained(model, "mattergen", name)


def fine_tuned_reference(
    model: MatterGen,
) -> tuple[Structures, dict[str, torch.Tensor]]:
    return reference(f"mattergen_{model.fine_tuning()}_reference.npz")


def test_pretrained_fine_tuned_network_matches_mattergen(fine_tuned):
    """The denoiser conditioned on a property, every other structure without it."""
    structures, expected = fine_tuned_reference(fine_tuned)
    assert expected["conditioned"].sum() == 20

    compare_network(fine_tuned, structures, expected, "clean")


def test_pretrained_fine_tuned_network_matches_mattergen_on_noisy_structures(
    fine_tuned,
):
    structures, expected = fine_tuned_reference(fine_tuned)
    noisy_structures = Structures(
        lattice=expected["noisy_l"],
        x=expected["noisy_x"],
        z=expected["noisy_z"],
        num_atoms=structures.num_atoms,
    )

    compare_network(fine_tuned, noisy_structures, expected, "noisy")


def test_pretrained_guidance_matches_mattergen(fine_tuned):
    """The scores of the guided sampling of MatterGen, from its two unguided scores."""
    _, expected = fine_tuned_reference(fine_tuned)
    scale = float(expected["sampling_guidance"])
    fine_tuning = fine_tuned.fine_tuning()
    assert fine_tuning is not None
    assert scale == fine_tuned.config.condition(fine_tuning).guidance_scale

    for call in ("corrector", "predictor"):
        for i in range(len(expected["sampling_steps"])):
            # the corrector does not read the scores of the elements: they are
            # only saved for the predictor
            no_logits = torch.zeros_like(expected["predictor_score_logits"])
            unconditional, conditional = (
                Denoised(
                    x=expected[f"{call}_{kind}_x"][i],
                    lattice=expected[f"{call}_{kind}_l"][i],
                    logits=expected.get(f"{call}_{kind}_logits", no_logits)[i],
                )
                for kind in ("unconditional", "conditional")
            )
            guided = guided_scores(unconditional, conditional, scale)

            assert torch.equal(guided.x, expected[f"{call}_score_x"][i])
            assert torch.equal(guided.lattice, expected[f"{call}_score_l"][i])
            if call == "predictor":
                assert torch.equal(guided.logits, expected["predictor_score_logits"][i])


def test_pretrained_guided_step_matches_mattergen(fine_tuned):
    """A step of the guided sampling: both scores of the network, then the step."""
    _, expected = fine_tuned_reference(fine_tuned)
    check_one_step(fine_tuned, expected, ("unconditional", "conditional"))


@pytest.mark.full
def test_pretrained_guided_steps_match_mattergen(fine_tuned):
    """Steps of the guided sampling of MatterGen, given its scores and its noise."""
    _, expected = fine_tuned_reference(fine_tuned)
    check_steps(fine_tuned, expected)

    if "chemical_system" in fine_tuned.CONDITIONS:
        # the crystals generated only have elements of their chemical systems
        systems = expected["sampling_chemical_systems"].bool()
        batch = torch.arange(4).repeat_interleave(expected["sampling_num_atoms"])
        assert systems[batch, expected["mean_z"]].all()


@pytest.mark.full
def test_pretrained_fine_tuned_scores_of_the_sampling_match_mattergen(fine_tuned):
    """Both scores the guidance mixes, where the graphs of the crystals are defined."""
    _, expected = fine_tuned_reference(fine_tuned)

    for kind in ("unconditional", "conditional"):
        assert compare_sampling_scores(fine_tuned, expected, kind) >= 2
