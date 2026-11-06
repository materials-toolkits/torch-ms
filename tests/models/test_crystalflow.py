"""Tests for the rewrite of CrystalFlow."""

import pytest
import torch

from helpers.compare import as_near_as_original, assert_close, assert_close_in_the_cell
from helpers.references import load_reference, reference_structures
from helpers.structures import generator, make_batch, make_structures
from helpers.weights import pretrained as load_pretrained
from torchms.data import Structures
from torchms.data.augmentation import (
    AtomTranslation,
    LatticeRotation,
    actions,
    sampling,
)
from torchms.graph.geometry import lattice_from_polar_parameters
from torchms.models.crystalflow import (
    Config,
    CrystalFlow,
    CrystalFlowGeneration,
    InvalidWeights,
    State,
)
from torchms.models.estimator import weights

NAME = "crystalflow-csp-mp20"
PRESSURE = "crystalflow-csp-calypso60-pressure"
GENERATION = "crystalflow-dng-mp20"
FORMATION_ENERGY = "crystalflow-dng-mp20-formation-energy"

# each set of weights, its class and the file of its reference
REFERENCES = {
    NAME: (CrystalFlow, "crystalflow_csp_mp20_reference.npz"),
    "crystalflow-csp-mpts52": (CrystalFlow, "crystalflow_csp_mpts52_reference.npz"),
    PRESSURE: (CrystalFlow, "crystalflow_csp_calypso60_pressure_reference.npz"),
    GENERATION: (CrystalFlowGeneration, "crystalflow_dng_mp20_reference.npz"),
    FORMATION_ENERGY: (
        CrystalFlowGeneration,
        "crystalflow_dng_mp20_formation_energy_reference.npz",
    ),
}

# the recorded step checked by default, in the middle of the flow: the other
# ones, chained steps and the whole sampling with --full
ONE_STEP = 50

# a network of 512 features on every pair of atoms
pytestmark = pytest.mark.timeout(60)

TIME = 0.4


@pytest.fixture(scope="module")
def model() -> CrystalFlow:
    """Model with random weights."""
    torch.manual_seed(0)
    return CrystalFlow().eval().requires_grad_(False)


def prepared(model: CrystalFlow, structures: Structures) -> tuple[Structures, State]:
    structures = model.transform()(structures)
    return structures, model.state_of(structures)


def test_pinned_checkpoints(tmp_path):
    config = Config.load()
    assert set(weights.CHECKPOINTS["crystalflow"]) == set(REFERENCES)
    assert config.default_weights == NAME
    assert config.generation.default_weights == GENERATION

    for name, files in weights.CHECKPOINTS["crystalflow"].items():
        (checkpoint,) = files
        # a release asset of the repository of the authors
        assert checkpoint.url.startswith(
            "https://github.com/ixsluo/CrystalFlow/releases/download/v1.0.0-alpha.1/"
        )
        assert len(checkpoint.sha256) == 64
        members = config.upstream.members_of(name)
        assert len(members.checkpoint_sha256) == len(members.hparams_sha256) == 64
        # a model with conditions has the standardization of its properties
        conditioned = bool(config.for_weights(name).model.conditions)
        assert (members.scalers is not None) == conditioned

    # four models share an archive: one file in the cache
    paths = {
        weights.checkpoint_paths("crystalflow", name, tmp_path)[0]
        for name in REFERENCES
    }
    assert paths == {
        tmp_path / "crystalflow" / "ckpt-v1.0.0.zip",
        tmp_path / "crystalflow" / "CSP-mpts-52.zip",
    }


def test_network_of_each_weights():
    for name, (cls, _) in REFERENCES.items():
        model = cls.for_weights(name)
        conditions = model.config.model.conditions
        assert model.default_weights() == name
        assert hasattr(model, "cond_emb") == bool(conditions)
        assert hasattr(model.decoder, "type_out") == (cls is CrystalFlowGeneration)
        assert hasattr(model.decoder, "cemb_adapter_0") == bool(conditions)

    assert CrystalFlowGeneration().default_weights() == GENERATION
    with pytest.raises(KeyError, match="no network"):
        CrystalFlow.for_weights("crystalflow-9")


# the representation of the lattice


def test_state_of_structures(model):
    structures = make_batch()
    state = model.state_of(structures)
    back = model.to_structures(state, structures.z, structures.num_atoms)

    assert state.lattice.shape == (2, 6)
    # a symmetric lattice of the same lengths and angles
    assert torch.allclose(back.lattice, back.lattice.mT, atol=1e-5)
    gram = structures.lattice @ structures.lattice.mT
    assert torch.allclose(back.lattice @ back.lattice.mT, gram, atol=1e-3)
    assert torch.equal(back.x, structures.x % 1.0)


def test_state_from_noise(model):
    lattice = torch.randn(3, 6, generator=generator())
    x = torch.rand(5, 3, generator=generator())
    state = model.state_from_noise(lattice, x)
    sigma = model.config.flow.lattice_polar_sigma

    assert torch.allclose(state.lattice[:, :5], sigma * lattice[:, :5])
    assert torch.allclose(state.lattice[:, 5], sigma * lattice[:, 5] + 1)
    assert torch.equal(state.x, x)


# the network with random weights


def test_velocity(model):
    structures, state = prepared(model, make_batch())
    velocity = model.velocity(structures, state, TIME)

    assert velocity.lattice.shape == (2, 6) and velocity.x.shape == state.x.shape
    assert velocity.lattice.dtype == torch.float32
    assert velocity.lattice.isfinite().all() and velocity.x.isfinite().all()


def test_depends_on_the_time(model):
    structures, state = prepared(model, make_batch())

    assert not torch.allclose(
        model.velocity(structures, state, 0.4).x,
        model.velocity(structures, state, 0.6).x,
    )


def test_batch_equals_single(model):
    items = make_structures()
    velocity = model.velocity(*prepared(model, Structures.collate(items)), TIME)
    alone = [model.velocity(*prepared(model, item), TIME) for item in items]

    assert torch.allclose(
        velocity.lattice, torch.cat([v.lattice for v in alone]), atol=1e-5
    )
    assert torch.allclose(velocity.x, torch.cat([v.x for v in alone]), atol=1e-5)


def test_invariances(model):
    batch = make_batch()
    velocity = model.velocity(*prepared(model, batch), TIME)

    # the cell is read through its polar decomposition, the coordinates
    # through their differences
    for augment in (
        LatticeRotation(generator=generator()),
        AtomTranslation(generator=generator()),
    ):
        moved = model.velocity(*prepared(model, augment(batch)), TIME)
        assert torch.allclose(moved.lattice, velocity.lattice, atol=1e-4)
        assert torch.allclose(moved.x, velocity.x, atol=1e-4)

    permutation = sampling.random_permutation(batch.batch, generator())
    permuted = model.velocity(
        *prepared(model, actions.permute_atoms(batch, permutation)), TIME
    )
    assert torch.allclose(permuted.lattice, velocity.lattice, atol=1e-4)
    assert torch.allclose(permuted.x, velocity.x[permutation], atol=1e-4)


def test_needs_its_graph(model):
    structures = make_batch()
    with pytest.raises(ValueError, match=r"apply CrystalFlow.transform\(\)"):
        model.velocity(structures, model.state_of(structures), TIME)


def test_unknown_element(model):
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[101]
    )

    with pytest.raises(ValueError, match="not supported"):
        model.sample(structures.z, structures.num_atoms, num_steps=1)


# sampling with random weights


def test_sample_step(model):
    structures, state = prepared(model, make_batch())
    step, num_steps = 30, 100

    moved = model.sample_step(structures, state, step, num_steps, anneal_slope=0.0)
    velocity = model.velocity(structures, state, step / num_steps)
    assert torch.allclose(moved.lattice, state.lattice + velocity.lattice / num_steps)
    assert_close_in_the_cell(moved.x, state.x + velocity.x / num_steps)
    assert ((moved.x >= 0) & (moved.x < 1)).all()

    # the coordinates are annealed, by 1 + slope * max(t - offset, 0)
    annealed = model.sample_step(
        structures, state, step, num_steps, anneal_slope=2.0, anneal_offset=0.1
    )
    assert torch.equal(annealed.lattice, moved.lattice)
    assert_close_in_the_cell(annealed.x, state.x + 1.4 * velocity.x / num_steps)


def test_sample(model):
    batch = make_batch()
    structures = model.sample(
        batch.z, batch.num_atoms, num_steps=3, generator=generator()
    )

    assert torch.equal(structures.z, batch.z) and torch.equal(
        structures.num_atoms, batch.num_atoms
    )
    assert torch.allclose(structures.lattice, structures.lattice.mT, atol=1e-5)
    again = model.sample(batch.z, batch.num_atoms, num_steps=3, generator=generator())
    assert torch.equal(again.x, structures.x)


# the generation of the elements, with random weights


@pytest.fixture(scope="module")
def generation() -> CrystalFlowGeneration:
    torch.manual_seed(0)
    return CrystalFlowGeneration().eval().requires_grad_(False)


def test_table_of_the_elements(generation):
    table = generation.type_encoding
    z = torch.arange(1, 119)
    vectors = table.encode(z)

    assert vectors.shape == (118, 28) and (vectors.sum(dim=1) == 2).all()
    assert torch.equal(table.decode(vectors), z)
    # the column is the largest one of an element in the largest row
    noisy = vectors + 0.1 * torch.randn(vectors.shape, generator=generator())
    assert torch.equal(table.decode(noisy), z)
    assert (table.decode(torch.randn(500, 28, generator=generator())) > 0).all()


def test_generation_velocity(generation):
    structures = generation.transform()(make_batch())
    state = generation.state_of(structures)
    velocity = generation.velocity(structures, state, TIME)

    assert state.elements is not None and state.elements.shape == (
        len(structures.z),
        28,
    )
    assert (
        velocity.elements is not None
        and velocity.elements.shape == state.elements.shape
    )
    back = generation.to_structures(state, structures.num_atoms)
    assert torch.equal(back.z, structures.z)

    # the elements are read from their vectors, not from the atomic numbers
    other = Structures(
        lattice=structures.lattice,
        x=structures.x,
        z=torch.ones_like(structures.z),
        num_atoms=structures.num_atoms,
    )
    again = generation.velocity(generation.transform()(other), state, TIME)
    assert torch.equal(again.elements, velocity.elements)


def test_generation_sample_step(generation):
    structures = generation.transform()(make_batch())
    state = generation.state_of(structures)
    moved = generation.sample_step(structures, state, 30, 100)
    velocity = generation.velocity(structures, state, 0.3)

    assert torch.allclose(moved.elements, state.elements + velocity.elements / 100)


def test_generation_sample(generation):
    num_atoms = generation.sample_num_atoms(3, generator())
    structures = generation.sample(num_atoms, num_steps=2, generator=generator())

    assert torch.equal(structures.num_atoms, num_atoms)
    assert ((structures.z >= 1) & (structures.z <= 118)).all()
    numbers = generation.sample_num_atoms(2000, generator())
    assert numbers.min() >= 1 and numbers.max() <= 20


# conditions, with random weights


@pytest.fixture(scope="module")
def conditioned() -> CrystalFlow:
    torch.manual_seed(0)
    model = CrystalFlow.for_weights(PRESSURE).eval().requires_grad_(False)
    model.condition_means.fill_(100.0)
    model.condition_stds.fill_(50.0)
    return model


def test_conditions(conditioned):
    structures = conditioned.transform()(make_batch())
    state = conditioned.state_of(structures)
    pressure = torch.tensor([[0.0], [200.0]])

    assert conditioned.conditions == ("pressure",)
    embedding = conditioned.embed_conditions(pressure)
    expansion = conditioned.cond_emb.sub_emb_dict["pressure"].expansion_net
    assert torch.allclose(embedding, expansion.expand(torch.tensor([[-2.0], [2.0]])))
    # the first and the last gaussians are centered on these values
    assert torch.allclose(embedding[0, 0], torch.tensor(1.0)) and torch.allclose(
        embedding[1, -1], torch.tensor(1.0)
    )

    with_pressure = conditioned.velocity(structures, state, TIME, pressure)
    without = conditioned.velocity(structures, state, TIME)
    assert not torch.allclose(with_pressure.x, without.x)
    with pytest.raises(ValueError, match="conditioned on"):
        conditioned.velocity(structures, state, TIME, torch.zeros(2, 2))


def test_guidance(conditioned, model):
    structures = conditioned.transform()(make_batch())
    state = conditioned.state_of(structures)
    pressure = torch.tensor([[10.0], [20.0]])
    step = dict(step=40, num_steps=100, anneal_slope=0.0)

    guided = conditioned.velocity(structures, state, 0.4, pressure)
    free = conditioned.velocity(structures, state, 0.4)
    moved = conditioned.sample_step(
        structures, state, conditions=pressure, guide_factor=1.5, **step
    )
    expected = state.lattice + (1.5 * guided.lattice - 0.5 * free.lattice) / 100
    assert torch.allclose(moved.lattice, expected, atol=1e-6)
    alone = conditioned.sample_step(
        structures, state, conditions=pressure, guide_factor=1.0, **step
    )
    assert torch.allclose(
        alone.lattice, state.lattice + guided.lattice / 100, atol=1e-6
    )

    with pytest.raises(ValueError, match="give them"):
        conditioned.sample_step(structures, state, **step)
    with pytest.raises(ValueError, match="give them"):
        model.sample_step(structures, state, conditions=pressure, **step)


def nested(sections: dict) -> dict:
    """A configuration of a training with the arguments of each section."""
    hparams: dict = {}
    for section, values in sections.items():
        node = hparams
        for key in section.split("."):
            node = node.setdefault(key, {})
        node.update(values)
    return hparams


def test_conditions_need_their_standardization(conditioned):
    other = CrystalFlow.for_weights(PRESSURE).eval()
    state = dict(conditioned.state_dict())
    hparams = nested(other.config.upstream.sections)
    hparams["data"]["properties"] = ["enthalpy_per_atom", "pressure"]

    with pytest.raises(InvalidWeights, match="standardization"):
        other.load_pretrained(state, hparams)

    # the scaler of the pressure, the second property of the training
    other.load_pretrained(state, hparams, [(0.2, 6.5), (109.0, 138.0)])
    assert torch.allclose(other.condition_means, torch.tensor([109.0]))
    assert torch.allclose(other.condition_stds, torch.tensor([138.0]))

    # the weights of a model without conditions are refused
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(
            CrystalFlow().state_dict(), hparams, [(0.0, 1.0), (0.0, 1.0)]
        )


# weights


def test_load_pretrained(model):
    config = Config.load()
    state = dict(model.state_dict())
    unused = {
        "decoder.cemb_mixin_0.weight": torch.zeros(1),
        "beta_scheduler.betas": torch.zeros(1),
    }

    other = CrystalFlow().eval()
    hparams = {
        "model": {
            **config.upstream.sections["model"],
            "decoder": dict(config.upstream.sections["model.decoder"]),
        }
    }
    other.load_pretrained({**state, **unused}, hparams)
    assert torch.equal(
        other.decoder.lattice_out.weight, model.decoder.lattice_out.weight
    )

    changed = {"model": {**hparams["model"], "lattice_polar": False}}
    with pytest.raises(InvalidWeights, match="lattice_polar"):
        other.load_pretrained(state, changed)
    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state, "decoder.other": torch.zeros(1)})
    del state["decoder.coord_out.weight"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state)


# with the pretrained weights, against CrystalFlow


@pytest.fixture(scope="module", params=list(REFERENCES))
def weights_name(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def pretrained(weights_name):
    cls, _ = REFERENCES[weights_name]
    return load_pretrained(cls, "crystalflow", weights_name)


@pytest.fixture(scope="module")
def reference(weights_name):
    return load_reference(REFERENCES[weights_name][1])


def generates(model) -> bool:
    return isinstance(model, CrystalFlowGeneration)


def check(model, found, reference, key: str, name: str) -> None:
    """Values within 1e-5 of the largest one of CrystalFlow, or within its rounding.

    The network which generates the elements multiplies the rounding of float32
    near the time 1 (up to 3e-4 on the velocity of the elements of a
    perovskite at the time 1): a value further than 1e-5 is held to be as near
    to CrystalFlow in double precision as CrystalFlow in float32 is.
    """
    expected = reference[key]
    try:
        assert_close(found, expected, name=name)
    except AssertionError:
        if not generates(model):
            raise
        assert as_near_as_original(found, expected, reference[f"{key}_double"]), name


def test_polar_decomposition_matches_reference(pretrained, reference):
    state = pretrained.state_of(reference_structures())

    assert_close(state.lattice, reference["k"], name="k")
    matrix = lattice_from_polar_parameters(reference["k"].float())
    assert_close(matrix, reference["matrix"], name="lattice")


@pytest.mark.parametrize("name", [GENERATION, FORMATION_ENERGY])
def test_table_matches_reference(name):
    pretrained = load_pretrained(CrystalFlowGeneration, "crystalflow", name)

    # the buffers of the weights are the table of config.yaml
    built = type(pretrained.type_encoding)(pretrained.config.generation)
    for key, values in built.state_dict().items():
        assert torch.equal(pretrained.type_encoding.state_dict()[key], values), key


def test_network_matches_reference(pretrained, reference):
    structures = pretrained.transform()(reference_structures())
    times = reference["times"].float()
    element = (
        reference["elements"].float() if generates(pretrained) else structures.z - 1
    )
    conditions = [("", None)]
    if pretrained.conditions:
        embedding = pretrained.embed_conditions(reference["conditions"].float())
        conditions = [("", embedding), ("_free", None)]

    for suffix, condition in conditions:
        lattice, x, elements = pretrained.decoder(
            pretrained.embed_time(times),
            element,
            structures.graph,
            structures.x,
            reference["k"].float(),
            reference["matrix"].float(),
            condition,
        )
        check(
            pretrained, lattice, reference, f"pred_l{suffix}", "velocity of the lattice"
        )
        check(
            pretrained, x, reference, f"pred_f{suffix}", "velocity of the coordinates"
        )
        if generates(pretrained):
            check(
                pretrained,
                elements,
                reference,
                f"pred_t{suffix}",
                "velocity of the elements",
            )
        else:
            assert elements is None


def sampled(pretrained, reference) -> tuple[Structures, State]:
    num_atoms = reference["sample_num_atoms"]
    elements = reference["sample_t0"].float() if generates(pretrained) else None
    initial = pretrained.state_from_noise(
        reference["sample_l0"].float(), reference["sample_f0"].float(), elements
    )
    if generates(pretrained):
        structures = pretrained.to_structures(initial, num_atoms)
    else:
        structures = pretrained.to_structures(initial, reference["sample_z"], num_atoms)
    return pretrained.transform()(structures), initial


def recorded(pretrained, reference, step: int) -> State:
    elements = reference.get(f"sample_step{step}_t")
    return State(
        reference[f"sample_step{step}_l"].float(),
        reference[f"sample_step{step}_f"].float(),
        None if elements is None else elements.float(),
    )


def sample_conditions(pretrained, reference):
    if not pretrained.conditions:
        return None, None
    return reference["sample_conditions"].float(), float(reference["guide_factor"])


def check_step(pretrained, reference, step: int) -> None:
    structures, _ = sampled(pretrained, reference)
    state = recorded(pretrained, reference, step)
    conditions, guide_factor = sample_conditions(pretrained, reference)

    passes = [("", conditions)] + ([("_free", None)] if conditions is not None else [])
    for suffix, given in passes:
        velocity = pretrained.velocity(structures, state, step / 100, given)
        prefix = f"sample_step{step}_pred"
        check(pretrained, velocity.lattice, reference, f"{prefix}_l{suffix}", "lattice")
        check(pretrained, velocity.x, reference, f"{prefix}_f{suffix}", "coordinates")
        if generates(pretrained):
            check(
                pretrained,
                velocity.elements,
                reference,
                f"{prefix}_t{suffix}",
                "elements",
            )

    # the step, from the recorded state to the next recorded one
    if f"sample_step{step + 1}_l" in reference:
        moved = pretrained.sample_step(
            structures, state, step, conditions=conditions, guide_factor=guide_factor
        )
        following = recorded(pretrained, reference, step + 1)
        assert_close(moved.lattice, following.lattice, name="lattice after the step")
        assert_close_in_the_cell(moved.x, following.x)
        if generates(pretrained):
            assert_close(
                moved.elements, following.elements, name="elements after the step"
            )


def test_initial_state_matches_reference(pretrained, reference):
    _, initial = sampled(pretrained, reference)
    first = recorded(pretrained, reference, 1)

    assert torch.equal(initial.lattice, first.lattice)
    assert torch.equal(initial.x, first.x)
    if generates(pretrained):
        assert torch.equal(initial.elements, first.elements)


def test_one_step_matches_reference(pretrained, reference):
    check_step(pretrained, reference, ONE_STEP)


@pytest.mark.full
def test_every_recorded_step_matches_reference(pretrained, reference):
    for step in reference["sample_steps"].tolist():
        check_step(pretrained, reference, step)


@pytest.mark.full
@pytest.mark.timeout(600)
def test_whole_sampling_is_as_near_to_crystalflow_as_its_rounding(
    pretrained, reference
):
    # the flow multiplies a relative change of 1e-7 by about 1e5 in 100
    # steps: after them CrystalFlow in single and in double precision differ
    # by up to 2e-2. The rewrite is held to three times this gap, every step
    # above to 1e-5
    num_atoms = reference["sample_num_atoms"]
    _, initial = sampled(pretrained, reference)
    conditions, guide_factor = sample_conditions(pretrained, reference)
    if generates(pretrained):
        state = pretrained.integrate(
            initial, num_atoms, conditions=conditions, guide_factor=guide_factor
        )
        found = pretrained.to_structures(state, num_atoms)
        assert torch.equal(found.z, reference["sample_z"])
    else:
        z = reference["sample_z"]
        state = pretrained.integrate(
            initial, z, num_atoms, conditions=conditions, guide_factor=guide_factor
        )
        found = pretrained.to_structures(state, z, num_atoms)

    lattice, lattice_double = (
        reference["sample_lattice"],
        reference["sample_lattice_double"],
    )
    gap = (lattice - lattice_double).abs()
    difference = (found.lattice.double() - lattice).abs()
    assert difference.max() <= 3 * gap.max() and difference.mean() <= 3 * gap.mean()

    def in_cell(a, b):
        return ((a - b + 0.5) % 1.0 - 0.5).abs()

    gap = in_cell(reference["sample_x"], reference["sample_x_double"])
    difference = in_cell(found.x.double(), reference["sample_x"])
    assert difference.max() <= 3 * gap.max() and difference.mean() <= 3 * gap.mean()
