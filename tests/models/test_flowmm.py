"""Tests for the rewrite of FlowMM, Riemannian flow matching of crystals."""

from dataclasses import replace

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
from torchms.graph.transforms import AddCompleteGraph
from torchms.models.flowmm import Config, FlowMM, InvalidWeights, State
from torchms.models.flowmm.model import torus_difference
from torchms.models.estimator import weights

NAME = "flowmm-mp20-essflow"

# the recorded step checked by default, in the middle of the flow: the other
# ones and the whole sampling with --full
ONE_STEP = 500

# near the time 1 the network multiplies the rounding of float32: the
# velocity of the bits at these steps is held to be as near to FlowMM in
# double precision as FlowMM in float32 is, the other ones to 1e-5
ROUNDING_STEPS = (998, 999)

# a network of 512 features on every pair of atoms
pytestmark = pytest.mark.timeout(60)

TIME = torch.tensor(0.4)


@pytest.fixture(scope="module")
def model() -> FlowMM:
    """Model with random weights."""
    torch.manual_seed(0)
    return FlowMM().eval().requires_grad_(False)


def prepared(structures: Structures) -> tuple[Structures, State]:
    """Structures with their graph, and their point of the flow."""
    structures = FlowMM.transform()(structures)
    return structures, FlowMM().state_of(structures)


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS["flowmm"]) == {NAME}
    assert Config.load().default_weights == NAME

    for checkpoint in weights.CHECKPOINTS["flowmm"][NAME]:
        # a commit of the repository of the third party, not a branch
        assert checkpoint.url.startswith(
            "https://huggingface.co/Adhi94/essflow-flowmm-mp20/resolve/"
        )
        assert "/main/" not in checkpoint.url
        assert len(checkpoint.sha256) == 64

    paths = weights.checkpoint_paths("flowmm", NAME, tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        f"flowmm/{NAME}/flowmm_mp20.ckpt",
        f"flowmm/{NAME}/config.yaml",
    ]


# the spaces of the flow


def test_analog_bits(model):
    z = torch.arange(1, 129)
    bits = model.analog_bits(z)

    assert bits.shape == (128, 7)
    assert set(bits.unique().tolist()) == {-1.0, 1.0}
    assert torch.equal(model.atomic_numbers(bits), z)
    # a bit is set if it is positive, zero is not
    assert model.atomic_numbers(torch.zeros(1, 7)).item() == 1


def test_angles(model):
    flow = model.config.flow
    degrees = torch.tensor([[60.0, 90.0, 120.0], [75.0, 100.5, 110.0]])
    unconstrained = model.unconstrained_angles(degrees)

    assert torch.allclose(unconstrained[0, 1], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(model.angles_in_degrees(unconstrained), degrees, atol=1e-4)

    # any value of the unconstrained space is an angle of the interval
    found = model.angles_in_degrees(torch.tensor([[-200.0, 0.0, 200.0]]))
    assert (found >= flow.angle_min).all() and (found <= flow.angle_max).all()


def test_state_of_structures(model):
    structures = make_batch()
    state = model.state_of(structures)

    assert state.bits.shape == (len(structures.z), 7)
    assert state.lattice.shape == (2, 6)
    assert torch.equal(model.atomic_numbers(state.bits), structures.z)

    # the lattice is read back up to a rotation: the same lengths and angles
    back = model.to_structures(state, structures.num_atoms)
    lengths = back.lattice.norm(dim=-1)
    assert torch.allclose(lengths, structures.lattice.norm(dim=-1), atol=1e-4)
    gram = back.lattice @ back.lattice.transpose(-1, -2)
    expected = structures.lattice @ structures.lattice.transpose(-1, -2)
    assert torch.allclose(gram, expected, atol=1e-3)
    assert torch.allclose(back.x, structures.x)


def test_torus_difference():
    start = torch.rand(100, 3, generator=generator())
    end = torch.rand(100, 3, generator=generator(1))
    difference = torus_difference(start, end)

    assert (difference.abs() <= 0.5 + 1e-6).all()
    assert_close_in_the_cell(start + difference, end)


# the velocity with random weights


def test_velocity(model):
    structures, state = prepared(make_batch())
    velocity = model.velocity(structures, state, TIME)

    assert velocity.bits.shape == state.bits.shape
    assert velocity.x.shape == state.x.shape
    assert velocity.lattice.shape == state.lattice.shape
    assert velocity.bits.dtype == torch.float32
    assert all(
        v.isfinite().all() for v in (velocity.bits, velocity.x, velocity.lattice)
    )

    # the coordinates have no drift
    mean = structures.graph.aggregate_nodes_to_structures(velocity.x, "mean")
    assert torch.allclose(mean, torch.zeros(2, 3), atol=1e-6)


def test_depends_on_the_time(model):
    structures, state = prepared(make_batch())
    velocity = model.velocity(structures, state, TIME)
    later = model.velocity(structures, state, torch.tensor(0.6))

    assert not torch.allclose(later.bits, velocity.bits)


def test_needs_its_graph(model):
    structures = make_batch()
    with pytest.raises(ValueError, match=r"apply FlowMM.transform\(\)"):
        model.velocity(structures, model.state_of(structures), TIME)

    graph = FlowMM.transform()(structures).graph
    assert graph is not None
    assert torch.equal(
        graph.num_edges, structures.num_atoms * (structures.num_atoms - 1)
    )


def test_reads_the_graph_attached(model):
    structures = make_batch()
    state = model.state_of(structures)
    velocity = model.velocity(FlowMM.transform()(structures), state, TIME)
    other = model.velocity(AddCompleteGraph(self_loops=True)(structures), state, TIME)

    # an edge from an atom to itself has no direction: its cosines are not defined
    assert not torch.allclose(other.x.nan_to_num(), velocity.x, atol=1e-4)


def test_batch_equals_single(model):
    items = [*make_structures(), Structures.diamond()]
    structures, state = prepared(Structures.collate(items))
    velocity = model.velocity(structures, state, TIME)

    singles = [model.velocity(*prepared(item), TIME) for item in items]
    for field in ("bits", "x", "lattice"):
        expected = torch.cat([getattr(single, field) for single in singles])
        assert torch.allclose(getattr(velocity, field), expected, atol=1e-5)


def test_a_structure_of_one_atom(model):
    # an atom has no edge: it only reads itself
    one = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[26]
    )
    structures, state = prepared(Structures.collate([one, Structures.diamond()]))

    velocity = model.velocity(structures, state, TIME)
    alone = model.velocity(*prepared(one), TIME)

    assert torch.equal(velocity.x[0], torch.zeros(3))
    assert torch.allclose(velocity.bits[:1], alone.bits, atol=1e-5)


def test_follows_a_permutation(model):
    batch = make_batch()
    permutation = sampling.random_permutation(batch.batch, generator())
    permuted = actions.permute_atoms(batch, permutation)

    velocity = model.velocity(*prepared(batch), TIME)
    moved = model.velocity(*prepared(permuted), TIME)

    assert torch.allclose(moved.bits, velocity.bits[permutation], atol=1e-4)
    assert torch.allclose(moved.x, velocity.x[permutation], atol=1e-4)
    assert torch.allclose(moved.lattice, velocity.lattice, atol=1e-4)


def test_translation_and_rotation_invariance(model):
    batch = make_batch()
    velocity = model.velocity(*prepared(batch), TIME)

    for augment in (
        AtomTranslation(generator=generator()),
        LatticeRotation(generator=generator()),
    ):
        moved = model.velocity(*prepared(augment(batch)), TIME)
        for field in ("bits", "x", "lattice"):
            assert torch.allclose(
                getattr(moved, field), getattr(velocity, field), atol=1e-3
            )


def test_annealing(model):
    structures, state = prepared(make_batch())
    time = torch.tensor(0.75)
    plain = model.velocity(structures, state, time)
    annealed = model.velocity(
        structures, state, time, inference_anneal_slope=4.0, inference_anneal_offset=0.5
    )

    # only the coordinates, by 1 + slope * max(t - offset, 0)
    assert torch.allclose(annealed.x, 2.0 * plain.x, atol=1e-6)
    assert torch.equal(annealed.bits, plain.bits)
    assert torch.equal(annealed.lattice, plain.lattice)
    early = model.velocity(structures, state, torch.tensor(0.25), 4.0, 0.5)
    assert torch.equal(early.x, model.velocity(structures, state, torch.tensor(0.25)).x)


# sampling with random weights


def test_sample_step(model):
    structures, state = prepared(make_batch())
    time, next_time = torch.tensor(0.3), torch.tensor(0.31)

    moved = model.sample_step(structures, state, time, next_time)
    velocity = model.velocity(structures, state, time)

    assert torch.allclose(moved.bits, state.bits + 0.01 * velocity.bits, atol=1e-6)
    assert torch.allclose(
        moved.lattice, state.lattice + 0.01 * velocity.lattice, atol=1e-6
    )
    assert_close_in_the_cell(moved.x, state.x + 0.01 * velocity.x)
    assert ((moved.x >= 0) & (moved.x < 1)).all()


def test_times(model):
    times = model.times()

    assert len(times) == model.config.sampling.num_steps + 1
    assert times[0] == 0.0 and times[-1] == 1.0 and times.dtype == torch.float32
    assert torch.equal(model.times(4), torch.linspace(0.0, 1.0, 5))


def test_initial_state(model):
    num_atoms = torch.tensor([3, 5, 20])
    state = model.initial_state(num_atoms, generator())
    flow = model.config.flow

    assert state.bits.shape == (28, 7) and state.x.shape == (28, 3)
    assert ((state.x >= 0) & (state.x < 1)).all()
    assert (state.lattice[:, :3] > 0).all()
    angles = model.angles_in_degrees(state.lattice[:, 3:])
    assert (angles > flow.angle_min).all() and (angles < flow.angle_max).all()

    again = model.initial_state(num_atoms, generator())
    assert torch.equal(again.bits, state.bits) and torch.equal(
        again.lattice, state.lattice
    )


def test_sample(model):
    num_atoms = torch.tensor([3, 1, 5])
    structures = model.sample(num_atoms, num_steps=3, generator=generator())

    assert torch.equal(structures.num_atoms, num_atoms)
    assert structures.lattice.shape == (3, 3, 3)
    assert ((structures.z >= 1) & (structures.z <= 128)).all()
    assert structures.graph is None

    # num_steps steps of integrate from the initial state
    state = model.integrate(model.initial_state(num_atoms, generator()), num_atoms, 3)
    assert torch.equal(model.to_structures(state, num_atoms).x, structures.x)


def test_sample_num_atoms(model):
    num_atoms = model.sample_num_atoms(2000, generator())

    assert num_atoms.min() >= 1 and num_atoms.max() <= 20
    assert (num_atoms == 4).sum() > (num_atoms == 3).sum()


# weights


def upstream_state_dict(model: FlowMM) -> dict[str, torch.Tensor]:
    """State dict as FlowMM saves it: the network and the average of its parameters."""
    upstream = model.config.upstream
    state = {"model.num_updates": torch.tensor(10)}
    state.update(
        {
            upstream.model_prefix + name[len("model.") :]: value
            for name, value in model.state_dict().items()
        }
    )
    state.update(
        {
            f"{upstream.ema_prefix}{i}": torch.full_like(value, 0.5)
            for i, (_, value) in enumerate(model.named_parameters())
        }
    )
    return state


def upstream_hparams(config: Config) -> dict:
    hparams: dict = {"base_distribution_from_data": False}
    for section, values in config.upstream.sections.items():
        node = hparams
        for key in section.split("."):
            node = node.setdefault(key, {})
        node.update(values)
    return hparams


def test_load_pretrained_reads_the_moving_average(model):
    other = FlowMM().eval()
    state = upstream_state_dict(model)
    other.load_pretrained(state, upstream_hparams(model.config))

    # the averages, not the parameters, and the buffers of the network
    assert (other.model.cspnet.lattice_out.weight == 0.5).all()
    assert torch.equal(other.model.lat_x_t_std, model.model.lat_x_t_std)


def test_load_pretrained_refuses_other_weights(model):
    other = FlowMM().eval()
    state = upstream_state_dict(model)
    hparams = upstream_hparams(model.config)

    changed = upstream_hparams(model.config)
    changed["model"]["manifold_getter"]["lattice_manifold"] = "non_symmetric"
    with pytest.raises(InvalidWeights, match="lattice_manifold"):
        other.load_pretrained(state, changed)
    with pytest.raises(InvalidWeights, match="base_distribution_from_data"):
        other.load_pretrained(state, {**hparams, "base_distribution_from_data": True})

    prefix = model.config.upstream.ema_prefix
    fewer = {name: value for name, value in state.items() if name != f"{prefix}0"}
    with pytest.raises(InvalidWeights, match="averages"):
        other.load_pretrained(fewer)

    missing = {
        name: value for name, value in state.items() if "lat_x_t_mean" not in name
    }
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(missing)


def test_a_model_follows_its_configuration():
    config = Config.load()
    small = replace(
        config,
        model=replace(
            config.model, hidden_dim=16, time_dim=8, num_layers=1, num_freqs=4
        ),
        sampling=replace(config.sampling, num_steps=2),
    )
    model = FlowMM(small).eval()

    assert model.config is small and len(model.times()) == 3
    assert model.sample(torch.tensor([2, 3])).num_atoms.tolist() == [2, 3]
    assert FlowMM.transform(small) == FlowMM.transform()


# with the weights of ESS-Flow, against FlowMM


@pytest.fixture(scope="module")
def pretrained() -> FlowMM:
    return load_pretrained(FlowMM, "flowmm", NAME)


@pytest.fixture(scope="module")
def reference():
    return load_reference("flowmm_reference.npz")


def recorded_state(reference, prefix: str) -> State:
    return State(
        *(reference[f"{prefix}_{k}"].float() for k in ("bits", "x", "lattice"))
    )


def test_representation_matches_reference(pretrained, reference):
    structures = reference_structures()
    state = pretrained.state_of(structures)

    assert torch.equal(state.bits, reference["crystals_bits"].float())
    assert torch.equal(state.x, reference["crystals_x"].float())
    # lengths and unconstrained angles: an angle near a bound has a large value
    assert_close(state.lattice, reference["crystals_lattice"], name="lattice")

    matrix = pretrained.lattice_matrix(reference["crystals_lattice"].float())
    assert_close(matrix, reference["crystals_matrix"], name="matrix")


def test_velocity_of_crystals_matches_reference(pretrained, reference):
    structures = reference_structures()
    state = recorded_state(reference, "crystals")
    times = reference["crystals_times"].float()

    found = []
    for i in range(len(structures.num_atoms)):
        one = pretrained.transform()(structures.select(torch.tensor([i])))
        atoms = slice(int(structures.ptr[i]), int(structures.ptr[i + 1]))
        found.append(
            pretrained.velocity(
                one,
                State(state.bits[atoms], state.x[atoms], state.lattice[i : i + 1]),
                times[i],
            )
        )

    for field in ("bits", "x", "lattice"):
        values = torch.cat([getattr(velocity, field) for velocity in found])
        assert_close(values, reference[f"crystals_v_{field}"], name=field)


def check_step(pretrained, reference, step: int) -> None:
    """The velocity and the step from a state FlowMM reached."""
    num_atoms = reference["sample_num_atoms"]
    structures = pretrained.transform()(
        pretrained.to_structures(recorded_state(reference, "sample_x0"), num_atoms)
    )
    times = pretrained.times()
    assert times[step] == reference[f"sample_step{step}_t"].float()
    state = recorded_state(reference, f"sample_step{step}_x")

    velocity = pretrained.velocity(structures, state, times[step])
    for field in ("bits", "x", "lattice"):
        found, expected = (
            getattr(velocity, field),
            reference[f"sample_step{step}_v_{field}"],
        )
        if field == "bits" and step in ROUNDING_STEPS:
            double = reference[f"sample_step{step}_double_v_{field}"]
            assert as_near_as_original(found, expected, double), step
        else:
            assert_close(found, expected, name=f"velocity of {field}")

    moved = pretrained.sample_step(structures, state, times[step], times[step + 1])
    following = recorded_state(reference, f"sample_step{step}_next")
    assert_close(moved.bits, following.bits, name="bits")
    assert_close(moved.lattice, following.lattice, name="lattice")
    assert_close_in_the_cell(moved.x, following.x)


def test_one_step_matches_reference(pretrained, reference):
    check_step(pretrained, reference, ONE_STEP)


@pytest.mark.full
def test_every_recorded_step_matches_reference(pretrained, reference):
    for step in reference["sample_steps"].tolist():
        check_step(pretrained, reference, step)


@pytest.mark.full
@pytest.mark.timeout(600)
def test_whole_sampling_matches_reference(pretrained, reference):
    # the 1000 steps of the flow, chained from the initial state of FlowMM
    num_atoms = reference["sample_num_atoms"]
    state = pretrained.integrate(recorded_state(reference, "sample_x0"), num_atoms)
    expected = recorded_state(reference, "sample_x1")

    assert_close(state.bits, expected.bits, name="bits")
    assert_close(state.lattice, expected.lattice, name="lattice")
    assert_close_in_the_cell(state.x, expected.x)
    assert torch.equal(pretrained.atomic_numbers(state.bits), reference["sample_z"])


def test_crystals_read_from_the_last_state(pretrained, reference):
    num_atoms = reference["sample_num_atoms"]
    structures = pretrained.to_structures(
        recorded_state(reference, "sample_x1"), num_atoms
    )

    assert torch.equal(structures.z, reference["sample_z"])
    assert_close(structures.lattice, reference["sample_matrix"], name="lattice")
