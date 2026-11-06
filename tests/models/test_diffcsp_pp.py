"""Tests for the rewrite of the two models of DiffCSP++."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from helpers.compare import assert_close, assert_close_in_the_cell
from helpers.references import load_reference, reference_structures
from helpers.structures import generator
from helpers.weights import pretrained as load_pretrained
from torchms.data import Structures
from torchms.data.constants import get_wyckoff_positions
from torchms.data.symmetry import WyckoffSites
from torchms.graph.geometry import lattice_parameters
from torchms.models.diffcsp_pp import (
    Config,
    CrystalFamily,
    DiffCSPPlusPlus,
    DiffCSPPlusPlusGeneration,
    InvalidWeights,
    Noise,
    State,
)
from torchms.models.estimator import weights

NAME = "diffcsp-pp-mp20"
GENERATION_NAME = "diffcsp-pp-gen-mp20"

# weights, their model and the file of what DiffCSP++ computed with them
REFERENCES = {
    "diffcsp-pp-mp20": (DiffCSPPlusPlus, "diffcsp_pp_mp20_reference.npz"),
    "diffcsp-pp-mpts52": (DiffCSPPlusPlus, "diffcsp_pp_mpts52_reference.npz"),
    "diffcsp-pp-perov5": (DiffCSPPlusPlus, "diffcsp_pp_perov5_reference.npz"),
    "diffcsp-pp-carbon24": (DiffCSPPlusPlus, "diffcsp_pp_carbon24_reference.npz"),
    "diffcsp-pp-gen-mp20": (
        DiffCSPPlusPlusGeneration,
        "diffcsp_pp_gen_mp20_reference.npz",
    ),
    "diffcsp-pp-gen-perov5": (
        DiffCSPPlusPlusGeneration,
        "diffcsp_pp_gen_perov5_reference.npz",
    ),
}

# the sites of the sampling of tests/references/reference_diffcsp_pp.py, one
# structure of each crystal family
SITES = [
    (58, ["2a", "2d", "4g"], [25, 3, 8]),
    (194, "abff", [69, 69, 28, 33]),
    (225, "ab", [11, 17]),
    (221, "abc", [38, 22, 8]),
    (12, "ai", [29, 8]),
    (2, "ii", [14, 8]),
    (166, "abc", [83, 52, 34]),
    (139, "ade", [56, 26, 33]),
]

# the recorded step checked by default, at the time 500: the other ones with --full
ONE_TIME = 500

# a network of 512 features on every pair of atoms: slower than the default timeout
pytestmark = pytest.mark.timeout(60)


def sites_of(indices: list[int] | None = None) -> tuple[WyckoffSites, torch.Tensor]:
    """Sites of some structures of SITES, and the atomic number of each site."""
    chosen = SITES if indices is None else [SITES[i] for i in indices]
    sites = WyckoffSites.from_letters([c[0] for c in chosen], [c[1] for c in chosen])

    return sites, torch.tensor([z for c in chosen for z in c[2]])


def complete(model, sites: WyckoffSites, z: torch.Tensor) -> Structures:
    """Structures of the sites with their complete graph, for the network."""
    return model.transform()(
        Structures(
            lattice=torch.eye(3).expand(len(sites.num_atoms), 3, 3),
            x=torch.zeros(len(z), 3),
            z=z,
            num_atoms=sites.num_atoms,
        )
    )


def as_sets(x: torch.Tensor, decimals: int = 4) -> set[tuple[float, ...]]:
    """Fractional coordinates as a set, modulo the cell."""
    wrapped = torch.round(x.double() % 1.0, decimals=decimals) % 1.0

    return {tuple(row) for row in wrapped.tolist()}


def short_diffusion(steps: int = 6) -> Config:
    """Configuration of a diffusion of a few steps."""
    config = Config.load()
    model = replace(
        config.model, schedules=replace(config.model.schedules, timesteps=steps)
    )

    return replace(config, model=model)


@pytest.fixture(scope="module")
def model() -> DiffCSPPlusPlus:
    """Model with random weights, and scores scaled as the ones of small noise."""
    torch.manual_seed(0)
    model = DiffCSPPlusPlus().eval().requires_grad_(False)
    sigmas = model.sigma_scheduler.sigmas
    model.sigma_scheduler.sigmas_norm[1:] = 1.0 / sigmas[1:] ** 2

    return model


@pytest.fixture(scope="module")
def generation() -> DiffCSPPlusPlusGeneration:
    torch.manual_seed(1)
    model = DiffCSPPlusPlusGeneration().eval().requires_grad_(False)
    sigmas = model.sigma_scheduler.sigmas
    model.sigma_scheduler.sigmas_norm[1:] = 1.0 / sigmas[1:] ** 2

    return model


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS["diffcsp_pp"]) == set(REFERENCES) | {
        "diffcsp-pp-carbon24"
    }
    assert Config.load().default_weights == NAME
    assert Config.load().generation.default_weights == GENERATION_NAME
    assert set(Config.load().sampling.step_sizes) == set(REFERENCES)

    for name in REFERENCES:
        for checkpoint in weights.CHECKPOINTS["diffcsp_pp"][name]:
            # a file of Google Drive by its identifier, not a folder or a name
            assert checkpoint.url.startswith(
                "https://drive.usercontent.google.com/download?id="
            )
            assert len(checkpoint.sha256) == 64

        paths = weights.checkpoint_paths("diffcsp_pp", name, tmp_path)
        assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
            f"diffcsp_pp/{name}/last.ckpt",
            f"diffcsp_pp/{name}/hparams.yaml",
        ]


# the lattices of the crystal families


def angles_and_lengths(lattice: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lengths, angles = lattice_parameters(lattice.double())
    return lengths[0], angles[0]


@pytest.mark.parametrize(
    "spacegroup, equal_lengths, right_angles, angle_120",
    [
        (1, [], [], False),
        (12, [], [0, 2], False),
        (58, [], [0, 1, 2], False),
        (139, [(0, 1)], [0, 1, 2], False),
        (194, [(0, 1)], [0, 1], True),
        (225, [(0, 1), (1, 2)], [0, 1, 2], False),
    ],
)
def test_lattice_of_a_crystal_family(
    spacegroup, equal_lengths, right_angles, angle_120
):
    family = CrystalFamily(Config.load().lattice)
    k = torch.randn(1, 6, generator=generator(spacegroup))
    k = family.project(k, torch.tensor([spacegroup]))
    lengths, angles = angles_and_lengths(family.lattice(k))

    for i, j in equal_lengths:
        assert torch.isclose(lengths[i], lengths[j])
    for i in right_angles:
        assert torch.isclose(angles[i], torch.tensor(90.0, dtype=torch.float64))
    if angle_120:
        assert torch.isclose(angles[2], torch.tensor(120.0, dtype=torch.float64))
    # the projection keeps a projected lattice
    assert torch.equal(family.project(k, torch.tensor([spacegroup])), k)


def test_numbers_of_a_lattice():
    family = CrystalFamily(Config.load().lattice)
    k = torch.randn(4, 6, generator=generator(1), dtype=torch.float64)
    family = family.double()

    assert torch.allclose(family.parameters_of(family.lattice(k)), k)

    # a rotation of the cell has the same numbers
    rotation = torch.linalg.qr(torch.randn(3, 3, generator=generator(2)).double())[0]
    rotated = family.lattice(k) @ rotation
    assert torch.allclose(family.parameters_of(rotated), k)


# the network, with random weights


def test_output(model):
    sites, z = sites_of([0, 1])
    z = sites.unpool_sites(z)
    structures = complete(model, sites, z)
    k = model.crystal_family.project(torch.randn(2, 6), sites.spacegroup)
    x = sites.place(torch.rand(len(z), 3))

    found = model(structures, State(k, x), torch.tensor([3, 700]))
    assert found.lattice.shape == (2, 6) and found.x.shape == (len(z), 3)
    assert found.elements is None
    assert found.lattice.dtype == torch.float32


def test_batch_equals_single(model):
    sites, z = sites_of([0, 1])
    z = sites.unpool_sites(z)
    k = torch.randn(2, 6, generator=generator(3))
    x = torch.rand(len(z), 3, generator=generator(4))
    times = torch.tensor([3, 700])
    together = model(complete(model, sites, z), State(k, x), times)

    first = int(sites.num_atoms[0])
    for i, atoms in enumerate([slice(0, first), slice(first, None)]):
        alone, alone_z = sites_of([i])
        found = model(
            complete(model, alone, alone.unpool_sites(alone_z)),
            State(k[i : i + 1], x[atoms]),
            times[i : i + 1],
        )
        assert_close(found.lattice, together.lattice[i : i + 1])
        assert_close(found.x, together.x[atoms])


def test_translation_and_permutation(model):
    sites, z = sites_of([0])
    z = sites.unpool_sites(z)
    structures = complete(model, sites, z)
    k = torch.randn(1, 6, generator=generator(5))
    x = torch.rand(len(z), 3, generator=generator(6))
    times = torch.tensor([400])
    found = model(structures, State(k, x), times)

    moved = model(structures, State(k, (x + torch.tensor([0.3, 0.1, 0.7])) % 1), times)
    assert_close(moved.lattice, found.lattice)
    assert_close(moved.x, found.x)

    order = torch.randperm(len(z), generator=generator(7))
    shuffled = complete(model, sites, z[order])
    permuted = model(shuffled, State(k, x[order]), times)
    assert_close(permuted.lattice, found.lattice)
    assert_close(permuted.x, found.x[order])


def test_needs_its_graph(model):
    sites, z = sites_of([2])
    z = sites.unpool_sites(z)
    structures = Structures(
        lattice=torch.eye(3)[None], x=torch.zeros(8, 3), z=z, num_atoms=sites.num_atoms
    )
    state = State(torch.zeros(1, 6), torch.zeros(8, 3))

    with pytest.raises(ValueError, match="transform"):
        model(structures, state, torch.tensor([1]))


def test_unknown_element(model):
    sites, _ = sites_of([2])
    with pytest.raises(ValueError, match="not supported"):
        model.sample(sites, torch.tensor([11, 101]))


# the steps keep the symmetry


def assert_symmetric(sites: WyckoffSites, state: State, model) -> None:
    """Each structure is a crystal of its space group, on its Wyckoff positions."""
    assert torch.equal(
        model.crystal_family.project(state.lattice, sites.spacegroup), state.lattice
    )
    assert_close_in_the_cell(sites.place(state.x), state.x, atol=1e-5)

    ptr = torch.cat([torch.zeros(1, dtype=torch.long), sites.num_atoms.cumsum(0)])
    for i, spacegroup in enumerate(sites.spacegroup.tolist()):
        x = state.x[ptr[i] : ptr[i + 1]].double()
        group = get_wyckoff_positions(spacegroup)[0].operations
        images = (
            torch.einsum("gij, aj -> gai", group[:, :3, :3], x) + group[:, None, :3, 3]
        )
        assert as_sets(images.reshape(-1, 3), 3) == as_sets(x, 3)


def test_sample_step_keeps_the_symmetry(model):
    sites, z = sites_of()
    z = sites.unpool_sites(z)
    structures = complete(model, sites, z)
    state = model.initial_state(sites, generator(8))
    assert_symmetric(sites, state, model)

    for time in (900, 2):
        noise = model.step_noise(sites, time, generator(time))
        moved = model.sample_step(structures, sites, state, time, noise)
        assert_symmetric(sites, moved, model)
        assert not torch.allclose(moved.x, state.x)


def test_step_size_of_the_weights(model):
    assert model.weights is None
    assert model.step_size() == Config.load().sampling.step_sizes[NAME]
    assert model.step_size(2e-6) == 2e-6

    other = DiffCSPPlusPlus()
    other.weights = "diffcsp-pp-perov5"
    assert other.step_size() == 5e-7
    assert DiffCSPPlusPlusGeneration().step_size() == 5e-6


def test_sample():
    torch.manual_seed(0)
    model = DiffCSPPlusPlus(short_diffusion()).eval().requires_grad_(False)
    sites, z = sites_of([0, 2])

    structures = model.sample(sites, z, generator=generator())
    assert structures.graph is None
    assert structures.lattice.shape == (2, 3, 3)
    assert torch.equal(structures.z, sites.unpool_sites(z))
    assert torch.equal(structures.num_atoms, sites.num_atoms)
    assert (structures.x >= 0).all() and (structures.x < 1).all()
    # a symmetric lattice; the one of the rock salt is cubic
    assert torch.allclose(structures.lattice, structures.lattice.mT, atol=1e-5)
    cubic = structures.lattice[1]
    assert torch.allclose(cubic, cubic[0, 0] * torch.eye(3), atol=1e-5)

    again = model.sample(sites, z, generator=generator())
    assert torch.equal(again.x, structures.x)
    other = model.sample(sites, z, generator=generator(1))
    assert not torch.equal(other.lattice, structures.lattice)


# the generation, with random weights


def test_generation_output(generation):
    sites, z = sites_of([0])
    z = sites.unpool_sites(z)
    structures = complete(generation, sites, torch.ones_like(z))
    state = State(
        torch.zeros(1, 6), sites.place(torch.rand(8, 3)), generation.element_vectors(z)
    )

    found = generation(structures, state, torch.tensor([10]))
    assert found.elements is not None and found.elements.shape == (8, 100)

    # the atomic numbers of the structures are not read
    other = generation(complete(generation, sites, z), state, torch.tensor([10]))
    assert torch.equal(other.elements, found.elements)

    with pytest.raises(ValueError, match="vectors of the elements"):
        generation(structures, State(state.lattice, state.x), torch.tensor([10]))
    with pytest.raises(ValueError, match="not supported"):
        generation.element_vectors(torch.tensor([0]))


def test_generation_step_shares_the_elements_of_a_site(generation):
    sites, z = sites_of([0, 6])
    structures = complete(generation, sites, sites.unpool_sites(torch.ones_like(z)))
    state = generation.initial_state(sites, generator(9))
    assert state.elements is not None
    assert torch.equal(
        state.elements, sites.unpool_sites(sites.anchors(state.elements))
    )

    noise = generation.step_noise(sites, 500, generator(10))
    moved = generation.sample_step(structures, sites, state, 500, noise)
    assert moved.elements is not None
    assert torch.equal(
        moved.elements, sites.unpool_sites(sites.anchors(moved.elements))
    )
    assert not torch.allclose(moved.elements, state.elements)
    assert_symmetric(sites, moved, generation)


def test_generation_sample():
    torch.manual_seed(0)
    model = DiffCSPPlusPlusGeneration(short_diffusion()).eval().requires_grad_(False)
    sites, _ = sites_of([1, 3])

    structures = model.sample(sites, generator=generator())
    assert torch.equal(structures.num_atoms, sites.num_atoms)
    # one element per site
    assert torch.equal(structures.z, sites.unpool_sites(sites.anchors(structures.z)))
    assert ((structures.z >= 1) & (structures.z <= 100)).all()


# the weights


def upstream_hparams(generation: bool = False) -> dict[str, object]:
    """Configuration of the training of DiffCSP++ a rewrite is written for."""
    arguments = Config.load().upstream.sections(generation)

    return {
        "model": {
            **arguments["model"],
            "decoder": dict(arguments["model.decoder"]),
            "beta_scheduler": dict(arguments["model.beta_scheduler"]),
            "sigma_scheduler": dict(arguments["model.sigma_scheduler"]),
        },
    }


def test_load_pretrained_checks_the_parameters(model, generation):
    state = dict(model.state_dict())
    other = DiffCSPPlusPlus()
    other.load_pretrained(state, upstream_hparams())
    assert torch.equal(
        other.decoder.lattice_out.weight, model.decoder.lattice_out.weight
    )

    # the weights of one model are refused by the other one
    with pytest.raises(InvalidWeights, match="diffusion_w_type"):
        other.load_pretrained(state, upstream_hparams(generation=True))
    with pytest.raises(InvalidWeights):
        DiffCSPPlusPlusGeneration().load_pretrained(state)
    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state, "decoder.other": torch.zeros(1)})
    del state["crystal_family.basis"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state)

    changed = upstream_hparams()
    changed["model"]["decoder"]["num_freqs"] = 10
    with pytest.raises(InvalidWeights, match="num_freqs"):
        other.load_pretrained(dict(model.state_dict()), changed)

    DiffCSPPlusPlusGeneration().load_pretrained(
        dict(generation.state_dict()), upstream_hparams(generation=True)
    )


# with the pretrained weights, against DiffCSP++


@pytest.fixture(scope="module", params=list(REFERENCES))
def weights_name(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def pretrained(weights_name):
    cls, _ = REFERENCES[weights_name]
    return load_pretrained(cls, "diffcsp_pp", weights_name)


@pytest.fixture(scope="module")
def reference(weights_name):
    return load_reference(REFERENCES[weights_name][1])


def generates(model) -> bool:
    return isinstance(model, DiffCSPPlusPlusGeneration)


def test_buffers_are_the_ones_of_the_configuration(pretrained):
    """The crystal families and the schedules of the weights are the computed ones."""
    computed = type(pretrained)()

    for name, values in computed.crystal_family.state_dict().items():
        assert torch.equal(pretrained.crystal_family.state_dict()[name], values), name
    for name in ("betas", "alphas", "alphas_cumprod", "sigmas"):
        found = getattr(pretrained.beta_scheduler, name)
        assert torch.allclose(found, getattr(computed.beta_scheduler, name), atol=1e-7)
    assert torch.equal(
        pretrained.sigma_scheduler.sigmas, computed.sigma_scheduler.sigmas
    )


def test_numbers_of_the_lattices_match_reference(pretrained, reference):
    structures = reference_structures()
    k = pretrained.crystal_family.parameters_of(structures.lattice.float())

    assert_close(k, reference["k"], name="k")


def test_network_matches_reference(pretrained, reference):
    structures = pretrained.transform()(reference_structures())
    times = reference["times"]
    x = structures.x.float()

    if generates(pretrained):
        clean = State(reference["k"], x, pretrained.element_vectors(structures.z))
        noisy = State(reference["noisy_k"], reference["noisy_x"], reference["noisy_t"])
    else:
        clean = State(reference["k"], x)
        noisy = State(reference["noisy_k"], reference["noisy_x"])

    for state, suffix in ((clean, ""), (noisy, "_noisy")):
        found = pretrained(structures, state, times)
        assert_close(
            found.lattice, reference[f"pred_l{suffix}"], name=f"pred_l{suffix}"
        )
        assert_close(found.x, reference[f"pred_x{suffix}"], name=f"pred_x{suffix}")
        if generates(pretrained):
            assert_close(
                found.elements, reference[f"pred_t{suffix}"], name=f"pred_t{suffix}"
            )


def test_sites_match_reference(weights_name, reference):
    """The operations DiffCSP++ takes from pyxtal are the ones of the table."""
    sites, z = sites_of()

    assert torch.equal(sites.spacegroup, reference["spacegroup"])
    assert torch.equal(sites.num_atoms, reference["num_atoms"])
    assert torch.equal(sites.operations, reference["ops"])
    assert torch.equal(sites.pseudo_inverses, reference["ops_inv"])
    assert torch.equal(
        sites.unpool_sites(sites.site_ptr[:-1]), reference["anchor_index"]
    )

    # the elements of the sites, none for the generation, carbon for Carbon-24
    if REFERENCES[weights_name][0] is DiffCSPPlusPlusGeneration:
        z = torch.zeros_like(z)
    elif "carbon" in weights_name:
        z = torch.full_like(z, 6)
    assert torch.equal(sites.unpool_sites(z), reference["atom_types"])


def recorded_step(pretrained, reference, index: int):
    """Sites, structures, state, noise and next state of a recorded step."""
    if generates(pretrained):
        structures_of = reference["generation_atoms"]
        chosen = [0, 3, 6]
        sites, _ = sites_of(chosen)
        rows = torch.tensor(chosen)
        x = reference["state_x"][index][structures_of]
        next_x = reference["next_x"][index][structures_of]
        state = State(reference["state_k"][index][rows], x, reference["state_t"][index])
        noise = Noise(
            reference["noise_corrector"][index][structures_of],
            reference["noise_lattice"][index][rows],
            reference["noise_predictor"][index][structures_of],
            reference["noise_elements"][index],
        )
        z = torch.ones(len(sites.site), dtype=torch.long)
        following = (reference["next_k"][index][rows], next_x)
    else:
        sites, _ = sites_of()
        state = State(reference["state_k"][index], reference["state_x"][index])
        noise = Noise(
            reference["noise_corrector"][index],
            reference["noise_lattice"][index],
            reference["noise_predictor"][index],
        )
        z = reference["atom_types"]
        following = (reference["next_k"][index], reference["next_x"][index])

    return sites, complete(pretrained, sites, z), state, noise, following


def check_step(pretrained, reference, index: int) -> None:
    time = int(reference["saved_times"][index])
    sites, structures, state, noise, (next_k, next_x) = recorded_step(
        pretrained, reference, index
    )
    moved = pretrained.sample_step(structures, sites, state, time, noise)

    assert_close(moved.lattice, next_k, name=f"k at {time}")
    assert_close_in_the_cell(moved.x, next_x, name=f"x at {time}")
    if generates(pretrained) and time > 1:
        assert_close(
            moved.elements, reference["next_t"][index], name=f"elements at {time}"
        )
    if generates(pretrained) and time == 1:
        atoms = reference["generation_atoms"]
        assert torch.equal(
            pretrained.atomic_numbers(moved.elements), reference["final_z"][atoms]
        )


def test_initial_state_matches_reference(pretrained, reference):
    sites, _ = sites_of()
    initial = State(
        pretrained.crystal_family.project(reference["initial_k"], sites.spacegroup),
        sites.place(reference["initial_x"]),
        (
            sites.unpool_sites(sites.anchors(reference["initial_t"]))
            if generates(pretrained)
            else None
        ),
    )

    assert torch.equal(initial.lattice, reference["state_k"][0])
    assert torch.equal(initial.x, reference["state_x"][0])
    if generates(pretrained):
        atoms = reference["generation_atoms"]
        assert torch.equal(initial.elements[atoms], reference["state_t"][0])


def test_one_step_matches_reference(pretrained, reference):
    times = reference["saved_times"].tolist()
    check_step(pretrained, reference, times.index(ONE_TIME))


@pytest.mark.full
def test_every_recorded_step_matches_reference(pretrained, reference):
    times = reference["saved_times"].tolist()
    assert times[0] == pretrained.timesteps - 1 and times[-1] == 1

    for index in range(len(times)):
        check_step(pretrained, reference, index)


@pytest.mark.full
def test_last_steps_chained_match_reference(pretrained, reference):
    """The last two steps of DiffCSP++, one after the other, give its result.

    The first steps multiply the rounding: a relative change of 1e-7 of the
    numbers of the lattice at the time 999 moves them by 1e-2 three steps
    later. The last ones do not.
    """
    times = reference["saved_times"].tolist()
    assert times[-2:] == [2, 1]

    sites, structures, state, _, _ = recorded_step(pretrained, reference, -2)
    for index in (-2, -1):
        _, _, _, noise, following = recorded_step(pretrained, reference, index)
        state = pretrained.sample_step(structures, sites, state, times[index], noise)

    assert_close(state.lattice, following[0], name="k")
    assert_close_in_the_cell(state.x, following[1])
    if generates(pretrained):
        atoms = reference["generation_atoms"]
        z = pretrained.atomic_numbers(state.elements)
        assert torch.equal(z, reference["final_z"][atoms])


def whole_sampling(pretrained, reference, change: float = 0.0) -> State:
    """The whole sampling of DiffCSP++ from its recorded draws, step by step."""
    chosen = reference["whole_structures"].tolist()
    sites, z = sites_of(chosen)
    if "carbon" in pretrained.weights:
        z = torch.full_like(z, 6)
    anchors = reference["whole_anchors"]

    def at_anchors(values: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(len(sites.site), 3)
        full[anchors] = values
        return full

    k = reference["whole_initial_k"] * (1.0 + change)
    state = State(
        pretrained.crystal_family.project(k, sites.spacegroup),
        sites.place(at_anchors(reference["whole_initial_x"])),
    )
    structures = complete(pretrained, sites, sites.unpool_sites(z))
    for index, time in enumerate(range(pretrained.timesteps - 1, 0, -1)):
        noise = Noise(
            at_anchors(reference["whole_corrector"][index]),
            reference["whole_lattice"][index],
            at_anchors(reference["whole_predictor"][index]),
        )
        state = pretrained.sample_step(structures, sites, state, time, noise)

    return state


@pytest.mark.full
@pytest.mark.timeout(1200)
def test_whole_sampling_is_as_near_to_diffcsp_pp_as_its_rounding(pretrained, reference):
    # the sampling multiplies the rounding: in double precision a relative
    # change of 1e-12 of the first lattice moves the last one by 5e-5 for the
    # monoclinic structure and by 0.1 for the triclinic one, and a sampling
    # can end at one of two near results. The rewrite is held, structure by
    # structure, to three times the largest of the gap between DiffCSP++ in
    # single and in double precision and of the moves of relative changes of
    # 1e-6 and -1e-6 of its first lattice; every step above to 1e-5
    if generates(pretrained):
        pytest.skip("the whole sampling is recorded for the models given the elements")

    found = whole_sampling(pretrained, reference)
    changed = [whole_sampling(pretrained, reference, c) for c in (1e-6, -1e-6)]
    expected = (reference["whole_final_k"], reference["whole_final_x"])
    double = (reference["whole_final_k_double"], reference["whole_final_x_double"])

    sites, _ = sites_of(reference["whole_structures"].tolist())
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), sites.num_atoms.cumsum(0)])

    def distance(a: torch.Tensor, b: torch.Tensor, in_cell: bool) -> float:
        difference = a.double() - b.double()
        if in_cell:
            difference = (difference + 0.5) % 1.0 - 0.5
        return float(difference.abs().max())

    for i in range(len(sites.num_atoms)):
        rows = [slice(i, i + 1), slice(int(ptr[i]), int(ptr[i + 1]))]
        for field, (name, rows_of, in_cell) in enumerate(
            (("lattice", rows[0], False), ("x", rows[1], True))
        ):
            mine = getattr(found, name)[rows_of]
            gap = max(
                [distance(expected[field][rows_of], double[field][rows_of], in_cell)]
                + [
                    distance(getattr(other, name)[rows_of], mine, in_cell)
                    for other in changed
                ]
            )
            difference = distance(mine, expected[field][rows_of], in_cell)
            assert difference <= 3 * gap, (name, i, difference, gap)
