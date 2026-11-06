"""Tests for the rewrite of ADiT, the All-atom Diffusion Transformer."""

from dataclasses import replace

import pytest
import torch

from helpers.compare import assert_close
from helpers.references import load_reference, reference_structures
from helpers.structures import generator, make_batch, make_structures
from helpers.weights import pretrained as load_pretrained
from torchms.data import Structures
from torchms.data.augmentation import LatticeRotation
from torchms.graph.geometry import lattice_from_parameters
from torchms.models.adit import ADiT, Config, Decoded, InvalidWeights
from torchms.models.estimator import weights

NAME = "adit-qm9-mp20"

# the recorded step checked by default, in the middle of the flow: the other
# ones and the whole sampling with --full
ONE_STEP = 50

# transformers of 512 and 768 features, and a checkpoint of 2.5 GB to read
pytestmark = pytest.mark.timeout(120)


@pytest.fixture(scope="module")
def model() -> ADiT:
    """Model with random weights."""
    torch.manual_seed(0)
    return ADiT().eval().requires_grad_(False)


def crystals_and_molecules() -> list[Structures]:
    """Two crystals and two molecules."""
    methane = Structures.molecule(
        [[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
         [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]],
        [6, 1, 1, 1, 1],
    )  # fmt: skip
    return [*make_structures(), Structures.water(), methane]


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS["adit"]) == {NAME}
    assert Config.load().default_weights == NAME

    (checkpoint,) = weights.CHECKPOINTS["adit"][NAME]
    # a commit of the repository on Hugging Face, not a branch
    assert checkpoint.url.startswith(
        "https://huggingface.co/chaitjo/all-atom-diffusion-transformer/resolve/"
    )
    assert "/main/" not in checkpoint.url
    assert len(checkpoint.sha256) == 64

    (path,) = weights.checkpoint_paths("adit", NAME, tmp_path)
    assert path.relative_to(tmp_path).as_posix() == "adit/ldm.ckpt"


def test_reads_no_graph(model):
    assert ADiT.GRAPH_FIELDS == ()
    structures = make_batch()
    assert ADiT.transform()(structures) is structures
    assert model.encode(structures).mean.shape == (len(structures.z), 8)


# the autoencoder with random weights


def test_encode(model):
    structures = Structures.collate(crystals_and_molecules())
    posterior = model.encode(structures)
    latent_dim = model.config.autoencoder.latent_dim

    assert (
        posterior.mean.shape
        == posterior.logvar.shape
        == (len(structures.z), latent_dim)
    )
    assert posterior.mean.dtype == torch.float32
    assert posterior.mean.isfinite().all()

    noise = torch.randn(posterior.mean.shape, generator=generator())
    drawn = posterior.sample(noise)
    assert torch.allclose(
        drawn, posterior.mean + (0.5 * posterior.logvar).exp() * noise
    )


def test_logvar_is_clamped(model):
    structures = make_batch()
    config = model.config.autoencoder
    clamped = replace(config, logvar_min=0.0, logvar_max=0.0)
    other = (
        ADiT(replace(model.config, autoencoder=clamped)).eval().requires_grad_(False)
    )
    other.load_state_dict(model.state_dict())

    assert torch.equal(
        other.encode(structures).logvar, torch.zeros(len(structures.z), 8)
    )
    assert torch.equal(other.encode(structures).mean, model.encode(structures).mean)


def test_batch_equals_single(model):
    items = crystals_and_molecules()
    batch = Structures.collate(items)

    posterior = model.encode(batch)
    singles = [model.encode(item) for item in items]
    assert torch.allclose(
        posterior.mean, torch.cat([s.mean for s in singles]), atol=1e-5
    )

    decoded = model.decode_outputs(posterior.mean, batch.num_atoms)
    alone = [
        model.decode_outputs(single.mean, item.num_atoms)
        for single, item in zip(singles, items)
    ]
    for field in ("atom_types", "lattices", "frac_coords", "pos"):
        expected = torch.cat([getattr(one, field) for one in alone])
        assert torch.allclose(getattr(decoded, field), expected, atol=1e-5)


def test_order_of_the_atoms_is_read(model):
    # the index of an atom in its structure is embedded: not a permutation
    # invariant model
    structures = make_batch()
    first = int(structures.num_atoms[0])
    swapped = torch.arange(len(structures.z))
    swapped[[0, first - 1]] = swapped[[first - 1, 0]]
    permuted = Structures(
        lattice=structures.lattice,
        x=structures.x[swapped],
        z=structures.z[swapped],
        num_atoms=structures.num_atoms,
    )

    mean = model.encode(structures).mean
    permuted_mean = model.encode(permuted).mean
    assert not torch.allclose(permuted_mean[swapped], mean, atol=1e-4)


def test_reads_positions_and_fractional_coordinates(model):
    batch = make_batch()
    rotated = LatticeRotation(generator=generator())(batch)

    # the positions are read in Å, not invariant by a rotation
    assert torch.equal(rotated.x, batch.x)
    assert not torch.allclose(
        model.encode(rotated).mean, model.encode(batch).mean, atol=1e-4
    )


def test_a_molecule_is_read_from_its_positions(model):
    water = Structures.water()
    rotation = LatticeRotation(generator=generator())(water).lattice
    framed = Structures(
        lattice=rotation,
        x=water.positions() @ torch.linalg.inv(rotation[0]),
        z=water.z,
        pbc=water.pbc,
    )
    assert torch.allclose(framed.positions(), water.positions(), atol=1e-5)

    assert torch.allclose(
        model.encode(framed).mean, model.encode(water).mean, atol=1e-4
    )


def test_crystals_and_molecules_are_told_apart(model):
    # the same atoms, read as a molecule, have no fractional coordinates
    crystal = Structures.diamond()
    molecule = Structures(
        lattice=crystal.lattice,
        x=crystal.x,
        z=crystal.z,
        pbc=torch.zeros(1, 3, dtype=torch.bool),
    )
    assert torch.equal(molecule.positions(), crystal.positions())

    assert not torch.allclose(model.encode(molecule).mean, model.encode(crystal).mean)
    assert model.is_crystal(Structures.collate([crystal, molecule])).tolist() == [
        True,
        False,
    ]


def test_refuses_slabs_and_unknown_elements(model):
    slab = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 3.0,
        x=torch.zeros(1, 3),
        z=[6],
        pbc=torch.tensor([[True, True, False]]),
    )
    with pytest.raises(ValueError, match="not slabs"):
        model.encode(slab)

    # einsteinium is the last row of the embedding, then fermium has none
    assert model.config.autoencoder.num_elements == 100
    model.encode(
        Structures(lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[99])
    )
    for z in (0, 100):
        structures = Structures(
            lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[z]
        )
        with pytest.raises(ValueError, match="not supported"):
            model.encode(structures)


def test_to_structures(model):
    num_atoms = torch.tensor([2, 8, 1])
    crystals = torch.tensor([True, False, True])
    random = generator()
    atom_types = torch.randn(11, 100, generator=random)
    atom_types[0, 0] = 100.0  # a masked atom: hydrogen
    atom_types[1, 26] = 100.0
    lattices = torch.tensor(
        [
            [1.0, 1.5, 2.0, 1.5, 1.6, 1.7],
            [9.0] * 6,
            [2.0, 2.0, 2.0, 1.5708, 1.5708, 1.5708],
        ]
    )
    decoded = Decoded(
        atom_types=atom_types,
        lattices=lattices,
        frac_coords=torch.rand(11, 3, generator=random) * 3 - 1,
        pos=torch.randn(11, 3, generator=random),
    )

    structures = model.to_structures(decoded, num_atoms, crystals)

    assert torch.equal(structures.num_atoms, num_atoms)
    assert structures.pbc.tolist() == [[True] * 3, [False] * 3, [True] * 3]
    assert structures.z[:2].tolist() == [1, 26]
    assert ((structures.z >= 1) & (structures.z < 100)).all()

    # a crystal: the cell of the lengths times the cubic root of the number of atoms
    expected = lattice_from_parameters(
        lattices[[0, 2], :3] * torch.tensor([[2.0 ** (1 / 3)], [1.0]]),
        torch.rad2deg(lattices[[0, 2], 3:]),
    )
    assert torch.allclose(structures.lattice[[0, 2]], expected, atol=1e-5)
    crystal_atoms = torch.tensor([True] * 2 + [False] * 8 + [True])
    assert torch.allclose(
        structures.x[crystal_atoms], decoded.frac_coords[crystal_atoms] % 1.0
    )
    assert (
        (structures.x[crystal_atoms] >= 0) & (structures.x[crystal_atoms] < 1)
    ).all()

    # a molecule: the positions in nm, in the frame of the identity
    assert torch.equal(structures.lattice[1], torch.eye(3))
    assert torch.allclose(
        structures.positions()[~crystal_atoms], 10.0 * decoded.pos[~crystal_atoms]
    )


def test_reconstruction(model):
    structures = Structures.collate(crystals_and_molecules())
    decoded = model(structures)

    assert torch.equal(decoded.num_atoms, structures.num_atoms)
    assert torch.equal(decoded.pbc, structures.pbc)


# the diffusion with random weights


def test_denoise_and_guidance(model):
    num_atoms = torch.tensor([3, 5])
    crystals = torch.tensor([True, False])
    latent = torch.randn(8, 8, generator=generator())
    time = torch.tensor(0.4)

    guided = model.denoise(latent, num_atoms, crystals, time)
    assert guided.shape == latent.shape and guided.isfinite().all()

    # the scale of the guidance mixes the prediction of the dataset with the
    # one without: 1 is the first one alone
    conditional = model.denoise(latent, num_atoms, crystals, time, cfg_scale=1.0)
    unconditional = model.denoise(latent, num_atoms, crystals, time, cfg_scale=0.0)
    scale = model.config.sampling.cfg_scale
    assert torch.allclose(
        guided, unconditional + scale * (conditional - unconditional), atol=1e-5
    )

    # the dataset of a structure is read, and only for this structure
    other = model.denoise(latent, num_atoms, torch.tensor([True, True]), time)
    assert torch.equal(other[:3], guided[:3])
    assert not torch.allclose(other[3:], guided[3:])
    unguided = model.denoise(
        latent, num_atoms, torch.tensor([True, True]), time, cfg_scale=0.0
    )
    assert torch.equal(unguided, unconditional)

    # the time and the prediction of the step before are read
    assert not torch.allclose(
        model.denoise(latent, num_atoms, crystals, torch.tensor(0.5)), guided
    )
    previous = model.denoise(latent, num_atoms, crystals, time, latent)
    assert not torch.allclose(previous, guided)


def test_denoise_does_not_depend_on_the_batch(model):
    num_atoms = torch.tensor([3, 7, 2])
    crystals = torch.tensor([True, False, True])
    latent = torch.randn(12, 8, generator=generator())
    previous = torch.randn(12, 8, generator=generator(1))
    time = torch.tensor(0.7)

    together = model.denoise(latent, num_atoms, crystals, time, previous)
    alone = [
        model.denoise(
            latent[start:stop],
            num_atoms[i : i + 1],
            crystals[i : i + 1],
            time,
            previous[start:stop],
        )
        for i, (start, stop) in enumerate([(0, 3), (3, 10), (10, 12)])
    ]

    assert torch.allclose(together, torch.cat(alone), atol=1e-5)


def test_sample_step(model):
    num_atoms = torch.tensor([3, 5])
    crystals = torch.tensor([True, False])
    latent = torch.randn(8, 8, generator=generator())
    time, next_time = torch.tensor(0.3), torch.tensor(0.32)

    moved, prediction = model.sample_step(latent, num_atoms, crystals, time, next_time)

    assert torch.equal(prediction, model.denoise(latent, num_atoms, crystals, time))
    expected = latent + (prediction - latent) / (1 - time) * (next_time - time)
    assert torch.allclose(moved, expected)


def test_times(model):
    times = model.times()
    sampling = model.config.sampling

    assert len(times) == sampling.num_timesteps and times.dtype == torch.float32
    assert times[0] == torch.tensor(sampling.min_t) and times[-1] == 1.0
    assert torch.equal(model.times(5), torch.linspace(sampling.min_t, 1.0, 5))


def test_initial_noise(model):
    num_atoms = torch.tensor([2, 4])
    noise = torch.randn(2, 4, 8, generator=generator())

    centered = model.center_noise(noise, num_atoms)
    assert centered.shape == (6, 8)

    # centered over each row of the padded layout, the padding included
    assert torch.allclose(centered[2:].mean(dim=0), torch.zeros(8), atol=1e-6)
    assert torch.allclose(centered[:2], noise[0, :2] - noise[0].mean(dim=0))
    assert not torch.allclose(centered[:2].mean(dim=0), torch.zeros(8), atol=1e-3)

    drawn = model.initial_noise(num_atoms, generator(3))
    assert torch.equal(
        drawn,
        model.center_noise(torch.randn(2, 4, 8, generator=generator(3)), num_atoms),
    )


def test_sample(model):
    num_atoms = torch.tensor([3, 5])
    crystals = torch.tensor([True, False])

    structures = model.sample(
        num_atoms, crystals, num_timesteps=3, generator=generator()
    )

    assert torch.equal(structures.num_atoms, num_atoms)
    assert structures.pbc[:, 0].tolist() == [True, False]
    again = model.sample(num_atoms, crystals, num_timesteps=3, generator=generator())
    assert torch.equal(again.x, structures.x) and torch.equal(again.z, structures.z)

    # one value for all of them
    assert model.sample(num_atoms, False, num_timesteps=2).pbc.any().item() is False


def test_sample_latents(model):
    num_atoms = torch.tensor([3, 5])
    crystals = torch.tensor([True, False])
    noise = model.initial_noise(num_atoms, generator())
    times = model.times(3)

    latent, previous = model.sample_step(noise, num_atoms, crystals, times[0], times[1])
    latent, previous = model.sample_step(
        latent, num_atoms, crystals, times[1], times[2], previous
    )
    expected = model.denoise(latent, num_atoms, crystals, times[2], previous)

    found = model.sample_latents(noise, num_atoms, crystals, num_timesteps=3)
    assert torch.allclose(found, expected, atol=1e-6)


def test_sample_num_atoms(model):
    crystals = model.sample_num_atoms(2000, crystals=True, generator=generator())
    molecules = model.sample_num_atoms(2000, crystals=False, generator=generator())

    assert crystals.dtype == torch.int64
    assert crystals.min() >= 1 and crystals.max() <= 20
    assert molecules.min() >= 3 and molecules.max() <= 29
    # QM9 has no molecule of 28 atoms
    assert not (molecules == 28).any()


# weights


def upstream_hparams(config: Config) -> dict:
    upstream = config.upstream
    return {
        "denoiser": dict(upstream.denoiser),
        "interpolant": dict(upstream.interpolant),
        "conditioning": dict(upstream.conditioning),
    }


def test_load_pretrained(model):
    other = ADiT().eval().requires_grad_(False)
    state = {
        **model.state_dict(),
        "autoencoder.loss_weights_kl": torch.ones(2),
    }
    hparams = upstream_hparams(model.config)
    other.load_pretrained(state, hparams)
    assert torch.equal(
        other.denoiser.x_embedder.weight, model.denoiser.x_embedder.weight
    )

    with pytest.raises(InvalidWeights, match="denoiser.d_model"):
        other.load_pretrained(state, {**hparams, "denoiser": {"d_model": 384}})
    with pytest.raises(InvalidWeights, match="conditioning.spacegroup"):
        other.load_pretrained(state, {**hparams, "conditioning": {"spacegroup": True}})
    with pytest.raises(InvalidWeights, match="no interpolant"):
        other.load_pretrained(
            state, {k: v for k, v in hparams.items() if k != "interpolant"}
        )

    missing = {
        k: v for k, v in state.items() if not k.startswith("denoiser.final_layer")
    }
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(missing)
    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state, "denoiser.other": torch.zeros(1)})


def test_a_model_follows_its_configuration():
    config = Config.load()
    small = replace(
        config,
        autoencoder=replace(
            config.autoencoder, d_model=32, dim_feedforward=64, num_layers=1, nhead=2
        ),
        denoiser=replace(config.denoiser, d_model=48, num_layers=1, nhead=2),
        sampling=replace(config.sampling, num_timesteps=4),
    )
    model = ADiT(small).eval()

    assert model.config is small and len(model.times()) == 4
    assert model.denoiser.final_layer.linear.in_features == 48
    assert model.sample(torch.tensor([2, 3])).num_atoms.tolist() == [2, 3]
    assert ADiT.transform(small) == ADiT.transform()


# with the pretrained weights, against ADiT


@pytest.fixture(scope="module")
def pretrained() -> ADiT:
    return load_pretrained(ADiT, "adit", NAME)


@pytest.fixture(scope="module")
def reference():
    return load_reference("adit_reference.npz")


def reference_batch(reference) -> Structures:
    """The crystals of every reference and the molecules of the one of ADiT."""
    num_crystals = int(reference["num_crystals"])
    sizes = reference["num_atoms"][num_crystals:].tolist()
    molecules = [
        Structures.molecule(positions.float(), z)
        for positions, z in zip(
            reference["molecules_pos"].split(sizes),
            reference["molecules_z"].split(sizes),
        )
    ]

    return Structures.collate([reference_structures(), *molecules])


def test_encoder_matches_reference(pretrained, reference):
    structures = reference_batch(reference)
    assert torch.equal(structures.num_atoms, reference["num_atoms"])

    posterior = pretrained.encode(structures)

    assert_close(posterior.mean, reference["enc_mean"], name="mean")
    assert_close(posterior.logvar, reference["enc_logvar"], name="logvar")


def assert_decoded(decoded: Decoded, reference, prefix: str) -> None:
    for field in ("atom_types", "lattices", "frac_coords", "pos"):
        assert_close(getattr(decoded, field), reference[f"{prefix}{field}"], name=field)


def test_decoder_matches_reference(pretrained, reference):
    # decoded from the mean of the posterior of ADiT
    decoded = pretrained.decode_outputs(
        reference["enc_mean"].float(), reference["num_atoms"]
    )

    assert_decoded(decoded, reference, "dec_")


def check_step(pretrained, reference, prefix: str, step: int) -> None:
    """A step of the flow from the latent vectors ADiT reached, its last one at the time 1."""
    num_atoms = reference[f"{prefix}_num_atoms"]
    crystals = torch.full((len(num_atoms),), prefix == "crystals")
    times = pretrained.times()
    x = reference[f"{prefix}_step{step}_x"].float()
    previous = reference.get(f"{prefix}_step{step}_previous")
    previous = None if previous is None else previous.float()

    if step < len(times) - 1:
        moved, prediction = pretrained.sample_step(
            x, num_atoms, crystals, times[step], times[step + 1], previous
        )
        assert_close(moved, reference[f"{prefix}_step{step}_next"], name="next")
    else:
        prediction = pretrained.denoise(x, num_atoms, crystals, times[step], previous)
    assert_close(
        prediction, reference[f"{prefix}_step{step}_prediction"], name="prediction"
    )


@pytest.mark.parametrize("prefix", ["crystals", "molecules"])
def test_one_step_matches_reference(pretrained, reference, prefix):
    check_step(pretrained, reference, prefix, ONE_STEP)


@pytest.mark.full
@pytest.mark.parametrize("prefix", ["crystals", "molecules"])
def test_every_recorded_step_matches_reference(pretrained, reference, prefix):
    for step in reference[f"{prefix}_steps"].tolist():
        check_step(pretrained, reference, prefix, step)


@pytest.mark.parametrize("prefix", ["crystals", "molecules"])
def test_initial_noise_matches_reference(pretrained, reference, prefix):
    num_atoms = reference[f"{prefix}_num_atoms"]
    noise = pretrained.center_noise(reference[f"{prefix}_raw_noise"].float(), num_atoms)

    assert_close(noise, reference[f"{prefix}_noise"], name="noise")
    assert torch.equal(reference[f"{prefix}_noise"], reference[f"{prefix}_step0_x"])


@pytest.mark.full
@pytest.mark.timeout(600)
@pytest.mark.parametrize("prefix", ["crystals", "molecules"])
def test_whole_sampling_matches_reference(pretrained, reference, prefix):
    # the 99 steps of the flow and the prediction at the time 1, chained from
    # the noise of ADiT
    num_atoms = reference[f"{prefix}_num_atoms"]
    crystals = torch.full((len(num_atoms),), prefix == "crystals")

    latent = pretrained.sample_latents(
        reference[f"{prefix}_noise"].float(), num_atoms, crystals
    )

    assert_close(latent, reference[f"{prefix}_step99_prediction"], name="latent")


@pytest.mark.parametrize("prefix", ["crystals", "molecules"])
def test_decoded_samples_match_reference(pretrained, reference, prefix):
    num_atoms = reference[f"{prefix}_num_atoms"]
    latent = reference[f"{prefix}_step99_prediction"].float()

    decoded = pretrained.decode_outputs(latent, num_atoms)
    assert_decoded(decoded, reference, f"{prefix}_dec_")

    # read as ADiT reads them: lengths times the cubic root of the number of
    # atoms, angles in degrees, positions in Å
    crystals = torch.full((len(num_atoms),), prefix == "crystals")
    structures = pretrained.to_structures(decoded, num_atoms, crystals)
    lattices = reference[f"{prefix}_dec_lattices"]
    if prefix == "crystals":
        lengths = lattices[:, :3] * num_atoms.double().unsqueeze(1) ** (1 / 3)
        expected = lattice_from_parameters(lengths, torch.rad2deg(lattices[:, 3:]))
        assert_close(structures.lattice, expected, name="lattice")
    else:
        assert_close(
            structures.positions(),
            10.0 * reference[f"{prefix}_dec_pos"],
            name="positions",
        )
    assert torch.equal(
        structures.z, reference[f"{prefix}_dec_atom_types"].argmax(dim=-1).clamp(min=1)
    )
