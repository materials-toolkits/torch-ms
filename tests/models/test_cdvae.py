"""Tests for the rewrite of CDVAE."""

from dataclasses import replace

import pytest
import torch

from helpers.compare import assert_close, assert_close_in_the_cell
from helpers.graphs import neighbours_at_the_same_distance as ties
from helpers.references import load_reference, reference_structures
from helpers.structures import generator, make_batch, make_structures, noisy
from helpers.weights import pretrained as load_pretrained
from torchms.data import Structures
from torchms.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    Compose,
    LatticeBasisChange,
    LatticeRotation,
)
from torchms.models.estimator import weights
from torchms.models.estimator.reader import Placeholder, read_pickle
from torchms.graph.geometry import lattice_from_parameters
from torchms.graph.graph import Graph
from torchms.graph.line_graph import reverse_edges
from torchms.graph.transforms import (
    AddRadiusGraph,
    KeepNearImages,
)
from torchms.models.cdvae import CDVAE, Config, InvalidWeights, sample_composition

NAME = "cdvae-mp20-cowboys"

# thousands of triplets per structure: slower than the default timeout
pytestmark = pytest.mark.timeout(60)

# the dynamics of the reference: its structures and its smallest noise levels
NUM_DYNAMICS = 6
NUM_LEVELS = 5

# a graph of bonds for the encoder, in place of the one of CrystalNN
BONDS = Compose([AddRadiusGraph(3.5, self_images=True), CDVAE.transform()])


def latent(n_struct: int, seed: int = 0) -> torch.Tensor:
    config = Config.load()
    return torch.randn(n_struct, config.model.latent_dim, generator=generator(seed))


@pytest.fixture(scope="module")
def model() -> CDVAE:
    """Model with random weights, and cells of a few Å near a cube."""
    torch.manual_seed(0)
    model = CDVAE().eval().requires_grad_(False)
    model.lattice_means.copy_(torch.tensor([2.5, 2.5, 2.5, 90.0, 90.0, 90.0]))
    model.lattice_stds.copy_(torch.tensor([0.2, 0.2, 0.2, 5.0, 5.0, 5.0]))

    return model


def test_pinned_checkpoints(tmp_path):
    assert set(weights.CHECKPOINTS["cdvae"]) == {NAME}
    for checkpoint in weights.CHECKPOINTS["cdvae"][NAME]:
        assert checkpoint.url.startswith("https://")
        # a commit of the repository, not a branch which can change
        assert "/resolve/283b73cdf207d87b42110ca0277da3fa2f3c73d8/" in checkpoint.url
        assert len(checkpoint.sha256) == 64

    paths = weights.checkpoint_paths("cdvae", NAME, tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        f"cdvae/{NAME}/epoch=745-step=79076.ckpt",
        f"cdvae/{NAME}/lattice_scaler.pt",
        f"cdvae/{NAME}/hparams.yaml",
    ]

    with pytest.raises(KeyError, match="unknown model"):
        weights.checkpoint_paths("cdvae", "cdvae-9")


def test_noise_levels(model):
    config = Config.load()
    assert model.sigmas.shape == (config.model.num_noise_levels,)
    assert torch.allclose(
        model.sigmas[[0, -1]],
        torch.tensor([config.model.sigma_begin, config.model.sigma_end]),
    )
    assert (model.sigmas[1:] < model.sigmas[:-1]).all()


# the encoder with random weights


def test_encoder_output(model):
    config = Config.load()
    mu, log_var = model(BONDS(make_batch()))

    assert mu.shape == log_var.shape == (2, config.model.latent_dim)
    assert mu.dtype == torch.float32
    assert mu.isfinite().all() and log_var.isfinite().all()
    assert not torch.allclose(mu[0], mu[1])


def test_encoder_batch_equals_single(model):
    alone = torch.cat([model(BONDS(structure))[0] for structure in make_structures()])

    assert torch.allclose(model(BONDS(make_batch()))[0], alone, atol=1e-5)


@pytest.mark.parametrize(
    "transform",
    [AtomPermutation, LatticeRotation, AtomTranslation, LatticeBasisChange],
)
def test_encoder_invariances(model, transform):
    batch = make_batch()
    augmented = transform(generator=generator())(batch)

    assert torch.allclose(model(BONDS(augmented))[0], model(BONDS(batch))[0], atol=2e-4)


def test_encoder_pairs_do_not_come_back(model):
    structures = BONDS(make_batch())
    graph, pairs = structures.graph, structures.line_graph.index

    assert pairs.shape[1] > 0
    assert (graph.receivers[pairs[0]] == graph.senders[pairs[1]]).all()
    assert (graph.senders[pairs[0]] != graph.receivers[pairs[1]]).all()


def test_encoder_needs_its_graphs(model):
    batch = make_batch()

    with pytest.raises(ValueError, match="needs a graph"):
        model.transform()(batch)

    with pytest.raises(ValueError, match="line_graph"):
        model(AddRadiusGraph(3.5)(batch))


def test_encoder_unknown_element(model):
    # americium has no row in the embedding of the encoder
    structures = Structures(
        lattice=torch.eye(3).unsqueeze(0) * 3.0, x=torch.zeros(1, 3), z=[95]
    )

    with pytest.raises(ValueError, match="not supported"):
        model(BONDS(structures))


# the quantities decoded from a latent vector


def test_decoded_quantities(model):
    config = Config.load()
    z = latent(3)
    num_atoms = torch.tensor([1, 8, 20])

    assert model.predict_num_atoms(z).shape == (3, config.model.max_atoms + 1)
    assert model.predict_composition(z).shape == (3, config.model.num_elements)

    lengths, angles = model.predict_lattice(z, num_atoms)
    assert lengths.shape == angles.shape == (3, 3)

    # the lengths grow as the cube root of the number of atoms
    one, _ = model.predict_lattice(z, torch.ones(3, dtype=torch.int64))
    assert torch.allclose(lengths, one * num_atoms.unsqueeze(1) ** (1 / 3), rtol=1e-5)


def test_sample_composition():
    probability = torch.tensor(
        [
            [0.5, 0.0, 0.5, 0.0],  # rounds to 2 and 2 atoms of 3: one too many
            [0.25, 0.75, 0.0, 0.0],
            [1 / 3, 1 / 3, 1 / 3, 0.0],  # rounds to 3 atoms of 4: one missing
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    num_atoms = torch.tensor([3, 4, 4, 2])

    for seed in range(20):
        z = sample_composition(probability, num_atoms, generator(seed))
        assert z.shape == (13,)

        first, second, third, last = z.split(num_atoms.tolist())
        assert sorted(first.tolist()) in ([1, 1, 3], [1, 3, 3])
        assert sorted(second.tolist()) == [1, 2, 2, 2]
        assert sorted(third.tolist())[:3] != [4, 4, 4] and 4 not in third
        assert sorted(set(third.tolist())) == [1, 2, 3]
        assert last.tolist() == [4, 4]

    # the atoms of a structure are in a random order
    orders = {
        tuple(sample_composition(probability, num_atoms, generator(seed))[3:7].tolist())
        for seed in range(20)
    }
    assert len(orders) > 1


def test_sample_composition_without_atoms():
    probability = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    z = sample_composition(probability, torch.tensor([0, 2]), generator())
    assert z.tolist() == [2, 2]


# the decoder with random weights


def decoder_graph(structures: Structures) -> Structures:
    return CDVAE.decoder_transform()(structures)


def test_decoder_output(model):
    config = Config.load()
    structures = decoder_graph(noisy(make_batch()))
    move, scores = model.denoise(latent(2), structures)

    assert move.shape == (len(structures.z), 3)
    assert scores.shape == (len(structures.z), config.model.num_elements)
    assert move.isfinite().all() and scores.isfinite().all()
    assert move.abs().max() > 0


def test_decoder_graph():
    config = Config.load()
    structures = decoder_graph(noisy(make_batch()))
    graph = structures.graph

    assert (graph.edges_distances() <= config.graph.decoder.radius).all()
    assert graph.edge_shift.abs().max() <= config.graph.decoder.max_image_shift
    assert (reverse_edges(graph) >= 0).all()

    # every edge is in one bond, and a triplet has two edges of the same atom
    bonds, triplets = structures.bonds.index, structures.triplets.index
    assert sorted(bonds.flatten().tolist()) == list(range(int(graph.num_edges.sum())))
    assert (graph.senders[triplets[0]] == graph.senders[triplets[1]]).all()
    assert (triplets[0] != triplets[1]).all()


def test_decoder_depends_on_the_latent(model):
    structures = decoder_graph(noisy(make_batch()))
    move, _ = model.denoise(latent(2, seed=0), structures)
    other, _ = model.denoise(latent(2, seed=1), structures)

    assert not torch.allclose(move, other)


def test_decoder_batch_equals_single(model):
    z = latent(2)
    moves, scores = zip(
        *[
            model.denoise(z[i : i + 1], decoder_graph(noisy(structure, seed=i)))
            for i, structure in enumerate(make_structures())
        ]
    )
    batch = Structures.collate(
        [noisy(structure, seed=i) for i, structure in enumerate(make_structures())]
    )
    move, score = model.denoise(z, decoder_graph(batch))

    assert torch.allclose(move, torch.cat(moves), atol=1e-4)
    assert torch.allclose(score, torch.cat(scores), atol=1e-4)


def test_decoder_follows_a_rotation(model):
    batch = noisy(make_batch())
    rotated = LatticeRotation(generator=generator())(batch)
    z = latent(2)

    move, scores = model.denoise(z, decoder_graph(batch))
    rotated_move, rotated_scores = model.denoise(z, decoder_graph(rotated))

    # the same graph: the scores of the elements do not change, the moves are
    # rotated with the cell
    assert torch.allclose(rotated_scores, scores, atol=1e-3)
    rotation = torch.linalg.solve(batch.lattice, rotated.lattice)[batch.batch]
    expected = (move.unsqueeze(1) @ rotation).squeeze(1)
    assert torch.allclose(rotated_move, expected, atol=1e-3)


def test_decoder_needs_its_graphs(model):
    config = Config.load()
    batch = noisy(make_batch())

    with pytest.raises(ValueError, match="decoder_transform"):
        model.denoise(latent(2), batch)

    with pytest.raises(ValueError, match="triplets, bonds"):
        model.denoise(latent(2), AddRadiusGraph(config.graph.decoder.radius)(batch))


# generation with random weights


def test_langevin_step(model):
    batch = noisy(make_batch())
    z = latent(2)
    noise = torch.zeros_like(batch.x)

    moved = model.langevin_step(z, batch, model.sigmas[-1], noise)
    assert moved.graph is None
    assert torch.equal(moved.lattice, batch.lattice)
    assert (moved.x >= 0).all() and (moved.x <= 1).all()

    # without noise, the atoms follow the move of the decoder
    move, scores = model.denoise(z, decoder_graph(batch))
    step = 1e-4
    expected = batch.positions() + step * move / model.sigmas[-1]
    difference = moved.positions() - expected
    cells = torch.linalg.solve(
        batch.lattice[batch.batch].transpose(1, 2), difference.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(cells, cells.round(), atol=1e-4)
    assert torch.equal(moved.z, scores.argmax(dim=1) + 1)

    kept = model.langevin_step(z, batch, model.sigmas[-1], noise, update_elements=False)
    assert torch.equal(kept.z, batch.z)
    assert torch.allclose(kept.x, moved.x)


def test_generate(model):
    config = Config.load()
    z = latent(3)
    options = {"steps_per_level": 1, "min_sigma": 5.0}

    structures = model.generate(z, generator=generator(), **options)
    assert structures.lattice.shape == (3, 3, 3)
    assert len(structures.z) == int(structures.num_atoms.sum())
    assert (structures.num_atoms <= config.model.max_atoms).all()
    assert torch.equal(structures.num_atoms, model.predict_num_atoms(z).argmax(dim=-1))
    assert (structures.x >= 0).all() and (structures.x <= 1).all()
    assert ((structures.z >= 1) & (structures.z <= config.model.num_elements)).all()

    # the cell is the one decoded from the latent vector
    lengths, _ = model.predict_lattice(z, structures.num_atoms)
    assert torch.allclose(structures.lattice.norm(dim=-1), lengths, rtol=1e-4)

    # the same draws give the same structures
    again = model.generate(z, generator=generator(), **options)
    assert torch.equal(again.x, structures.x) and torch.equal(again.z, structures.z)
    other = model.generate(z, generator=generator(1), **options)
    assert not torch.equal(other.x, structures.x)


def test_generate_with_given_atoms(model):
    num_atoms = torch.tensor([2, 5])
    z = torch.tensor([11, 17, 8, 8, 8, 22, 38])

    structures = model.generate(
        latent(2),
        num_atoms=num_atoms,
        z=z,
        steps_per_level=1,
        min_sigma=5.0,
        generator=generator(),
    )

    assert torch.equal(structures.num_atoms, num_atoms)
    assert torch.equal(structures.z, z)


def test_sample(model):
    structures = model.sample(
        4, steps_per_level=1, min_sigma=5.0, generator=generator()
    )

    assert structures.lattice.shape == (4, 3, 3)
    assert structures.x.isfinite().all()


# the weights


def upstream_state(model: CDVAE) -> dict[str, torch.Tensor]:
    """Parameters of a model under the names of cdvae."""
    config = Config.load()
    state = {
        name.replace(".rbf.centers", ".rbf.offset"): values
        for name, values in model.state_dict().items()
        if name not in config.upstream.constant_buffers
        and not name.startswith("lattice_")
    }
    state["type_sigmas"] = torch.ones(config.model.num_noise_levels)

    return state


def upstream_hparams() -> dict[str, object]:
    """Configuration of the training of cdvae the rewrite is written for."""
    config = Config.load()
    return {
        "data": dict(config.upstream.sections()["data"]),
        "model": {
            **config.upstream.sections()["model"],
            "encoder": dict(config.upstream.sections()["model.encoder"]),
            "decoder": dict(config.upstream.sections()["model.decoder"]),
        },
    }


def test_load_pretrained_checks_the_parameters(model):
    state = upstream_state(model)
    means, stds = torch.arange(6.0), torch.arange(1.0, 7.0)

    other = CDVAE()
    other.load_pretrained(state, means, stds, upstream_hparams())
    other.eval()
    with torch.no_grad():
        structures = BONDS(make_batch())
        assert torch.equal(other(structures)[0], model(structures)[0])

        z = latent(2)
        scaled = model.fc_lattice(z) * stds + means
        lengths, angles = other.predict_lattice(z, torch.tensor([1, 1]))
        assert torch.allclose(torch.cat([lengths, angles], dim=1), scaled)

    hparams = upstream_hparams()
    hparams["model"]["encoder"]["cutoff"] = 5.0
    hparams["data"]["max_atoms"] = 24
    with pytest.raises(InvalidWeights, match="another configuration") as error:
        other.load_pretrained(state, means, stds, hparams)
    assert "model.encoder.cutoff" in str(error.value)
    assert "data.max_atoms" in str(error.value)

    with pytest.raises(InvalidWeights, match="no model.decoder"):
        other.load_pretrained(
            state, means, stds, {"data": {}, "model": {"encoder": {}}}
        )

    with pytest.raises(InvalidWeights, match="Unexpected"):
        other.load_pretrained({**state, "unknown.weight": torch.zeros(1)}, means, stds)

    del state["fc_mu.weight"]
    with pytest.raises(InvalidWeights, match="Missing"):
        other.load_pretrained(state, means, stds)


class Scaler:
    """Stands for the scaler of cdvae, an object torch.load refuses to read."""

    def __init__(self):
        self.means = torch.zeros(6)
        self.stds = torch.ones(6)


def test_files_are_read_without_their_classes(tmp_path):
    torch.save(Scaler(), tmp_path / "scaler.pt")
    torch.save({"state_dict": {"a": torch.ones(2)}, "other": Scaler()}, tmp_path / "c")

    scaler = read_pickle(tmp_path / "scaler.pt")
    assert isinstance(scaler, Placeholder) and not isinstance(scaler, Scaler)
    assert set(scaler.state) == {"means", "stds"}

    checkpoint = read_pickle(tmp_path / "c")
    assert torch.equal(checkpoint["state_dict"]["a"], torch.ones(2))
    assert isinstance(checkpoint["other"], Placeholder)


# the pretrained model against the real cdvae


def reference() -> tuple[Structures, dict[str, torch.Tensor]]:
    """Structures of the tests and what cdvae computed for them."""
    structures = reference_structures()
    expected = dict(load_reference("cdvae_reference.npz").arrays)
    assert expected["index"].tolist() == list(range(len(structures.num_atoms)))

    return structures, expected


def with_bonds_of_cdvae(
    structures: Structures, expected: dict[str, torch.Tensor]
) -> Structures:
    """Structures with the graph CrystalNN of pymatgen built for cdvae."""
    n_struct = len(structures.num_atoms)
    graph = Graph.from_edges(
        structures,
        structures.num_atoms,
        torch.stack([expected["sender"].long(), expected["receiver"].long()]),
        expected["shift"].long(),
        edge_batch=torch.arange(n_struct).repeat_interleave(expected["num_bonds"]),
    )

    return replace(structures, graph=graph)


def neighbours_at_the_same_distance(structures: Structures) -> torch.Tensor:
    """Structures where the 20 nearest neighbours of the decoder are not defined.

    cdvae keeps one or the other of the two images of an atom itself on each
    side of it, from the rounding of their distances.
    """
    config = Config.load().graph.decoder
    near = Compose(
        [
            AddRadiusGraph(config.radius, self_images=True),
            KeepNearImages(config.max_image_shift),
        ]
    )

    return ties(structures, near, config.max_neighbors)


@pytest.fixture(scope="module")
def pretrained() -> CDVAE:
    return load_pretrained(CDVAE, "cdvae", NAME)


def test_pretrained_encoder_matches_cdvae(pretrained):
    """Encoder of cdvae f857f59, computed once outside of the project."""
    structures, expected = reference()
    structures = pretrained.transform()(with_bonds_of_cdvae(structures, expected))

    assert structures.line_graph.num_tuples.tolist() == expected["num_pairs"].tolist()

    mu, log_var = pretrained(structures)
    assert_close(mu, expected["mu"], name="mu")
    assert_close(log_var, expected["log_var"], name="log_var")
    assert (mu - expected["mu"]).abs().mean() < 1e-6

    # the spherical Bessel functions of both codes lose digits for short
    # bonds: without bonds under 2 Å, the difference itself is under 1e-5
    graph = structures.graph
    shortest = -graph.aggregate_edges_to_structures(-graph.edges_distances(), "amax")
    error = (mu - expected["mu"]).abs().amax(dim=1)
    assert (shortest >= 2.0).sum() > 20
    assert error[shortest >= 2.0].max() < 1e-5
    assert error.max() < 1e-4


def test_pretrained_decoded_quantities_match_cdvae(pretrained):
    structures, expected = reference()
    mu = expected["mu"]

    assert_close(
        pretrained.predict_num_atoms(mu),
        expected["num_atoms_logits"],
        name="num_atoms_logits",
    )

    lengths, angles = pretrained.predict_lattice(mu, structures.num_atoms)
    assert_close(lengths, expected["decoded_lengths"], name="decoded_lengths")
    assert_close(angles, expected["decoded_angles"], name="decoded_angles")

    # cdvae repeats the composition of a structure for each of its atoms
    composition = pretrained.predict_composition(mu)
    assert_close(
        composition[structures.batch],
        expected["composition_logits"],
        name="composition_logits",
    )


def test_pretrained_decoder_matches_cdvae(pretrained):
    """Decoder of cdvae on atoms moved at random, in the cells of the structures."""
    structures, expected = reference()
    moved = Structures(
        lattice=lattice_from_parameters(expected["lengths"], expected["angles"]),
        x=expected["noisy"],
        z=structures.z,
        num_atoms=structures.num_atoms,
    )
    defined = ~neighbours_at_the_same_distance(moved)
    assert defined.sum() >= 20

    moved = pretrained.decoder_transform()(moved)
    graph = moved.graph

    # the graphs are the ones of cdvae where its nearest neighbours are defined
    assert torch.equal(graph.num_edges[defined], expected["num_edges"][defined])
    assert torch.equal(
        moved.triplets.num_tuples[defined], expected["num_triplets"][defined]
    )
    distance_sum = graph.aggregate_edges_to_structures(graph.edges_distances().double())
    assert torch.allclose(
        distance_sum[defined], expected["distance_sum"][defined], atol=1e-3
    )

    move, scores = pretrained.denoise(expected["mu"], moved)
    atoms = defined[moved.batch]
    assert_close(move[atoms], expected["score"][atoms], name="score")
    assert_close(scores[atoms], expected["types_logits"][atoms], name="types_logits")
    assert (move[atoms] - expected["score"][atoms]).abs().mean() < 1e-5

    # elsewhere the graphs differ by a few edges: the outputs stay near
    assert move.isfinite().all()
    assert (move - expected["score"]).abs().mean() < 0.02


def start_of_the_dynamics(
    pretrained: CDVAE, structures: Structures, expected: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, Structures]:
    """The latent vectors and the random atoms the dynamics of cdvae started from."""
    num_atoms = structures.num_atoms[:NUM_DYNAMICS]
    mu = expected["mu"][:NUM_DYNAMICS]

    return mu, Structures(
        lattice=lattice_from_parameters(*pretrained.predict_lattice(mu, num_atoms)),
        x=expected["dynamics_start"],
        z=structures.z[: int(num_atoms.sum())],
        num_atoms=num_atoms,
    )


def test_pretrained_langevin_step_matches_cdvae(pretrained):
    """The first step of the dynamics of cdvae, from its random atoms and its noise."""
    structures, expected = reference()
    mu, start = start_of_the_dynamics(pretrained, structures, expected)
    sigma = pretrained.sigmas[-NUM_LEVELS]

    after = pretrained.langevin_step(
        mu, start, sigma, expected["dynamics_noise"][0], update_elements=False
    )

    defined = ~neighbours_at_the_same_distance(start)
    assert defined.sum() >= 2
    atoms = defined[after.batch]
    assert_close_in_the_cell(after.x[atoms], expected["dynamics_states"][0][atoms])


@pytest.mark.full
def test_pretrained_dynamics_match_cdvae(pretrained):
    """The ten chained steps of the dynamics of cdvae, from its start and its noise."""
    structures, expected = reference()
    noise = expected["dynamics_noise"]
    levels = pretrained.sigmas[-NUM_LEVELS:]
    assert len(noise) == 2 * len(levels)
    mu, current = start_of_the_dynamics(pretrained, structures, expected)

    defined = torch.ones(NUM_DYNAMICS, dtype=torch.bool)
    for step, sigma in enumerate(levels.repeat_interleave(2)):
        defined &= ~neighbours_at_the_same_distance(current)
        current = pretrained.langevin_step(
            mu, current, sigma, noise[step], update_elements=False
        )
        # each step against the state cdvae reached, where all are defined
        atoms = defined[current.batch]
        assert_close_in_the_cell(
            current.x[atoms],
            expected["dynamics_states"][step][atoms],
            name=f"step {step}",
        )

    assert defined.sum() >= 2
    difference = (current.x - expected["dynamics_end"] + 0.5) % 1.0 - 0.5
    assert difference.abs().max() < 0.01
