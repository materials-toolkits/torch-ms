"""Structures which are not periodic, alone and batched with crystals."""

import dataclasses

import h5py
import pytest
import torch

from torchms.data.augmentation import (
    AtomPermutation,
    AtomTranslation,
    Compose,
    LatticeBasisChange,
    LatticeRotation,
    Supercell,
)
from torchms.data.augmentation.actions import (
    change_lattice_basis,
    make_supercell,
    translate_atoms,
)
from torchms.data.batching import InvalidStructure, Structures
from torchms.data.hdf5 import HDF5Dataset
from torchms.data.memory import InMemoryDataset
from torchms.graph.atomic_radius import atomic_radius
from torchms.graph.cutoff import cutoff
from torchms.graph.transforms import AddQuadruplets, AddRadiusGraph, AddTriplets


def random_molecule(generator: torch.Generator, n_atoms: int, size: float):
    """Atoms drawn in a box centered on the origin, far from [0, 1)."""
    positions = size * (torch.rand(n_atoms, 3, generator=generator) - 0.5)
    z = torch.randint(1, 10, (n_atoms,), generator=generator)

    return Structures.molecule(positions, z)


def triclinic(generator: torch.Generator, n_atoms: int, size: float) -> Structures:
    lattice = size * torch.eye(3) + (torch.rand(3, 3, generator=generator) - 0.5) * size
    x = torch.rand(n_atoms, 3, generator=generator)

    return Structures(lattice=lattice[None], x=x, z=torch.ones(n_atoms).long())


def mixed_batch(seed: int = 0) -> Structures:
    generator = torch.Generator().manual_seed(seed)

    return Structures.collate(
        [
            Structures.diamond(),
            random_molecule(generator, 9, 6.0),
            triclinic(generator, 5, 4.0),
            Structures.water(),
            random_molecule(generator, 1, 1.0),
            Structures.zincblende(),
        ]
    )


def graph_edges(graph) -> list[tuple[int, ...]]:
    rows = torch.cat([graph.edge_index.t(), graph.edge_shift], dim=1)
    return sorted(map(tuple, rows.tolist()))


def ase_edges(structures: Structures, radius: float) -> list[tuple[int, ...]]:
    """Reference neighbour list of a single structure, as sorted (i, j, shift) rows."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(
        numbers=structures.z.numpy(),
        positions=structures.positions().double().numpy(),
        cell=structures.lattice[0].double().numpy(),
        pbc=structures.pbc[0].numpy(),
    )
    i, j, shift = neighbor_list("ijS", atoms, radius)
    rows = torch.cat(
        [torch.tensor(i)[:, None], torch.tensor(j)[:, None], torch.tensor(shift)], 1
    )
    return sorted(map(tuple, rows.tolist()))


class TestStructures:
    def test_periodic_by_default(self):
        assert Structures.diamond().pbc.tolist() == [[True, True, True]]
        assert Structures.diamond().properties == {}

    def test_molecule(self):
        positions = torch.tensor([[10.0, -3.0, 0.5], [11.0, -3.0, 0.5]])
        molecule = Structures.molecule(positions, [6, 8])

        assert molecule.pbc.tolist() == [[False, False, False]]
        assert torch.equal(molecule.lattice, torch.eye(3)[None])
        assert torch.equal(molecule.positions(), positions)

    def test_positions_of_a_crystal(self):
        structures = triclinic(torch.Generator().manual_seed(0), 6, 5.0)

        assert torch.allclose(
            structures.positions(), structures.x @ structures.lattice[0], atol=1e-6
        )

    def test_invalid_pbc(self):
        with pytest.raises(InvalidStructure, match='"pbc"'):
            Structures(
                lattice=torch.eye(3)[None],
                x=torch.zeros(1, 3),
                z=[1],
                pbc=torch.tensor([True, True, True]),
            )

    def test_invalid_property(self):
        with pytest.raises(InvalidStructure, match='property "homo"'):
            Structures(
                lattice=torch.eye(3)[None],
                x=torch.zeros(1, 3),
                z=[1],
                properties={"homo": torch.zeros(2)},
            )

    def test_collate_select_split_separate(self):
        batch = mixed_batch()
        periodic = [True, False, True, False, False, True]

        assert batch.pbc.all(dim=1).tolist() == periodic
        assert (batch.pbc.any(dim=1) == batch.pbc.all(dim=1)).all()

        selected = batch.select(torch.tensor([3, 0, 1]))
        assert selected.pbc.all(dim=1).tolist() == [False, True, False]
        assert torch.equal(selected.z[:3], Structures.water().z)

        parts = batch.split(4)
        assert [part.pbc.all(dim=1).tolist() for part in parts] == [
            periodic[:4],
            periodic[4:],
        ]

        separated = Structures.separate(batch)
        assert [bool(s.pbc.all()) for s in separated] == periodic
        assert torch.equal(Structures.collate(separated).pbc, batch.pbc)

    def test_properties_are_batched(self):
        water = dataclasses.replace(
            Structures.water(), properties={"homo": torch.tensor([-7.0])}
        )
        diamond = dataclasses.replace(
            Structures.diamond(), properties={"density": torch.tensor([3.5])}
        )
        batch = Structures.collate([water, diamond, Structures.zincblende(), water])

        # every property of an element, NaN for the structures without it
        assert list(batch.properties) == ["homo", "density"]
        assert batch.properties["homo"].isnan().tolist() == [False, True, True, False]
        assert batch.properties["density"].isnan().tolist() == [True, False, True, True]
        assert batch.properties["homo"][3] == -7.0

        selected = batch.select(torch.tensor([1, 0]))
        assert selected.properties["density"][0] == 3.5
        assert selected.properties["homo"][1] == -7.0

        last = batch.split(3)[-1]
        assert last.properties["homo"].tolist() == [-7.0]

        separated = Structures.separate(batch)
        assert separated[1].properties["density"].tolist() == [3.5]

        moved = batch.to("meta")
        assert moved.properties["homo"].device == torch.device("meta")
        assert batch.properties["homo"].device == torch.device("cpu")


class TestGraphs:
    @pytest.mark.parametrize("radius", [0.5, 1.2, 3.0, 20.0])
    def test_cutoff_of_molecules_is_the_brute_force_search(self, radius):
        generator = torch.Generator().manual_seed(int(10 * radius))
        molecules = [
            random_molecule(generator, n_atoms, size)
            for n_atoms, size in [(1, 1.0), (2, 1.0), (14, 5.0), (40, 8.0)]
        ]
        batch = Structures.collate(molecules + [Structures.water()])

        graph = cutoff(batch, radius, self_images=True)

        positions = batch.positions()
        dists = torch.cdist(positions, positions)
        same = batch.batch[:, None] == batch.batch[None, :]
        expected = (same & (dists <= radius)).fill_diagonal_(False).nonzero().t()

        assert torch.equal(graph.edge_index, expected)
        assert (graph.edge_shift == 0).all()
        assert torch.allclose(
            graph.edges_distances(), dists[expected[0], expected[1]], atol=1e-5
        )

    @pytest.mark.parametrize("radius", [1.5, 3.0, 5.0])
    def test_cutoff_of_a_molecule_matches_ase(self, radius):
        molecule = random_molecule(torch.Generator().manual_seed(1), 25, 6.0)

        assert graph_edges(cutoff(molecule, radius)) == ase_edges(molecule, radius)

    @pytest.mark.parametrize("pbc", [(True, True, False), (False, True, False)])
    @pytest.mark.parametrize("radius", [2.0, 4.5])
    def test_cutoff_of_a_slab_matches_ase(self, pbc, radius):
        """Periodic along some of the lattice vectors only."""
        generator = torch.Generator().manual_seed(2)
        structures = triclinic(generator, 9, 4.0)
        # inside the cell along the periodic vectors, anywhere along the others
        periodic = torch.tensor([pbc])
        x = torch.where(periodic, structures.x, 3.0 * structures.x - 1.0)
        slab = dataclasses.replace(structures, x=x, pbc=periodic)

        graph = cutoff(slab, radius, self_images=True)

        assert graph_edges(graph) == ase_edges(slab, radius)
        assert (graph.edge_shift[:, ~periodic[0]] == 0).all()

    @pytest.mark.parametrize("radius", [1.6, 3.0, 5.0])
    @pytest.mark.parametrize("self_images", [False, True])
    def test_cutoff_of_a_mixed_batch(self, radius, self_images):
        """Each structure has the graph it has alone."""
        batch = mixed_batch()

        graph = cutoff(batch, radius, self_images=self_images)
        alone = [
            cutoff(structures, radius, self_images=self_images)
            for structures in Structures.separate(batch)
        ]
        expected = type(graph).collate(alone, batch)

        assert torch.equal(graph.num_edges, expected.num_edges)
        assert torch.equal(graph.edge_index, expected.edge_index)
        assert torch.equal(graph.edge_shift, expected.edge_shift)
        assert (graph.edge_shift[~batch.pbc.all(dim=1)[graph.edge_batch]] == 0).all()
        # the crystals keep their periodic images
        assert (graph.edge_shift[graph.edge_batch == 0] != 0).any()

    def test_atomic_radius_of_water(self):
        graph = atomic_radius(Structures.water(), scaling=1.1, ensure_gap=0.0)

        assert graph.edge_index.tolist() == [[0, 0, 1, 2], [1, 2, 0, 0]]
        assert (graph.edge_shift == 0).all()

    def test_atomic_radius_of_a_mixed_batch(self):
        batch = Structures.collate(
            [Structures.zincblende(), Structures.water(), Structures.diamond()]
        )

        graph = atomic_radius(batch, scaling=1.1, ensure_gap=0.0)
        alone = [
            atomic_radius(structures, scaling=1.1, ensure_gap=0.0)
            for structures in Structures.separate(batch)
        ]

        assert graph.num_edges.tolist() == [int(g.num_edges) for g in alone]
        assert graph.num_edges[1] == 4

    def test_tuples_of_edges_of_a_mixed_batch(self):
        transform = Compose(
            [
                AddRadiusGraph(1.6, geometry=True),
                AddTriplets(geometry=True),
                AddQuadruplets(geometry=True),
            ]
        )
        batch = transform(
            Structures.collate([Structures.diamond(), Structures.water()])
        )

        # diamond: 4 bonds of 1.54 Å per atom. Water: its two bonds and its
        # two hydrogens, 1.51 Å apart, which make a triangle.
        assert batch.graph.num_edges.tolist() == [32, 6]
        angle = batch.triplets.angle[batch.triplets.batch == 1].rad2deg()
        assert torch.allclose(angle.max(), torch.tensor(104.52), atol=0.05)
        assert torch.allclose(angle.sum(), torch.tensor(360.0), atol=0.05)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cutoff_on_gpu_is_the_same(self):
        batch = mixed_batch()

        graph = cutoff(batch, 3.0)
        on_gpu = cutoff(batch.to("cuda"), 3.0)

        assert torch.equal(on_gpu.edge_index.cpu(), graph.edge_index)
        assert torch.equal(on_gpu.edge_shift.cpu(), graph.edge_shift)


def distances(structures: Structures) -> list[torch.Tensor]:
    """Sorted distances between the neighbours of each structure."""
    graph = cutoff(structures, 4.0, self_images=True)
    dists = graph.edges_distances()

    return [
        dists[graph.edge_batch == i].sort().values
        for i in range(len(structures.num_atoms))
    ]


class TestAugmentations:
    def test_translation_does_not_wrap_a_molecule(self):
        batch = mixed_batch()
        molecule = ~batch.pbc.all(dim=1)[batch.batch]
        translation = torch.full((len(batch.num_atoms), 3), 0.75)

        translated = translate_atoms(batch, translation)

        assert torch.allclose(translated.x[molecule], batch.x[molecule] + 0.75)
        assert ((translated.x[~molecule] >= 0) & (translated.x[~molecule] < 1)).all()

    def test_basis_change_leaves_a_molecule(self):
        batch = mixed_batch()
        molecule = ~batch.pbc.all(dim=1)
        basis = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 1]]).expand(6, 3, 3)

        sheared = change_lattice_basis(batch, basis)

        assert torch.equal(sheared.lattice[molecule], batch.lattice[molecule])
        assert torch.equal(
            sheared.x[molecule[batch.batch]], batch.x[molecule[batch.batch]]
        )
        assert not torch.equal(sheared.lattice[~molecule], batch.lattice[~molecule])

    def test_supercell_of_a_molecule(self):
        with pytest.raises(ValueError, match="not periodic"):
            make_supercell(Structures.water(), torch.tensor([[2, 1, 1]]))

        same = make_supercell(Structures.water(), torch.tensor([[1, 1, 1]]))
        assert torch.equal(same.x, Structures.water().x)

    def test_supercell_transform_repeats_the_crystals_only(self):
        batch = mixed_batch()
        molecule = ~batch.pbc.all(dim=1)

        for seed in range(5):
            generator = torch.Generator().manual_seed(seed)
            supercell = Supercell(max_repeat=3, generator=generator)(batch)

            assert torch.equal(supercell.num_atoms[molecule], batch.num_atoms[molecule])
            assert torch.equal(supercell.pbc, batch.pbc)

    @pytest.mark.parametrize(
        "transform",
        [
            AtomPermutation,
            LatticeRotation,
            AtomTranslation,
            LatticeBasisChange,
            lambda generator: Compose(
                [
                    AtomPermutation(generator=generator),
                    LatticeBasisChange(generator=generator),
                    AtomTranslation(generator=generator),
                    LatticeRotation(generator=generator),
                ]
            ),
        ],
    )
    def test_distances_are_invariant(self, transform):
        batch = mixed_batch()

        augmented = transform(generator=torch.Generator().manual_seed(3))(batch)
        augmented.check_tensors()

        assert torch.equal(augmented.pbc, batch.pbc)
        for before, after in zip(distances(batch), distances(augmented)):
            assert before.shape == after.shape
            assert torch.allclose(before, after, atol=1e-4)

    def test_rotation_rotates_a_molecule(self):
        water = Structures.water()

        rotated = LatticeRotation(generator=torch.Generator().manual_seed(0))(water)

        assert not torch.allclose(rotated.positions(), water.positions(), atol=1e-3)
        assert torch.allclose(
            torch.cdist(rotated.positions(), rotated.positions()),
            torch.cdist(water.positions(), water.positions()),
            atol=1e-5,
        )


class TestFiles:
    @staticmethod
    def labelled() -> Structures:
        batch = mixed_batch()
        n_struct = len(batch.num_atoms)
        homo = torch.arange(n_struct, dtype=torch.float32)
        homo[2] = torch.nan

        return dataclasses.replace(batch, properties={"homo": homo, "qed": homo / 10})

    @staticmethod
    def assert_same(a: Structures, b: Structures):
        assert torch.equal(a.num_atoms, b.num_atoms)
        assert torch.equal(a.pbc, b.pbc)
        assert torch.equal(a.x, b.x) and torch.equal(a.z, b.z)
        assert torch.equal(a.lattice, b.lattice)
        assert list(a.properties) == list(b.properties)
        for name, values in a.properties.items():
            assert torch.equal(
                values.nan_to_num(-1.0), b.properties[name].nan_to_num(-1.0)
            )

    def test_round_trip(self, tmp_path):
        batch = self.labelled()
        file = tmp_path / "mixed.hdf5"
        HDF5Dataset.from_structures(file, batch)
        dataset = HDF5Dataset(file)

        assert dataset.has_pbc() and dataset.property_names() == ["homo", "qed"]
        self.assert_same(dataset[:], batch)

        index = [4, 1, 5, 1]
        self.assert_same(dataset[index], batch.select(torch.tensor(index)))

        self.assert_same(InMemoryDataset.from_hdf5(file).structures, batch)
        self.assert_same(InMemoryDataset.from_hdf5(file)[index], dataset[index])

    def test_several_batches(self, tmp_path):
        """Properties of the first batch, NaN for the batches without them."""
        batch = self.labelled()
        file = tmp_path / "mixed.hdf5"
        HDF5Dataset.from_structures(
            file, [batch, batch.without_graphs(), mixed_batch()]
        )

        loaded = HDF5Dataset(file)[:]

        assert len(loaded.num_atoms) == 18
        assert torch.equal(loaded.pbc, batch.pbc.repeat(3, 1))
        assert loaded.properties["qed"][12:].isnan().all()
        assert torch.equal(
            loaded.properties["qed"][6:12].nan_to_num(-1.0),
            batch.properties["qed"].nan_to_num(-1.0),
        )

    def test_unknown_property(self, tmp_path):
        with pytest.raises(ValueError, match="homo"):
            HDF5Dataset.from_structures(
                tmp_path / "mixed.hdf5", [mixed_batch(), self.labelled()]
            )

    def test_file_without_pbc_is_periodic(self, tmp_path):
        """Files written before molecules: crystals only."""
        batch = Structures.collate([Structures.diamond(), Structures.zincblende()])
        file = tmp_path / "crystals.hdf5"
        HDF5Dataset.from_structures(file, batch)

        with h5py.File(file, "a") as data_file:
            del data_file["pbc"]

        dataset = HDF5Dataset(file)
        assert not dataset.has_pbc() and dataset.property_names() == []
        assert dataset[:].pbc.all()
        assert dataset[[1]].pbc.shape == (1, 3)
        assert InMemoryDataset.from_hdf5(file).structures.pbc.all()
