"""Filters to select structures, when a dataset is loaded or preprocessed."""

from dataclasses import dataclass, field
from typing import Optional

import torch
from jaxtyping import Bool, Float

from torchms.data.batching import Structures
from torchms.graph.sparse import scatter

# labels of Structures a condition can be put on, with its named properties
LABELS = ("total_energy", "formation_energy_per_atom", "energy_above_hull", "band_gap")


def values(structures: Structures, name: str) -> Float[torch.Tensor, "n_struct"]:
    """Values of a label or of a named property of the structures.

    Raises:
        KeyError: If the structures have no label nor property with this name.

    Examples:
        >>> values(Structures.diamond(), "band_gap")
        tensor([nan])
    """
    if name in LABELS:
        return getattr(structures, name)

    if name in structures.properties:
        return structures.properties[name]

    raise KeyError(
        f'unknown label or property "{name}", must be one of '
        f"{[*LABELS, *structures.properties]}"
    )


@dataclass
class Filter:  # pylint: disable=too-many-instance-attributes
    """Select structures of a batch, returns a mask of the structures to keep.

    A structure is kept if it passes every setting given. It is given to a
    dataset to keep some of its structures (load_dataset, auto_loader,
    HDF5Dataset.filter) and to the preprocessing of CIF files.

    A condition on a label or on a named property is not passed by a structure
    without value for it (NaN).

    Args:
        max_atoms: Maximum number of atoms per structure.
        exclude_elements: Atomic numbers that must not appear in a structure.
        max_energy_above_hull: Maximum energy above the hull in eV/atom
            (0 for stable, 0.1 is commonly used for meta-stable). Structures
            without energy above the hull are dropped.
        min_atoms: Minimum number of atoms per structure.
        elements: Atomic numbers a structure can be made of: a structure with
            another element is dropped.
        periodic: If True, only the structures periodic along their three
            lattice vectors are kept, crystals. If False, only the ones
            periodic along none, molecules.
        minimum: Lowest value of labels or named properties, by name.
        maximum: Highest value of labels or named properties, by name.
        equal: Value of labels or named properties, by name.
        known: Names of the labels or named properties a structure must have
            a value for.

    Examples:
        >>> structures = Structures.collate(
        ...     [Structures.diamond(), Structures.zincblende(), Structures.water()]
        ... )
        >>> Filter(exclude_elements=(16,))(structures)
        tensor([ True, False,  True])
        >>> Filter(max_atoms=4)(structures)
        tensor([False, False,  True])
        >>> Filter(elements=(1, 6, 8), periodic=True)(structures)
        tensor([ True, False, False])

        Conditions on labels and on named properties:

        >>> import dataclasses
        >>> structures = dataclasses.replace(
        ...     structures,
        ...     band_gap=torch.tensor([5.5, 3.7, torch.nan]),
        ...     properties={"disorder": torch.tensor([0.0, 1.0, 0.0])},
        ... )
        >>> Filter(minimum={"band_gap": 4.0})(structures)
        tensor([ True, False, False])
        >>> Filter(equal={"disorder": 0}, known=("band_gap",))(structures)
        tensor([ True, False, False])
    """

    max_atoms: Optional[int] = None
    exclude_elements: tuple[int, ...] = ()
    max_energy_above_hull: Optional[float] = None
    min_atoms: Optional[int] = None
    elements: Optional[tuple[int, ...]] = None
    periodic: Optional[bool] = None
    minimum: dict[str, float] = field(default_factory=dict)
    maximum: dict[str, float] = field(default_factory=dict)
    equal: dict[str, float] = field(default_factory=dict)
    known: tuple[str, ...] = ()

    def without(
        self, structures: Structures, elements: tuple[int, ...]
    ) -> Bool[torch.Tensor, "n_struct"]:
        """Mask of the structures with none of some elements."""
        listed = torch.tensor(elements, dtype=torch.int64, device=structures.device())
        count = scatter(
            torch.isin(structures.z, listed).long(),
            structures.batch,
            len(structures.num_atoms),
            reduce="sum",
        )

        return count == 0

    def without_others(
        self, structures: Structures, elements: tuple[int, ...]
    ) -> Bool[torch.Tensor, "n_struct"]:
        """Mask of the structures made only of some elements."""
        listed = torch.tensor(elements, dtype=torch.int64, device=structures.device())
        count = scatter(
            (~torch.isin(structures.z, listed)).long(),
            structures.batch,
            len(structures.num_atoms),
            reduce="sum",
        )

        return count == 0

    def __call__(self, structures: Structures) -> Bool[torch.Tensor, "n_struct"]:
        device = structures.device()
        n_struct = len(structures.num_atoms)

        mask = torch.ones(n_struct, dtype=torch.bool, device=device)

        if self.min_atoms is not None:
            mask &= structures.num_atoms >= self.min_atoms

        if self.max_atoms is not None:
            mask &= structures.num_atoms <= self.max_atoms

        if len(self.exclude_elements) > 0:
            mask &= self.without(structures, self.exclude_elements)

        if self.elements is not None:
            mask &= self.without_others(structures, self.elements)

        if self.periodic is True:
            mask &= structures.pbc.all(dim=1)
        elif self.periodic is False:
            mask &= ~structures.pbc.any(dim=1)

        if self.max_energy_above_hull is not None:
            # NaN (unknown) compares False and is dropped
            mask &= structures.energy_above_hull <= self.max_energy_above_hull

        # NaN compares False to anything: a structure without value is dropped
        for name, value in self.minimum.items():
            mask &= values(structures, name) >= value

        for name, value in self.maximum.items():
            mask &= values(structures, name) <= value

        for name, value in self.equal.items():
            mask &= values(structures, name) == value

        for name in self.known:
            mask &= ~values(structures, name).isnan()

        return mask
