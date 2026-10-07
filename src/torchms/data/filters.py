"""Filters to select structures during preprocessing."""

from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Bool

from torchms.data.batching import Structures
from torchms.graph.sparse import scatter


@dataclass
class Filter:
    """Select structures of a batch, returns a mask of the structures to keep.

    Args:
        max_atoms: Maximum number of atoms per structure.
        exclude_elements: Atomic numbers that must not appear in a structure.
        max_energy_above_hull: Maximum energy above the hull in eV/atom
            (0 for stable, 0.1 is commonly used for meta-stable). Structures
            without energy above the hull are dropped.

    Examples:
        >>> structures = Structures.collate(
        ...     [Structures.diamond(), Structures.zincblende()]
        ... )
        >>> Filter(exclude_elements=(16,))(structures)
        tensor([ True, False])
        >>> Filter(max_atoms=4)(structures)
        tensor([False, False])
    """

    max_atoms: Optional[int] = None
    exclude_elements: tuple[int, ...] = ()
    max_energy_above_hull: Optional[float] = None

    def __call__(self, structures: Structures) -> Bool[torch.Tensor, "n_struct"]:
        device = structures.device()
        n_struct = len(structures.num_atoms)

        mask = torch.ones(n_struct, dtype=torch.bool, device=device)

        if self.max_atoms is not None:
            mask &= structures.num_atoms <= self.max_atoms

        if len(self.exclude_elements) > 0:
            excluded = torch.tensor(
                self.exclude_elements, dtype=torch.int64, device=device
            )
            num_excluded = scatter(
                torch.isin(structures.z, excluded).long(),
                structures.batch,
                n_struct,
                reduce="sum",
            )
            mask &= num_excluded == 0

        if self.max_energy_above_hull is not None:
            # NaN (unknown) compares False and is dropped
            mask &= structures.energy_above_hull <= self.max_energy_above_hull

        return mask
