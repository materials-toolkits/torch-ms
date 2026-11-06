"""Symmetry operations on crystal structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from torchms.data.batching import Structures
from torchms.data.constants import get_symmetry_operations, get_wyckoff_positions
from torchms.graph.sparse import scatter

# atoms closer than this in fractional coordinates are one, as in
# ase.spacegroup.crystal (symprec)
MERGE_TOLERANCE = 1e-3


def generate_symmetric_structure(
    lattice: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    spacegroup: int | str,
    tol: float = MERGE_TOLERANCE,
) -> Structures:
    """Generate full symmetric crystal by applying spacegroup operations.

    The operations are the ones of the standard setting of the group, see
    torchms.data.constants.get_symmetry_operations.

    Args:
        lattice: Lattice vectors with shape (3, 3) in Angstroms.
        positions: Atomic positions in fractional coordinates [0, 1) with shape (N, 3).
        atomic_numbers: Atomic numbers with shape (N,).
        spacegroup: Space group number (int) or symbol (str).
        tol: Atoms closer than this, in fractional coordinates and across the
            faces of the cell, are merged.

    Returns:
        Structures object with all symmetry-equivalent atoms.
    """
    operations = get_symmetry_operations(spacegroup).to(positions.dtype)

    return expand_symmetry(lattice, positions, atomic_numbers, operations, tol)


def expand_symmetry(
    lattice: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    operations: torch.Tensor,
    tol: float = MERGE_TOLERANCE,
) -> Structures:
    """Apply symmetry operations to the atoms of an asymmetric unit.

    Args:
        lattice: Lattice vectors with shape (3, 3) in Angstroms.
        positions: Fractional coordinates of the atoms, shape (N, 3).
        atomic_numbers: Atomic numbers, shape (N,).
        operations: Affine operations on fractional coordinates, shape
            (n_ops, 4, 4): [R|t; 0 0 0 1].
        tol: Images closer than this, in fractional coordinates and across the
            faces of the cell, are one atom: the image of the first atom of
            the asymmetric unit, whatever the elements, as in
            ase.spacegroup.crystal.

    Returns:
        Structures with one atom per distinct image, in the order of the atoms
        then of the operations.

    Examples:
        >>> identity = torch.eye(4)
        >>> inversion = torch.diag(torch.tensor([-1.0, -1.0, -1.0, 1.0]))
        >>> x = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
        >>> crystal = expand_symmetry(
        ...     torch.eye(3), x, torch.tensor([11, 17]),
        ...     torch.stack([identity, inversion]),
        ... )
        >>> crystal.x.tolist()
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
        >>> crystal.z.tolist()
        [11, 17, 17]
    """
    rot = operations[:, :3, :3].to(positions.dtype)
    trans = operations[:, :3, 3].to(positions.dtype)

    # images of each atom in turn
    new_positions = torch.einsum("gmn,xn->xgm", rot, positions) + trans
    all_positions = torch.remainder(new_positions.reshape(-1, 3), 1.0)
    all_atomic_numbers = atomic_numbers.repeat_interleave(operations.shape[0])

    unique_mask = _find_unique_atoms(all_positions, tol)

    return Structures(
        lattice=lattice.unsqueeze(0),
        x=all_positions[unique_mask],
        z=all_atomic_numbers[unique_mask],
        _validate=True,
    )


def _find_unique_atoms(
    positions: torch.Tensor, tol: float = MERGE_TOLERANCE
) -> torch.Tensor:
    """Mask of the first atom of each group of atoms at the same place.

    Two atoms are at the same place when their fractional coordinates differ
    by less than tol, modulo 1: 0.0 and 0.9999 are the same coordinate.

    Args:
        positions: Fractional positions with shape (N, 3).
        tol: Tolerance on the distance in fractional coordinates.

    Returns:
        Boolean mask of the atoms kept, shape (N,).
    """
    difference = positions.unsqueeze(1) - positions.unsqueeze(0)
    difference = difference - torch.round(difference)
    same_place = torch.sum(difference**2, dim=-1) < tol * tol

    # an atom is dropped when an atom before it is at the same place
    duplicate = torch.tril(same_place, diagonal=-1).any(dim=1)

    return ~duplicate


@dataclass(frozen=True)
class WyckoffSites:
    """Atoms of structures on Wyckoff positions of their space groups.

    The atoms of a structure are grouped in sites, each one an orbit of a
    Wyckoff position: every atom of a site is the image of a free point by an
    operation of the position, the first atom of the site by the first
    operation. The sites of a structure follow each other, and the atoms of a
    site too. The operations are the ones of get_wyckoff_positions.

    Build them with from_letters.

    Args:
        spacegroup: Space group of each structure, shape (n_struct,).
        num_atoms: Number of atoms of each structure, shape (n_struct,).
        operations: Affine map of each atom from the free point of its site,
            shape (n_atoms, 4, 4).
        pseudo_inverses: Pseudo-inverse of the linear part of each map, shape
            (n_atoms, 3, 3): a vector of an atom brought back to the free
            point.
        site: Site of each atom, from 0 over the whole batch, shape (n_atoms,).
        site_ptr: First atom of each site, and the number of atoms, shape
            (n_sites + 1,).
    """

    spacegroup: Int[torch.Tensor, "n_struct"]
    num_atoms: Int[torch.Tensor, "n_struct"]
    operations: Float[torch.Tensor, "n_atoms 4 4"]
    pseudo_inverses: Float[torch.Tensor, "n_atoms 3 3"]
    site: Int[torch.Tensor, "n_atoms"]
    site_ptr: Int[torch.Tensor, "n_sites_plus_1"]

    @classmethod
    def from_letters(
        cls,
        spacegroups: Sequence[int],
        letters: Sequence[str | Sequence[str]],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> WyckoffSites:
        """Sites of structures given by their space groups and Wyckoff letters.

        Args:
            spacegroups: Space group of each structure, from 1 to 230.
            letters: Wyckoff positions of the sites of each structure: a
                string of letters ("adg"), or a list of positions, each one
                its letter with its multiplicity before it or not
                (["2a", "2d", "4g"]). A position appears once per site.
            device: Device of the tensors.
            dtype: Type of the operations.

        Raises:
            ValueError: If a space group or a letter does not exist, or a
                multiplicity is not the one of its position.

        Examples:
            >>> sites = WyckoffSites.from_letters([58, 225], ["adg", ["4a"]])
            >>> sites.num_atoms.tolist(), sites.site_ptr.tolist()
            ([8, 4], [0, 2, 4, 8, 12])
        """
        if len(spacegroups) != len(letters):
            raise ValueError("one list of Wyckoff positions per space group")

        operations, num_atoms, multiplicities = [], [], []
        for spacegroup, positions in zip(spacegroups, letters):
            if not 1 <= spacegroup <= 230:
                raise ValueError(f"space group {spacegroup} does not exist")
            table = {p.letter: p for p in get_wyckoff_positions(int(spacegroup))}
            total = 0
            for position in positions:
                letter, multiplicity = position[-1], position[:-1]
                if letter not in table:
                    raise ValueError(
                        f"space group {spacegroup} has no Wyckoff position {letter}"
                    )
                wyckoff = table[letter]
                if multiplicity and int(multiplicity) != wyckoff.multiplicity:
                    raise ValueError(
                        f"the position {letter} of the space group {spacegroup} "
                        f"has the multiplicity {wyckoff.multiplicity}, "
                        f"not {multiplicity}"
                    )
                operations.append(wyckoff.operations)
                multiplicities.append(wyckoff.multiplicity)
                total += wyckoff.multiplicity
            num_atoms.append(total)

        affine = torch.cat(operations).to(device=device, dtype=dtype)
        counts = torch.tensor(multiplicities, device=device)

        return cls(
            spacegroup=torch.tensor(spacegroups, device=device),
            num_atoms=torch.tensor(num_atoms, device=device),
            operations=affine,
            # pylint: disable-next=not-callable
            pseudo_inverses=torch.linalg.pinv(affine[:, :3, :3]),
            site=torch.arange(len(counts), device=device).repeat_interleave(counts),
            site_ptr=F.pad(torch.cumsum(counts, dim=0), (1, 0), value=0),
        )

    @property
    def num_sites(self) -> int:
        """Number of sites of the batch."""
        return len(self.site_ptr) - 1

    @property
    def rotations(self) -> Float[torch.Tensor, "n_atoms 3 3"]:
        """Linear part of the map of each atom."""
        return self.operations[:, :3, :3]

    def anchors(self, values: Float[torch.Tensor, "n_atoms ..."]) -> torch.Tensor:
        """Values of the first atom of each site, shape (n_sites, ...)."""
        return values[self.site_ptr[:-1]]

    def unpool_sites(self, values: torch.Tensor) -> torch.Tensor:
        """Values of the sites given to their atoms, shape (n_atoms, ...).

        Examples:
            >>> sites = WyckoffSites.from_letters([58], ["ag"])
            >>> sites.unpool_sites(torch.tensor([3, 8])).tolist()
            [3, 3, 8, 8, 8, 8]
        """
        return values[self.site]

    def mean_over_sites(
        self, values: Float[torch.Tensor, "n_atoms ..."]
    ) -> Float[torch.Tensor, "n_sites ..."]:
        """Mean of the values of the atoms of each site."""
        return scatter(values, self.site, self.num_sites, "mean")

    def place(
        self, x: Float[torch.Tensor, "n_atoms 3"]
    ) -> Float[torch.Tensor, "n_atoms 3"]:
        """Fractional coordinates of each atom from the ones of the first of its site.

        The coordinates of the first atom of a site are the free point: each
        atom of the site, the first one included, is its image by its
        operation, in the cell. The coordinates of the other atoms are not
        read.

        Examples:
            >>> sites = WyckoffSites.from_letters([58], [["2a", "4g"]])
            >>> x = torch.tensor([[0.3, 0.3, 0.3]] * 2 + [[0.1, 0.2, 0.3]] * 4)
            >>> sites.place(x)
            tensor([[0.0000, 0.0000, 0.0000],
                    [0.5000, 0.5000, 0.5000],
                    [0.1000, 0.2000, 0.0000],
                    [0.9000, 0.8000, 0.0000],
                    [0.4000, 0.7000, 0.5000],
                    [0.6000, 0.3000, 0.5000]])
        """
        free = self.unpool_sites(self.anchors(x))
        homogeneous = torch.cat([free, torch.ones_like(free[:, :1])], dim=-1)

        return (self.operations @ homogeneous.unsqueeze(-1)).squeeze(-1)[:, :3] % 1.0

    def spread(
        self, vectors: Float[torch.Tensor, "n_atoms 3"]
    ) -> Float[torch.Tensor, "n_atoms 3"]:
        """A displacement of each atom, from the one of the first atom of its site.

        The vector of the first atom is brought back to the free point, then
        moved to each atom of the site by the linear part of its operation:
        the atoms of a site stay images of one point. The vectors of the other
        atoms are not read.
        """
        free = self.anchors(vectors)
        free = (self.anchors(self.pseudo_inverses) @ free.unsqueeze(-1)).squeeze(-1)

        return (self.rotations @ self.unpool_sites(free).unsqueeze(-1)).squeeze(-1)

    def symmetrize(
        self, vectors: Float[torch.Tensor, "n_atoms 3"]
    ) -> Float[torch.Tensor, "n_atoms 3"]:
        """Displacements of the atoms which keep each site an orbit.

        The vector of each atom is brought back to the free point of its site,
        the mean of the site is taken there and moved to each atom by the
        linear part of its operation.

        Examples:
            >>> sites = WyckoffSites.from_letters([58], ["g"])
            >>> sites.symmetrize(torch.tensor([[1.0, 1.0, 1.0]] + [[0.0] * 3] * 3))
            tensor([[ 0.2500,  0.2500,  0.0000],
                    [-0.2500, -0.2500,  0.0000],
                    [-0.2500,  0.2500,  0.0000],
                    [ 0.2500, -0.2500,  0.0000]])
        """
        free = torch.einsum("bij, bj-> bi", self.pseudo_inverses, vectors)
        free = self.unpool_sites(self.mean_over_sites(free))

        return (self.rotations @ free.unsqueeze(-1)).squeeze(-1)
