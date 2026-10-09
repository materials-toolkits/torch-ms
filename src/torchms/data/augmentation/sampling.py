"""Random elements of the groups acting on crystal structures.

A generator must be on the device of the tensors to create. Without generator,
the global random state of torch is used, which is seeded in each worker of a
DataLoader.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int


def random_permutation(
    batch: Int[torch.Tensor, "n_atoms"], generator: torch.Generator | None = None
) -> Int[torch.Tensor, "n_atoms"]:
    """Draw a uniform permutation of the atoms of each structure.

    Args:
        batch: Structure index of each atom, in ascending order, shape (n_atoms,).
        generator: Random number generator.

    Returns:
        Index into the atoms of the batch, shape (n_atoms,). The atoms of a
        structure are permuted among themselves.

    Examples:
        >>> batch = torch.tensor([0, 0, 0, 1, 1])
        >>> permutation = random_permutation(batch)
        >>> batch[permutation]
        tensor([0, 0, 0, 1, 1])
    """
    order = torch.randperm(len(batch), generator=generator, device=batch.device)

    # shuffle the whole batch, then put back the structures in order
    return order[torch.argsort(batch[order], stable=True)]


def random_orthogonal(
    n: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    proper: bool = False,
) -> Float[torch.Tensor, "n 3 3"]:
    """Draw matrices of O(3), or of SO(3), uniformly (Haar measure).

    In O(3), half of them are rotations (determinant 1), half are
    rotoreflections (determinant -1).

    Args:
        n: Number of matrices.
        device: Device of the matrices.
        generator: Random number generator.
        proper: If True, only rotations are drawn: the matrices are in SO(3).

    Returns:
        Orthogonal matrices with shape (n, 3, 3).

    Examples:
        >>> rotation = random_orthogonal(4, torch.device("cpu"))
        >>> identity = torch.bmm(rotation, rotation.transpose(1, 2))
        >>> torch.allclose(identity, torch.eye(3).expand(4, 3, 3), atol=1e-5)
        True
        >>> rotation = random_orthogonal(4, torch.device("cpu"), proper=True)
        >>> torch.linalg.det(rotation).round()
        tensor([1., 1., 1., 1.])
    """
    gaussian = torch.randn(n, 3, 3, generator=generator, device=device)

    # Q of the QR decomposition with a positive diagonal of R, which is
    # uniform: the columns orthonormalised one after the other (Gram-Schmidt).
    # torch.linalg.qr gives it too, forty times slower on a GPU for matrices
    # this small.
    a, b, c = gaussian.unbind(dim=2)
    first = F.normalize(a, dim=1)

    # subtracted twice: once leaves a part along the first column when the
    # two columns are almost parallel
    b = b - (first * b).sum(dim=1, keepdim=True) * first
    b = b - (first * b).sum(dim=1, keepdim=True) * first
    second = F.normalize(b, dim=1)

    # the third column is normal to the two others, on the side of c
    normal = torch.cross(first, second, dim=1)
    side = (normal * c).sum(dim=1, keepdim=True)
    third = F.normalize(normal, dim=1) * torch.where(side < 0, -1.0, 1.0)

    q = torch.stack([first, second, third], dim=2)

    if proper:
        # in odd dimension -Q is the rotation of a rotoreflection Q
        q = q * torch.linalg.det(q).sign().view(n, 1, 1)  # pylint: disable=not-callable

    return q


def random_translation(
    n: int, device: torch.device, generator: torch.Generator | None = None
) -> Float[torch.Tensor, "n 3"]:
    """Draw translations uniformly in the cell, in fractional coordinates.

    Examples:
        >>> random_translation(4, torch.device("cpu")).shape
        torch.Size([4, 3])
    """
    return torch.rand(n, 3, generator=generator, device=device)


def random_unimodular(
    n: int,
    num_shears: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Int[torch.Tensor, "n 3 3"]:
    """Draw matrices of SL(3, Z) as products of random elementary matrices.

    An elementary matrix adds or subtracts a row to another one. They generate
    SL(3, Z): every element is reached with enough of them, small entries are
    more likely. The cells get more skewed as num_shears grows.

    Args:
        n: Number of matrices.
        num_shears: Number of elementary matrices in each product.
        device: Device of the matrices.
        generator: Random number generator.

    Returns:
        Integer matrices of determinant 1 with shape (n, 3, 3).

    Examples:
        >>> basis = random_unimodular(4, num_shears=3, device=torch.device("cpu"))
        >>> torch.linalg.det(basis.double()).round().long()
        tensor([1, 1, 1, 1])
    """
    basis = torch.eye(3, dtype=torch.int64, device=device).repeat(n, 1, 1)
    index = torch.arange(n, device=device)

    for _ in range(num_shears):
        # one of the 6 pairs of different rows, and a sign
        pair = torch.randint(0, 6, (n,), generator=generator, device=device)
        sign = 2 * torch.randint(0, 2, (n, 1), generator=generator, device=device) - 1

        target = pair // 2
        source = (target + 1 + pair % 2) % 3

        basis[index, target] += sign * basis[index, source]

    return basis


def random_repeat(
    num_atoms: Int[torch.Tensor, "n_struct"],
    max_repeat: int,
    max_atoms: int | None = None,
    generator: torch.Generator | None = None,
    pbc: Bool[torch.Tensor, "n_struct 3"] | None = None,
) -> Int[torch.Tensor, "n_struct 3"]:
    """Draw the number of cells of a supercell along each lattice vector.

    Args:
        num_atoms: Number of atoms per structure, shape (n_struct,).
        max_repeat: Maximum number of cells along a lattice vector, drawn
            uniformly from 1 to max_repeat.
        max_atoms: Maximum number of atoms of a supercell. A structure whose
            supercell is larger is not repeated.
        generator: Random number generator.
        pbc: Whether each structure is periodic along each lattice vector,
            shape (n_struct, 3). A structure is not repeated along a lattice
            vector along which it is not periodic. Every structure is periodic
            by default.

    Returns:
        Number of cells with shape (n_struct, 3).

    Examples:
        >>> random_repeat(torch.tensor([8, 8]), max_repeat=3, max_atoms=8)
        tensor([[1, 1, 1],
                [1, 1, 1]])
    """
    repeat = torch.randint(
        1,
        max_repeat + 1,
        (len(num_atoms), 3),
        generator=generator,
        device=num_atoms.device,
    )

    if pbc is not None:
        repeat = torch.where(pbc, repeat, 1)

    if max_atoms is not None:
        repeat[num_atoms * repeat.prod(dim=1) > max_atoms] = 1

    return repeat
