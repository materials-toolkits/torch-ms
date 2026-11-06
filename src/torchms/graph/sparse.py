from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, UInt32, Num

Reduce = Literal["sum", "mean", "prod", "amax", "amin", "hash"]


def mix32(x: UInt32[torch.Tensor, "..."]) -> UInt32[torch.Tensor, "..."]:
    """
    shith register not supported for torch.uint32
    Examples:
        >>> mix32(torch.tensor(42, dtype=torch.uint32))
        tensor(142593372, dtype=torch.uint32)

    https://dlib.net/dlib/general_hash/murmur_hash3.h.html
    """
    x = x.to(torch.int64).clone()
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x.to(torch.uint32)


def scatter(
    src: Num[torch.Tensor, "n ..."],
    index: Int[torch.Tensor, "n"],
    n_out: int,
    reduce: Reduce = "sum",
) -> Float[torch.Tensor, "n_out ..."]:
    """Aggregate src values into n_out slots according to index.

    Args:
        src: Source tensor of shape (N, *).
        index: Slot index for each row, shape (N,).
        n_out: Number of output slots.
        reduce: Reduction to apply. "sum", "mean", "prod", "amax", "amin" are
            forwarded to scatter_reduce_. "hash" computes a permutation-invariant
            hash of the contributing elements via power sums of mix32 values.

    Returns:
        Tensor of shape (n_out, *). Empty slots are zero.

    Examples:
        >>> src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> index = torch.tensor([0, 0, 1, 2])
        >>> scatter(src, index, n_out=3)
        tensor([3., 3., 4.])
        >>> scatter(src, index, n_out=3, reduce="mean")
        tensor([1.5000, 3.0000, 4.0000])
    """
    _VALID = {"sum", "mean", "prod", "amax", "amin", "hash"}
    if reduce not in _VALID:
        raise ValueError(f"unknown reduce {reduce!r}, expected one of {sorted(_VALID)}")

    if reduce == "hash":
        mix = mix32(src)
        h1 = mix.to(torch.int64)
        h2 = (h1 * h1) & 0xFFFFFFFF
        h3 = (h2 * h1) & 0xFFFFFFFF
        h = (
            scatter(torch.stack([h1, h2, h3], dim=-1), index, n_out, reduce="sum")
            & 0xFFFFFFFF
        )
        h = h[..., 0] ^ h[..., 1] ^ h[..., 2]
        return mix32(h)

    out = src.new_zeros(n_out, *src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_reduce_(0, idx, src, reduce=reduce, include_self=False)


def count_by(index: Int[torch.Tensor, "n"], n_out: int) -> Int[torch.Tensor, "n_out"]:
    """Number of elements of each slot of an index.

    Same as torch.bincount with a minimum length, which reads the largest index
    back from the device before counting: on a GPU, the CPU waits for it.

    Args:
        index: Slot of each element, in [0, n_out), shape (n,).
        n_out: Number of slots.

    Returns:
        Number of elements per slot, shape (n_out,).

    Examples:
        >>> count_by(torch.tensor([0, 0, 2]), 4)
        tensor([2, 0, 1, 0])
    """
    count = torch.zeros(n_out, dtype=torch.int64, device=index.device)

    return count.scatter_add_(0, index, torch.ones_like(index))


def padding_mask(
    num_items: Int[torch.Tensor, "n_struct"],
) -> Bool[torch.Tensor, "n_struct max_items"]:
    """Mask of the items of each structure in a padded layout.

    A batch of structures of different sizes, padded to the size of the
    largest one: the layout of a transformer. The row of a structure is True
    for its items and False for the padding after them.

    Args:
        num_items: Number of items of each structure, shape (n_struct,).

    Returns:
        Mask of shape (n_struct, max_items), max_items the largest size, 0 if
        there is no structure.

    Examples:
        >>> padding_mask(torch.tensor([2, 0, 3]))
        tensor([[ True,  True, False],
                [False, False, False],
                [ True,  True,  True]])
    """
    max_items = int(num_items.max()) if len(num_items) > 0 else 0
    position = torch.arange(max_items, device=num_items.device)

    return position.unsqueeze(0) < num_items.unsqueeze(1)


def to_dense(
    x: Num[torch.Tensor, "n ..."], num_items: Int[torch.Tensor, "n_struct"]
) -> tuple[
    Num[torch.Tensor, "n_struct max_items ..."],
    Bool[torch.Tensor, "n_struct max_items"],
]:
    """Padded layout of the items of a batch, the ones of a structure in a row.

    The inverse is `dense[mask]`, which gives the items back in their order.

    Args:
        x: Values of the items, the ones of a structure together, in the
            order of the structures, shape (n, ...).
        num_items: Number of items of each structure, shape (n_struct,).

    Returns:
        Values padded with zeros, shape (n_struct, max_items, ...), and the
        mask of the items, see padding_mask.

    Examples:
        >>> dense, mask = to_dense(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1, 2]))
        >>> dense
        tensor([[1., 0.],
                [2., 3.]])
        >>> dense[mask]
        tensor([1., 2., 3.])
    """
    mask = padding_mask(num_items)
    dense = x.new_zeros(*mask.shape, *x.shape[1:])
    dense[mask] = x

    return dense, mask


def batched_cartesian_product_2(
    ptr_a: Int[torch.Tensor, "n_struct+1"],
    ptr_b: Int[torch.Tensor, "n_struct+1"],
) -> tuple[
    Int[torch.Tensor, "total 2"],
    Int[torch.Tensor, "total"],
    Int[torch.Tensor, "n_struct+1"],
]:
    """Batched cartesian product of two index ranges given as CSR pointers.

    For each structure i, enumerates all pairs (a, b) with a in [ptr_a[i], ptr_a[i+1])
    and b in [ptr_b[i], ptr_b[i+1]), without loops.

    Args:
        ptr_a: CSR pointers for the first range, shape (n_struct+1,).
        ptr_b: CSR pointers for the second range, shape (n_struct+1,).

    Returns:
        Global pair indices, shape (total, 2), structure index per pair,
        shape (total,), and CSR pointers into the flat output, shape
        (n_struct+1,).

    Examples:
        Two structures: struct 0 with 2+3 atoms, struct 1 with 2+3 atoms.

        >>> ptr_a = torch.tensor([0, 2, 4])  # struct 0: [0,2), struct 1: [2,4)
        >>> ptr_b = torch.tensor([0, 3, 6])  # struct 0: [0,3), struct 1: [3,6)
        >>> indices, batch, ptr = batched_cartesian_product_2(ptr_a, ptr_b)
        >>> indices.shape  # 2×3 + 2×3 = 12 pairs total
        torch.Size([12, 2])
        >>> indices[:6]  # struct 0: a in [0,2), b in [0,3)
        tensor([[0, 0],
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 1],
                [1, 2]])
        >>> indices[6:]  # struct 1: a in [2,4), b in [3,6)
        tensor([[2, 3],
                [2, 4],
                [2, 5],
                [3, 3],
                [3, 4],
                [3, 5]])
    """
    device = ptr_a.device
    sizes_a = ptr_a[1:] - ptr_a[:-1]
    sizes_b = ptr_b[1:] - ptr_b[:-1]

    num_per_struct = sizes_a * sizes_b
    ptr = F.pad(torch.cumsum(num_per_struct, dim=0), (1, 0), value=0)
    total = int(ptr[-1])

    batch = torch.arange(len(sizes_a), device=device).repeat_interleave(num_per_struct)
    local = torch.arange(total, device=device) - ptr[batch]

    idx_a = local // sizes_b[batch] + ptr_a[batch]
    idx_b = local % sizes_b[batch] + ptr_b[batch]

    return torch.stack([idx_a, idx_b], dim=-1), batch, ptr


def batched_cartesian_product_3(
    ptr_a: Int[torch.Tensor, "n_struct+1"],
    ptr_b: Int[torch.Tensor, "n_struct+1"],
    ptr_c: Int[torch.Tensor, "n_struct+1"],
) -> tuple[
    Int[torch.Tensor, "total 3"],
    Int[torch.Tensor, "total"],
    Int[torch.Tensor, "n_struct+1"],
]:
    """Batched cartesian product of three index ranges given as CSR pointers.

    For each structure i, enumerates all triplets (a, b, c) with
    a in [ptr_a[i], ptr_a[i+1]), b in [ptr_b[i], ptr_b[i+1]),
    c in [ptr_c[i], ptr_c[i+1]), without loops.

    Args:
        ptr_a: CSR pointers for the first range, shape (n_struct+1,).
        ptr_b: CSR pointers for the second range, shape (n_struct+1,).
        ptr_c: CSR pointers for the third range, shape (n_struct+1,).

    Returns:
        Global triplet indices, shape (total, 3), structure index per
        triplet, shape (total,), and CSR pointers into the flat output, shape
        (n_struct+1,).

    Examples:
        Two structures with different range sizes.
        Struct 0: a in [0,2), b in [0,3), c in [0,2) → 2×3×2 = 12 triplets.
        Struct 1: a in [2,4), b in [3,5), c in [2,4) → 2×2×2 = 8 triplets.

        >>> ptr_a = torch.tensor([0, 2, 4])
        >>> ptr_b = torch.tensor([0, 3, 5])
        >>> ptr_c = torch.tensor([0, 2, 4])
        >>> indices, batch, out_ptr = batched_cartesian_product_3(ptr_a, ptr_b, ptr_c)
        >>> indices.shape  # 12 + 8 = 20 triplets total
        torch.Size([20, 3])
        >>> out_ptr
        tensor([ 0, 12, 20])
        >>> indices[:12]  # struct 0: global indices in [0,2)×[0,3)×[0,2)
        tensor([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 2, 0],
                [0, 2, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [1, 2, 0],
                [1, 2, 1]])
        >>> indices[12:]  # struct 1: global indices in [2,4)×[3,5)×[2,4)
        tensor([[2, 3, 2],
                [2, 3, 3],
                [2, 4, 2],
                [2, 4, 3],
                [3, 3, 2],
                [3, 3, 3],
                [3, 4, 2],
                [3, 4, 3]])
    """
    device = ptr_a.device
    sizes_a = ptr_a[1:] - ptr_a[:-1]
    sizes_b = ptr_b[1:] - ptr_b[:-1]
    sizes_c = ptr_c[1:] - ptr_c[:-1]

    num_per_struct = sizes_a * sizes_b * sizes_c
    ptr = F.pad(torch.cumsum(num_per_struct, dim=0), (1, 0), value=0)
    total = int(ptr[-1])

    batch = torch.arange(len(sizes_a), device=device).repeat_interleave(num_per_struct)
    local = torch.arange(total, device=device) - ptr[batch]

    Sb = sizes_b[batch]
    Sc = sizes_c[batch]

    idx_a = local // (Sb * Sc) + ptr_a[batch]
    idx_b = (local // Sc) % Sb + ptr_b[batch]
    idx_c = local % Sc + ptr_c[batch]

    return torch.stack([idx_a, idx_b, idx_c], dim=-1), batch, ptr


def batched_join(
    key_a: Int[torch.Tensor, "n_a"],
    key_b: Int[torch.Tensor, "n_b"],
    num_keys: int,
) -> tuple[
    Int[torch.Tensor, "total 2"],
    Int[torch.Tensor, "total"],
    Int[torch.Tensor, "num_keys+1"],
]:
    """All the pairs of elements of two sets which have the same key, without loops.

    The key is, for instance, the atom shared by two edges: the pairs are then
    the pairs of edges around each atom. Elements do not have to be sorted.

    Args:
        key_a: Key of each element of the first set, in [0, num_keys), shape (n_a,).
        key_b: Key of each element of the second set, in [0, num_keys), shape (n_b,).
        num_keys: Number of keys.

    Returns:
        Pairs (a, b) of indices with key_a[a] == key_b[b], shape (total, 2),
        sorted by key, their key, shape (total,), and CSR pointers of the
        pairs of each key, shape (num_keys+1,).

    Examples:
        >>> key_a = torch.tensor([1, 0, 1])
        >>> key_b = torch.tensor([1, 1, 2, 0])
        >>> pairs, key, ptr = batched_join(key_a, key_b, num_keys=3)
        >>> pairs
        tensor([[1, 3],
                [0, 0],
                [0, 1],
                [2, 0],
                [2, 1]])
        >>> key
        tensor([0, 1, 1, 1, 1])
        >>> ptr
        tensor([0, 1, 5, 5])
    """
    order_a = torch.argsort(key_a, stable=True)
    ptr_a = F.pad(torch.cumsum(count_by(key_a, num_keys), dim=0), (1, 0), value=0)

    if key_b is key_a:
        # a set joined with itself is sorted and counted once
        order_b, ptr_b = order_a, ptr_a
    else:
        order_b = torch.argsort(key_b, stable=True)
        ptr_b = F.pad(torch.cumsum(count_by(key_b, num_keys), dim=0), (1, 0), value=0)

    # pairs of positions in the sets sorted by key
    indices, key, ptr = batched_cartesian_product_2(ptr_a, ptr_b)
    pairs = torch.stack([order_a[indices[:, 0]], order_b[indices[:, 1]]], dim=-1)

    return pairs, key, ptr


def equivalent_partition(
    a: Int[torch.Tensor, "n_nodes"],
    b: Int[torch.Tensor, "n_nodes"],
    batch: Int[torch.Tensor, "n_nodes"],
) -> Bool[torch.Tensor, "n_graphs"]:
    """Check whether a and b define the same partition of nodes in each graph.

    Two partitions are equivalent when the mapping from a-classes to b-classes
    is a bijection within each graph: every a-class maps to exactly one b-class
    and vice versa.

    Args:
        a: Partition labels under scheme A, shape (n_nodes,).
        b: Partition labels under scheme B, shape (n_nodes,).
        batch: Graph index per node, shape (n_nodes,).

    Returns:
        Boolean tensor of shape (n_graphs,), True where the partitions are equivalent.

    Examples:
        >>> a = torch.tensor([0, 0, 1, 0, 0, 1])
        >>> b = torch.tensor([5, 5, 9, 5, 9, 9])
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 1])
        >>> equivalent_partition(a, b, batch)
        tensor([ True, False])
    """
    partition_map = torch.unique(torch.stack([a, b, batch]), dim=1)

    sources_a = torch.unique(partition_map[[0, 2]], dim=1)
    sources_b = torch.unique(partition_map[[1, 2]], dim=1)

    count_a = torch.bincount(sources_a[1])
    count_b = torch.bincount(sources_b[1])
    count_pairs = torch.bincount(partition_map[2])

    return (count_a == count_b) & (count_a == count_pairs)
