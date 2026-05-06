from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, UInt32, Num


def mix32(x: UInt32[torch.Tensor, "..."]) -> UInt32[torch.Tensor, "..."]:
    """
    shith register not supported for torch.uint32
    Example:
        >>> mix32(torch.tensor(42, dtype=torch.uint32))
        tensor(142593372, dtype=torch.uint32)

    https://dlib.net/dlib/general_hash/murmur_hash3.h.html
    """
    x = x.to(torch.int64).clone()
    x ^= x >> 16
    x = (x*0x85ebca6b)&0xffffffff
    x ^= x >> 13
    x = (x*0xc2b2ae35)&0xffffffff
    x ^= x >> 16
    return x.to(torch.uint32)


def scatter(
    src: Num[torch.Tensor, "n ..."],
    index: Int[torch.Tensor, "n"],
    n_out: int,
    reduce: Literal["sum", "mean", "prod", "amax", "amin", "hash"] = "sum",
    expand_hash:int=1
) -> Float[torch.Tensor, "n_out ..."]:
    """Aggregate src values into n_out slots according to index.

    Args:
        src: Source tensor of shape (N, *).
        index: Slot index for each row, shape (N,).
        n_out: Number of output slots.
        reduce: Reduction — any value accepted by scatter_reduce_:
            "sum", "mean", "prod", "amax", "amin".

    Returns:
        Tensor of shape (n_out, *). Empty slots are zero.

    Example:
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

    if reduce=="hash":
        mix = mix32(src)
        if expand_hash==1:
            h1 = mix.to(torch.int64)
            h2 = (h1*h1)&0xffffffff
            h3 = (h2*h1)&0xffffffff
            h = scatter(torch.stack([h1, h2, h3],dim=-1), index, n_out, reduce="sum")&0xffffffff
            h = h[...,0]^h[...,1]^h[...,2]
        else:
            h = torch.cumprod(mix.unsqueeze(-1).expand(*mix.shape,expand_hash),dim=-1)&0xffffffff
            h = scatter(h, index, n_out, reduce="sum")&0xffffffff
        return mix32(h)

    out = src.new_zeros(n_out, *src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_reduce_(0, idx, src, reduce=reduce, include_self=False)


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
        indices: Global pair indices, shape (total, 2).
        batch: Structure index per pair, shape (total,).
        ptr: CSR pointers into the flat output, shape (n_struct+1,).

    Example:
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
        indices: Global triplet indices, shape (total, 3).
        batch: Structure index per triplet, shape (total,).
        ptr: CSR pointers into the flat output, shape (n_struct+1,).

    Example:
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
