import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


def scatter_sum(
    src: Float[torch.Tensor, "n ..."],
    index: Int[torch.Tensor, "n"],
    n_out: int,
) -> Float[torch.Tensor, "n_out ..."]:
    """Sum src values into n_out slots according to index.

    Args:
        src: Source tensor of shape (N, *).
        index: Slot index for each row, shape (N,).
        n_out: Number of output slots.

    Returns:
        Tensor of shape (n_out, *) with summed values.
    """
    out = src.new_zeros(n_out, *src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_reduce_(0, idx, src, reduce="sum", include_self=False)


def scatter_mean(
    src: Float[torch.Tensor, "n ..."],
    index: Int[torch.Tensor, "n"],
    n_out: int,
) -> Float[torch.Tensor, "n_out ..."]:
    """Average src values into n_out slots according to index.

    Args:
        src: Source tensor of shape (N, *).
        index: Slot index for each row, shape (N,).
        n_out: Number of output slots.

    Returns:
        Tensor of shape (n_out, *) with averaged values. Empty slots are zero.
    """
    out = src.new_zeros(n_out, *src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_reduce_(0, idx, src, reduce="mean", include_self=False)


def batched_cartesian_product_2(
    sizes_a: Int[torch.Tensor, "n_struct"],
    sizes_b: Int[torch.Tensor, "n_struct"],
) -> tuple[
    Int[torch.Tensor, "total 2"],
    Int[torch.Tensor, "total"],
    Int[torch.Tensor, "n_struct+1"],
]:
    """Batched cartesian product of two ranges with variable sizes.

    For each structure i, enumerates all pairs (a, b) with a in [0, sizes_a[i])
    and b in [0, sizes_b[i]), without loops.

    Args:
        sizes_a: Size of the first range per structure, shape (n_struct,).
        sizes_b: Size of the second range per structure, shape (n_struct,).

    Returns:
        indices: Stacked pair indices, shape (total, 2).
        batch: Structure index per pair, shape (total,).
        ptr: CSR pointers into the flat output, shape (n_struct+1,).
    """
    device = sizes_a.device
    num_per_struct = sizes_a * sizes_b
    ptr = F.pad(torch.cumsum(num_per_struct, dim=0), (1, 0), value=0)
    total = int(ptr[-1])

    batch = torch.arange(len(sizes_a), device=device).repeat_interleave(num_per_struct)
    local = torch.arange(total, device=device) - ptr[batch]

    idx_a = local // sizes_b[batch]
    idx_b = local %  sizes_b[batch]

    return torch.stack([idx_a, idx_b], dim=-1), batch, ptr


def batched_cartesian_product_3(
    sizes_a: Int[torch.Tensor, "n_struct"],
    sizes_b: Int[torch.Tensor, "n_struct"],
    sizes_c: Int[torch.Tensor, "n_struct"],
) -> tuple[
    Int[torch.Tensor, "total 3"],
    Int[torch.Tensor, "total"],
    Int[torch.Tensor, "n_struct+1"],
]:
    """Batched cartesian product of three ranges with variable sizes.

    For each structure i, enumerates all triplets (a, b, c) with a in [0, sizes_a[i]),
    b in [0, sizes_b[i]), c in [0, sizes_c[i]), without loops.

    Args:
        sizes_a: Size of the first range per structure, shape (n_struct,).
        sizes_b: Size of the second range per structure, shape (n_struct,).
        sizes_c: Size of the third range per structure, shape (n_struct,).

    Returns:
        indices: Stacked triplet indices, shape (total, 3).
        batch: Structure index per triplet, shape (total,).
        ptr: CSR pointers into the flat output, shape (n_struct+1,).
    """
    device = sizes_a.device
    num_per_struct = sizes_a * sizes_b * sizes_c
    ptr = F.pad(torch.cumsum(num_per_struct, dim=0), (1, 0), value=0)
    total = int(ptr[-1])

    batch = torch.arange(len(sizes_a), device=device).repeat_interleave(num_per_struct)
    local = torch.arange(total, device=device) - ptr[batch]

    Sb = sizes_b[batch]
    Sc = sizes_c[batch]

    idx_a = local // (Sb * Sc)
    idx_b = (local // Sc) % Sb
    idx_c = local %  Sc

    return torch.stack([idx_a, idx_b, idx_c], dim=-1), batch, ptr
