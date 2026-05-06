import torch
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
