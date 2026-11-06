# Derived from All-atom Diffusion Transformers
# (https://github.com/facebookresearch/all-atom-diffusion-transformer),
# Copyright (c) Meta Platforms, Inc. and affiliates, CC BY-NC 4.0 (see LICENSE
# in this directory), itself adapted from DiT
# (https://github.com/facebookresearch/DiT).
#
# Changes: this file is a rewrite of src/models/denoisers/dit.py of ADiT, for
# inference only: the dropout of the labels and the initialization of the
# weights, which are for the training, are removed, and so is the guidance,
# which the model does in torchms.models.adit.model. The embeddings of the time
# and of the index of a token use torchms.models.layers.basis. Names of modules
# and parameters follow ADiT to load its weights.
"""DiT, the diffusion transformer of ADiT.

The network reads the noisy latent vector of each atom, with the prediction of
the step before (self-conditioning), and gives the latent vector without noise.
A structure is a sequence of tokens, one per atom, padded to the longest one of
the batch. Every block is a transformer layer whose normalizations are shifted
and scaled by a vector of the structure, and whose residuals are gated by it
(adaLN-Zero): the sum of the embeddings of the time, of the dataset and of the
space group of the structure.

DiT is published in "Scalable Diffusion Models with Transformers", Peebles and
Xie (arXiv:2212.09748); ADiT in "All-atom Diffusion Transformers: Unified
generative modelling of molecules and materials", Joshi, Fu, Liao,
Gharakhanyan, Miller, Sriram and Ulissi (arXiv:2503.03965).
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Bool, Float, Int
from torch import nn

from torchms.models.adit.config import DenoiserConfig
from torchms.models.layers.basis import sinusoidal_frequencies, sinusoids


def index_embedding(
    index: Int[torch.Tensor, "..."], dim: int, max_len: int
) -> Float[torch.Tensor, "... dim"]:
    """Embedding of the index of a token in its sequence, as in ADiT.

    sin(pi i f) then cos(pi i f), for dim / 2 frequencies f from 1 down to
    max_len^(-1 + 2 / dim): the period of the slowest one is about 2 max_len.

    Args:
        index: Index of each token, from 0.
        dim: Size of the embedding, an even number.
        max_len: Inverse of the lowest frequency, without its last step.

    Examples:
        >>> index_embedding(torch.tensor([0, 1]), 4, 2048).round(decimals=4) + 0.0
        tensor([[ 0.0000,  0.0000,  1.0000,  1.0000],
                [ 0.0000,  0.0694, -1.0000,  0.9976]])
    """
    frequencies = sinusoidal_frequencies(
        dim, max_len, endpoint=False, device=index.device
    )

    return sinusoids(index.unsqueeze(-1) * math.pi * frequencies)


def modulate(
    x: Float[torch.Tensor, "n_struct n_tokens d"],
    shift: Float[torch.Tensor, "n_struct d"],
    scale: Float[torch.Tensor, "n_struct d"],
) -> Float[torch.Tensor, "n_struct n_tokens d"]:
    """Features scaled and shifted by a vector of their structure."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embedding of the time: sines and cosines, then a network."""

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(config.frequency_embedding_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def forward(
        self, t: Float[torch.Tensor, "n_struct"]
    ) -> Float[torch.Tensor, "n_struct d"]:
        """Embedding of the time of each structure, from 0 to 1."""
        frequencies = sinusoidal_frequencies(
            self.config.frequency_embedding_dim,
            self.config.time_max_period,
            endpoint=False,
            device=t.device,
        )

        return self.mlp(sinusoids(t.unsqueeze(1) * frequencies, cosine_first=True))


class LabelEmbedder(nn.Module):
    """Embedding of a class: the class 0 stands for no class."""

    def __init__(self, num_classes: int, d_model: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, d_model)

    def forward(
        self, labels: Int[torch.Tensor, "n_struct"]
    ) -> Float[torch.Tensor, "n_struct d"]:
        """Embedding of the class of each structure."""
        return self.embedding_table(labels)


class Mlp(nn.Module):
    """Feed-forward network of a block, with the GELU approximated by tanh."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... d"]:
        """Features after the network."""
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """Transformer layer with adaLN-Zero conditioning."""

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        d_model, eps = config.d_model, config.layer_norm_eps
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads=config.nhead, bias=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
        self.mlp = Mlp(d_model, config.mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(  # pylint: disable=invalid-name
            nn.SiLU(), nn.Linear(d_model, 6 * d_model)
        )

    def forward(
        self,
        x: Float[torch.Tensor, "n_struct n_tokens d"],
        c: Float[torch.Tensor, "n_struct d"],
        mask: Bool[torch.Tensor, "n_struct n_tokens"],
    ) -> Float[torch.Tensor, "n_struct n_tokens d"]:
        """Features of the tokens after the block.

        Args:
            x: Features of the tokens.
            c: Vector of each structure.
            mask: True for a token, False for the padding.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attention, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attention

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)

        return x + gate_mlp.unsqueeze(1) * self.mlp(h)


class FinalLayer(nn.Module):
    """Normalization shifted and scaled by the structure, then a linear layer."""

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        d_model = config.d_model
        self.norm_final = nn.LayerNorm(
            d_model, elementwise_affine=False, eps=config.layer_norm_eps
        )
        self.linear = nn.Linear(d_model, config.latent_dim)
        self.adaLN_modulation = nn.Sequential(  # pylint: disable=invalid-name
            nn.SiLU(), nn.Linear(d_model, 2 * d_model)
        )

    def forward(
        self,
        x: Float[torch.Tensor, "n_struct n_tokens d"],
        c: Float[torch.Tensor, "n_struct d"],
    ) -> Float[torch.Tensor, "n_struct n_tokens latent_dim"]:
        """Output of each token."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)

        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """Diffusion transformer on the latent vectors of the atoms of structures.

    Args:
        config: Hyperparameters of the network.

    Examples:
        >>> from torchms.models.adit import Config
        >>> network = DiT(Config.load().denoiser).eval()
        >>> mask = torch.tensor([[True, True, False], [True, True, True]])
        >>> with torch.no_grad():
        ...     x = network(
        ...         torch.randn(2, 3, 8),
        ...         torch.tensor([0.5, 0.9]),
        ...         torch.tensor([1, 2]),
        ...         torch.tensor([0, 0]),
        ...         mask,
        ...     )
        >>> x.shape, bool((x[0, 2] == 0).all())
        (torch.Size([2, 3, 8]), True)
    """

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config
        self.x_embedder = nn.Linear(2 * config.latent_dim, config.d_model)
        self.t_embedder = TimestepEmbedder(config)
        self.dataset_embedder = LabelEmbedder(config.num_datasets, config.d_model)
        self.spacegroup_embedder = LabelEmbedder(config.num_spacegroups, config.d_model)
        self.blocks = nn.ModuleList(
            [DiTBlock(config) for _ in range(config.num_layers)]
        )
        self.final_layer = FinalLayer(config)

    def forward(  # pylint: disable=too-many-positional-arguments
        self,
        x: Float[torch.Tensor, "n_struct n_tokens latent_dim"],
        t: Float[torch.Tensor, "n_struct"],
        dataset: Int[torch.Tensor, "n_struct"],
        spacegroup: Int[torch.Tensor, "n_struct"],
        mask: Bool[torch.Tensor, "n_struct n_tokens"],
        x_sc: Float[torch.Tensor, "n_struct n_tokens latent_dim"] | None = None,
    ) -> Float[torch.Tensor, "n_struct n_tokens latent_dim"]:
        """Latent vectors without noise, from noisy ones.

        Args:
            x: Noisy latent vector of each atom, the atoms of a structure in a
                row, padded.
            t: Time of each structure, from 0 (noise) to 1 (no noise).
            dataset: Class of the dataset of each structure, 0 for none.
            spacegroup: Space group of each structure, 0 for none.
            mask: True for an atom, False for the padding.
            x_sc: Prediction of the step before, zero if None.

        Returns:
            Latent vector of each atom without noise, zero on the padding.
        """
        config = self.config
        index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        if x_sc is None:
            x_sc = torch.zeros_like(x)

        x = self.x_embedder(torch.cat([x, x_sc], dim=-1)) + index_embedding(
            index, config.d_model, config.position_max_len
        )
        c = (
            self.t_embedder(t)
            + self.dataset_embedder(dataset)
            + self.spacegroup_embedder(spacegroup)
        )

        for block in self.blocks:
            x = block(x, c, mask)

        return self.final_layer(x, c) * mask.unsqueeze(-1)
