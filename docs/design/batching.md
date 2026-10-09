# Batching convention

Crystals have different numbers of atoms, so a batch cannot be a dense tensor without padding.
TorchMS stores a batch as **flat concatenated tensors** with two index tensors, as PyTorch
Geometric does:

- `batch` (n_elements,): index of the structure of each element;
- `ptr` (n_struct + 1,): index of the first element of each structure, then the total.

"Elements" are atoms for `Structures` (`batch`, `ptr`) and edges for `Graph` (`edge_batch`,
`edge_ptr`).

```{.python notest}
ptr = F.pad(torch.cumsum(counts, dim=0), (1, 0), value=0)   # ptr from counts
batch = torch.arange(len(counts), device=device).repeat_interleave(counts)
```

## No loop over structures

Operations are written on the whole batch. Sums per structure use `scatter`:

```{.python notest}
from torchms.graph.sparse import scatter

energy = scatter(atom_energies, structures.batch, n_struct)          # sum over atoms
features = scatter(messages, graph.edge_index[1], n_atoms, "mean")   # mean over edges
```

Combinations inside each structure (pairs of atoms, atoms × periodic images) use a flat index
decomposed like a number in a mixed base. `batched_cartesian_product_2` and
`batched_cartesian_product_3` implement it:

```{.python notest}
from torchms.graph.sparse import batched_cartesian_product_2

indices, batch, ptr = batched_cartesian_product_2(ptr_a, ptr_b)
# indices[:, 0] in the first set, indices[:, 1] in the second, for each structure
```

## Devices

New tensors are created on the device of the input (`device=` is always passed), and a
function never moves its inputs: a batch on two devices is an error of the caller, reported by
`InvalidStructure`.

## What is not batched

Two places work on something else than tensors, by design:

- the convex hull (`torchms.data.hull`) loops over chemical systems, not structures, and
  uses numpy and SciPy;
- reading CIF files loops over data blocks.

## The HDF5 files

A file stores the same flat tensors. Its `ptr` has one value per structure (the first atom),
without the final total: batches can be appended to a file one after the other.

The named properties are in a group `properties`, one array per name. `pbc` says which
structures are periodic; a file without it holds crystals only, see
[Periodicity](periodicity.md).
