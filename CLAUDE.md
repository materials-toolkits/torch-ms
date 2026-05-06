# pycrystalgen — Claude Code notes

## Python environment

Always use `uv run` to run Python. Never bare `python3`.

## Batched graph indexing pattern

Variable-size batched tensors follow the PyG-style convention used in `Structures`
(`src/pycrystalgen/data/batching.py`): flat concatenated tensors with `ptr` (CSR-style
index pointers, shape `(n_struct+1,)`) and `batch` (per-element structure index, shape
`(n_elements,)`).

**Never use for loops** over structures. Use `repeat_interleave` + flat-index arithmetic
(`//` and `%`) to generate combinatorial pairs across a heterogeneous batch:

```python
num_edges  = N * N
edge_ptr   = F.pad(torch.cumsum(num_edges, dim=0), (1, 0), value=0)
n_edges    = int(edge_ptr[-1])

edge_batch = torch.arange(len(N)).repeat_interleave(num_edges)
local_e    = torch.arange(n_edges) - edge_ptr[edge_batch]
N_e        = N[edge_batch]

senders   = local_e // N_e + ptr[edge_batch]   # global atom indices
receivers = local_e %  N_e + ptr[edge_batch]
```

`local_e // block_size` → "row", `local_e % block_size` → "column" of any 2D
combinatorial product, fully vectorized across a heterogeneous batch.

This generalises to degree D (triplets, edge pairs, etc.) — a flat index is a mixed-radix number:

| dimension | expression |
|-----------|-----------|
| first     | `local // (N^(D-1))` |
| middle k  | `(local // N^(D-1-k)) % N` |
| last      | `local % N` |

The "size" tensor and "offset" ptr can be node-based (`N`, `ptr`) or edge-based
(`num_edges`, `edge_ptr`) depending on what is being enumerated. Example for triplets:

```python
num_triplets  = N * N * N
triplet_ptr   = F.pad(torch.cumsum(num_triplets, dim=0), (1, 0), value=0)
n_triplets    = int(triplet_ptr[-1])

triplet_batch = torch.arange(len(N)).repeat_interleave(num_triplets)
local_t       = torch.arange(n_triplets) - triplet_ptr[triplet_batch]
N_t           = N[triplet_batch]
offset        = ptr[triplet_batch]

node_a = local_t // (N_t * N_t) + offset
node_b = (local_t // N_t) % N_t + offset
node_c = local_t % N_t          + offset
```

Functions that build graph data return tuples, not dataclass instances.

## Batched cartesian product helpers

`batched_cartesian_product_2` and `batched_cartesian_product_3` in
`src/pycrystalgen/graph/scatter.py` implement the mixed-radix pattern above as reusable
helpers. Use them instead of re-implementing the pattern inline.

```python
from pycrystalgen.graph.scatter import batched_cartesian_product_2, batched_cartesian_product_3

# pairs — each structure can have different sizes
indices, batch, ptr = batched_cartesian_product_2(sizes_a, sizes_b)
# indices: (total, 2),  indices[:, 0] = first dim,  indices[:, 1] = second dim

# triplets
indices, batch, ptr = batched_cartesian_product_3(sizes_a, sizes_b, sizes_c)
# indices: (total, 3)
```

Indices are 0-based within each structure. To get centered shift vectors (e.g. -n..+n),
subtract the offset after: `shifts = indices - n[batch]`.
Device is derived from `sizes_a` internally — no need to pass it explicitly.

## Scatter aggregation

Use `scatter_sum` and `scatter_mean` from `src/pycrystalgen/graph/scatter.py` to aggregate
tensors by index without for loops (e.g. sum atom features per structure, aggregate edge
messages to nodes).

```python
from pycrystalgen.graph.scatter import scatter_sum, scatter_mean

energy        = scatter_sum(atom_energies, batch, n_struct)   # (n_struct,)
node_features = scatter_sum(messages, edge_index[1], n_atoms) # (n_atoms, F)
```

Both functions accept src of any shape `(N, *)` and a 1D `index` of shape `(N,)`.
`new_zeros` is used internally so device and dtype are inherited from `src` automatically.
Empty slots (no contributing elements) are zero. Built on `scatter_reduce_` with
`include_self=False`.

## Device consistency

Always pass `device=` explicitly when creating new tensors (`arange`, `zeros`, `ones`, `full`, etc.).
`torch.arange` and similar always create on CPU regardless of where other tensors live.

```python
device = structures.device()
edge_batch = torch.arange(len(N), device=device).repeat_interleave(num_edges)
local_e    = torch.arange(n_edges, device=device) - edge_ptr[edge_batch]
```

Derive the device from an existing tensor early in the function and thread it through every
new tensor creation. Never mix devices implicitly.

Never use `.to(device)` to silently fix a device mismatch on an input tensor — it hides
bugs in the caller and adds an unnecessary copy. The caller is responsible for creating
tensors on the correct device from the start.

## Computing ptr from counts

Always compute `ptr` (CSR-style index pointers) from a count tensor using `cumsum` + `F.pad`:

```python
ptr = F.pad(torch.cumsum(counts, dim=0), (1, 0), value=0)
```

The `(1, 0)` pad prepends a zero so `ptr[0] == 0` and `ptr[i+1] - ptr[i] == counts[i]` for all i.
Use this whenever building `ptr`, `edge_ptr`, or any other CSR pointer array.
