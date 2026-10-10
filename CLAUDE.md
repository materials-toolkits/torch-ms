# torchms — Claude Code notes

## Python environment

Always use `uv run` to run Python. Never bare `python3`.

## Batched graph indexing pattern

Variable-size batched tensors follow the PyG-style convention used in `Structures`
(`src/torchms/data/batching.py`): flat concatenated tensors with `ptr` (CSR-style
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

Functions that build graph data return tuples, not dataclass instances: the indices, their
structure and their pointers, as `triplets` and `edge_pairs` in `src/torchms/graph/line_graph.py`.

## Batched cartesian product helpers

`batched_cartesian_product_2` and `batched_cartesian_product_3` in
`src/torchms/graph/sparse.py` implement the mixed-radix pattern above as reusable
helpers. Use them instead of re-implementing the pattern inline. They take **CSR pointers**,
not sizes, and return **global** indices.

```python
from torchms.graph.sparse import batched_cartesian_product_2, batched_cartesian_product_3

# pairs (a, b) with a in [ptr_a[i], ptr_a[i+1]) and b in [ptr_b[i], ptr_b[i+1])
indices, batch, ptr = batched_cartesian_product_2(ptr_a, ptr_b)
# indices: (total, 2),  indices[:, 0] = first range,  indices[:, 1] = second range

# triplets
indices, batch, ptr = batched_cartesian_product_3(ptr_a, ptr_b, ptr_c)
# indices: (total, 3)
```

For ranges which start at 0 in every structure (shifts -n..+n, say), build the pointers
from the sizes (see "Computing ptr from counts") and subtract the offset of each structure
after: `local = indices - ptr_a[batch].unsqueeze(1)`. Device is derived from the pointers
internally. `batched_join` in the same module pairs the elements of two lists which share
a key, which is how triplets and line graphs are built.

## Aggregation

Aggregate with the methods of `Graph` and `EdgeTuples` (`src/torchms/graph/graph.py`), which
name what is done and carry their own sizes:

| From → to | Method |
|---|---|
| atoms → edges | `graph.unpool_senders(x)`, `graph.unpool_receivers(x)` |
| structures → atoms | `graph.unpool_structures(x)` |
| edges → atoms | `graph.aggregate_edges_to_nodes(x, reduce, to="receiver" \| "sender")` |
| atoms → structures | `graph.aggregate_nodes_to_structures(x, reduce)` |
| edges → structures | `graph.aggregate_edges_to_structures(x, reduce)` |
| edges → triplets, pairs, quadruplets | `tuples.unpool_edges(x, position)` |
| triplets, pairs, quadruplets → edges | `tuples.aggregate_to_edges(x, reduce, position)` |

They are built on `scatter(src, index, n_out, reduce)` in `src/torchms/graph/sparse.py`
(`reduce` is "sum", "mean", "prod", "amax", "amin" or "hash"; empty slots are zero; device
and dtype come from `src`), with `count_by(index, n_out)` for counts. Call `scatter`
directly only in the code which builds graphs, where there is no `Graph` yet. If an
aggregation is missing, add a method to `Graph` or `EdgeTuples` with a doctest.

```python
energy        = graph.aggregate_nodes_to_structures(atom_energies)   # (n_struct,)
node_features = graph.aggregate_edges_to_nodes(messages)             # (n_atoms, F)
```

## Models and layers

Everything in `src/torchms/models/` follows the same rules, pretrained models, layers and
networks to train alike. `.claude/skills/port-model/SKILL.md` has their reasons and the
procedure to port a model.

- **A model does not build its graph.** It reads the graph and the tuples of edges attached
  to the `Structures` and gives the transform which attaches them: a classmethod
  `Model.transform()` for a pretrained model, which takes its `Config` and uses the one of
  `config.yaml` by default, a method for a network to train (`LineGraphNetwork`). It refuses
  structures without them and names the transform.
- **No index in the arguments of a method.** A layer takes the `Graph`, the `EdgeTuples` or
  the `Structures` and the features, not `edge_index`, `senders`, `batch` or `ptr`.
- **No scatter by hand** in a model: no `scatter`, `scatter_add_`, `index_add_`, `bincount`
  or indexing with a graph index. Use the methods of the table above.
- **No dictionary as an argument or as a return value.** Tensors which travel together are
  the fields of a frozen dataclass (`Features` and `Geometry` in
  `models/layers/line_graph_network.py`, `dimenet.Config`), or separate arguments when there
  are few. String keys are not typed, a wrong key only fails when it is read, and the
  signature no longer says what a function needs. A dictionary is fine for what is a mapping
  by nature: a state dict, a parsed `config.yaml`, `Structures.properties`.
- **No hyperparameter written in the code, and none in a global variable.** Every value of
  a pretrained model is in the `config.yaml` of its directory. `config.py`, next to it, has
  a frozen dataclass `Config` (a `YamlConfig`, see `models/estimator/config.py`) with one
  field per section of the file and one dataclass per section; `Config.load()` reads the
  file once, strictly. The model takes a `Config` as its optional argument and gives it, or
  the section they need, to its sub-modules: no module does `CONFIG = load_config(...)` or
  keeps `RADIUS = ...`. What is derived from the values is a method of the `Config`.
- **The class is `Config` in every module**, named by its module at the call site:
  `cdvae.Config`, `m3gnet.Config`, `gemnet.Config`, never `CdvaeConfig`. A network of
  `models/layers` takes the `Config` of its module, a plain frozen dataclass: a shared layer
  has no default taken from one model, a network to train (`line_graph_network.Config`) has
  its own defaults. A number which is a limit of the implementation is one named constant
  with a comment saying so (`MAX_DEGREE` in `models/layers/basis.py`).
- **Reuse `torchms.models.layers` and `torchms.graph`** before writing a basis, a cutoff or
  a graph. What several models could use goes there, with its own test.
- **No new dependency.** scipy is used by the convex hull and the calibration of the
  estimator, not by the models and the layers.

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
