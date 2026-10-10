---
name: port-model
description: Port a published machine learning model (interatomic potential, property predictor, GNN on crystals or molecules) from its GitHub project into torchms as a rewrite in src/torchms/models/<name>/, loading the official pretrained weights and tested against the original code. Use this whenever the user asks to port, rewrite, reimplement, add, integrate or "bring in" an existing model or its pretrained weights into torchms — e.g. "port SevenNet", "add MatterSim to the package", "can we load the eSEN checkpoint", "rewrite DPA-3 with our graphs" — even if they only give a paper, a model name or a Hugging Face link and never say "port".
---

# Porting a model into torchms

A port is a rewrite of the inference of a published model with torch and torchms alone,
which loads the official weights and gives the same numbers as the original code. The
original package never becomes a dependency: it is run once, outside of the project, to
produce reference values the tests compare against.

Six ports exist already, each in `src/torchms/models/<name>/`: orb-v3, MACE, M3GNet, CHGNet
and ALIGNN, which predict an energy, and CDVAE, a generative model with two networks and two
graphs. They are the worked examples of everything below. A new port is a new directory
there; what it shares with other models goes to `src/torchms/models/layers/`.

Follow the steps in order. The first three are reading, not writing: most of the bad ports
come from writing `model.py` before knowing which weights exist and what torchms already
has.

## 1. Start from the GitHub project

The source of truth is the code of the official repository, not the paper: papers leave out
the constants, the clamps and the bugs the weights were trained with.

- If the user gave a paper, a name or a link to weights, find the official repository
  (README of the weights, "code" link of the paper) and say which one you took. If two
  repositories claim the model (an original and a maintained fork, a framework which
  re-implements it), ask which one.
- Clone it shallow into the scratchpad, in a directory of its own. It is untrusted: read it,
  do not run it from the environment of the project. Keep your own scratch scripts in
  another directory and never name one after a module of the standard library
  (`inspect.py`, `numbers.py`): Python imports it in place of the real one and torch fails
  to import with an error which points elsewhere.
- Write down the tag or commit, the version of the package on PyPI, and the license. Copy the
  license file: it ships with the port. If the license forbids derived work or has no
  license at all, stop and tell the user before writing code.
- Trace one prediction through the code, from the structure to the output: graph
  construction (radius, number of neighbours, self images, how ties are cut), features,
  layers, read-out, scaling and reference energies. List the constants on the way.

## 2. Find the pretrained weights

Look in the repository first, then online, and do both even if the first one succeeds:
the newest weights are often not the ones in the README.

- In the repository: README, releases and their assets, and the function which downloads
  weights (search for `http`, `hf_hub_download`, `figshare`, `zenodo`, `urlretrieve`,
  `.ckpt`, `.pt`, `.pth`, `.model`).
- Online: Hugging Face (`https://huggingface.co/api/models?search=<name>`, then the file
  list of the repository), Zenodo, Figshare, the release page of the group.

Then, for the weights to port:

- If several exist, list them to the user with what differs (training data, size, task) and
  port the one upstream recommends unless told otherwise. One is enough for a first port.
- Pin a URL which cannot change: `https://huggingface.co/<repo>/resolve/<commit sha>/<file>`
  rather than `resolve/main`, a release asset, a Figshare file id. Download the file to the
  scratchpad and compute its SHA-256: both go to `config.yaml`.
- Check how the file is read. A state dict or a dict of tensors loads with
  `torch.load(..., weights_only=True)`. A pickled model object cannot be loaded this way
  and must not be unpickled normally: see `models/mace/reader.py`, which collects the
  tensors with a restricted unpickler.
- Read the hyperparameters stored with the weights (config json, `init_args`, hparams).
  They are what the rewrite is written for, and they are checked when weights are loaded.

If the authors publish no weights for the model which was asked for, stop and ask, with
what you found: weights of another model of the same repository (CDVAE only ships
predictors of a property), weights trained by a third party, or a port checked on random
weights. Third-party weights are a legitimate choice of the user, not yours; when they are
used, say whose they are in `config.yaml`, in the documentation and in the report, and that
their quality is not known.

Weights from anyone are untrusted files. Before reading a file `torch.load` refuses, list
the names its pickle would import, without running it (`pickletools.genops` on the
`data.pkl` of the archive). Checkpoints of pytorch-lightning hold objects of omegaconf next
to the state dict: read them with `read_pickle` of `models/estimator/reader.py`, which builds
tensors and containers and puts a placeholder in place of everything else.

## 3. Read how the existing ports are set up

Read these before designing anything; the conventions are in the code, not all in this file.

| Read | For |
|---|---|
| `src/torchms/models/estimator/pretrained.py` | `PretrainedModel`: `transform()`, `GRAPH_FIELDS`, `check_graphs`, `from_pretrained`, `load_checkpoint`, `load_parameters`; `element_table`, `element_index`, `different_arguments` |
| `src/torchms/models/estimator/config.py`, `weights.py` | `load_config(__file__)`, `download(model, name, cache)`, the SHA-256 check, the cache |
| the closest existing port: `model.py`, `config.yaml`, `__init__.py` | the shape of a port |
| its `tests/test_<name>.py` and `scripts/reference_<name>.py` | the shape of its tests |
| its page `docs/models/<name>.md` | what is reported about a port |
| `src/torchms/graph/transforms.py`, `graph.py` (docstrings) | what graphs and aggregations exist |
| `src/torchms/models/layers/` (`basis.py`, `dimenet.py`, `gemnet.py`) | bases, cutoffs, harmonics and whole networks which exist |

Closest port: angles from triplets of bonds → `m3gnet`, `chgnet`; a line graph → `alignn`;
pairs only with spherical harmonics → `mace`, `orb`; nearest neighbours → `orb`, `alignn`;
DimeNet or GemNet, an encoder and a decoder, several files of weights, a model which
generates → `models/cdvae`.

## 4. Decide the scope, and say it

A port predicts what the weights were trained to predict, for inference. Forces, stress,
magnetic moments, training, datasets and converters from ASE or pymatgen are left out
unless the user asks for them. State the scope before writing, and keep the list of what is
removed: it goes in the header of `model.py`.

Behaviours of the original which look like mistakes (an off-by-one atomic number, a mean
where a sum is expected, a clamp before an arc cosine) are kept: the weights were trained
with them. Note each one for the documentation.

## 5. Layout

```
src/torchms/models/<name>/
├── __init__.py     # docstring naming the upstream and its license; exports the model
├── model.py        # the network, a subclass of PretrainedModel
├── config.yaml     # every value: weights, graph, hyperparameters, upstream arguments
└── LICENSE         # license of the upstream project
scripts/reference_<name>.py     # runs the original, once, outside of the project
tests/data/<name>_reference.npz # what it wrote
tests/test_<name>.py
```

Export the model class from `src/torchms/models/__init__.py`.

What loads weights is shared, in `torchms.models.estimator`: `pretrained.py`
(`PretrainedModel`), `config.py`, `weights.py` and `reader.py`. Two things to get right:

- `config_files()` in `models/estimator/config.py` finds the configuration of every
  directory of `torchms.models`, and from them `weights.CHECKPOINTS`. Two places list the
  models by name and fail when one is added: its doctest, and
  `test_checkpoints_are_read_from_the_configurations` in `tests/test_pretrained.py`.
- `torchms.models.estimator` also holds `EnergyEstimator`, which uses orb-v3. Its module
  imports the model inside the function which needs it; keep it so, or importing a model
  becomes a circular import.

A network which another model could reuse (an encoder, a decoder, a known architecture as
DimeNet++ or GemNet-T) goes to `models/layers/<name>.py`, not in the directory of the port.
A layer reads no `config.yaml`: it takes a frozen dataclass of its hyperparameters
(`DimeNetConfig`, `GemNetConfig`), which the port builds from its own `config.yaml`
(`ENCODER = DimeNetConfig(out_channels=LATENT_DIM, **CONFIG["encoder"])` in
`models/cdvae/model.py`). The names of its modules still follow upstream.

### config.yaml

`model.py` has the names, `config.yaml` has the values: no radius, width, number of layers
or URL is written in Python. Same sections as the existing files:

```yaml
default_weights: <weights-name>

checkpoints:
  <weights-name>:
    - url: https://...            # pinned, see step 2
      sha256: <64 hex digits>
      # filename: only if the last part of the URL is not a usable name

unused_parameters:               # prefixes of upstream parameters the port does not have
  - heads.forces.

graph:
  radius: &radius 5.0            # Å

model:
  latent_dim: &latent_dim 64     # one comment per value: what it is

upstream:                        # arguments stored with the weights, checked at loading
  model_args:
    cutoff: *radius
    units: *latent_dim
```

Use YAML anchors so that a value shared by the graph, the model and the upstream arguments
is written once. `model.py` reads it at import, `CONFIG = load_config(__file__)`, into
module constants (`RADIUS: float = CONFIG["graph"]["radius"]`).

## 6. Writing model.py

Start the file with the attribution header of the existing ports: derived from which
project, copyright, license, then "Changes:" with what was removed and replaced.

**Name modules and parameters as upstream does**, so that the official state dict loads
with `load_parameters` (strict) and nothing is renamed by hand. Heads which are left out
are listed in `unused_parameters`. Check the arguments stored with the weights against
`upstream` with `different_arguments` and raise `InvalidWeights`: loading weights of
another variant must fail loudly instead of predicting nonsense.

When a constant of the port is a buffer the weights do not have, or a buffer of upstream
has another name in a module reused from torchms, do not write the exception in Python:
list it in `config.yaml` (`upstream.constant_buffers`, `upstream.renamed_parameters`, see
`models/cdvae`) so that the loading stays strict for everything else.

**The model does not build its graph.** Building a graph in `forward` hides its cost,
repeats it at every call and prevents doing it in the workers of a loader or once for a
dataset. The class gives the transform instead:

```python
GRAPH_FIELDS = ("graph", "triplets")

@classmethod
def transform(cls) -> Transform:
    return Compose(
        [AddRadiusGraph(RADIUS, self_images=True), AddTriplets(radius=THREE_BODY_RADIUS)]
    )
```

`forward(structures)` calls `self.check_graphs(structures)` and reads what is attached with
`attached_graph` and `attached_tuples`. If the graph of the original cannot be expressed
with the transforms of `torchms.graph.transforms`, add or extend a transform there, general
and tested on its own; do not build edges inside the model.

Three cases met so far:

- the graph of upstream comes from a library torchms cannot replace (CrystalNN of
  pymatgen): ask. The transform of the model then only completes a graph the user
  attaches, and the reference script exports the graph of upstream for the tests;
- the model has two networks with two graphs: one classmethod per graph
  (`transform()`, `decoder_transform()`), each with its check;
- the model moves atoms and needs a new graph at every step (a dynamics, a relaxation):
  the step applies the transform itself, on structures given without graph, and says so.

Read how upstream cuts its graph before choosing transforms: which images it looks at,
whether an atom is its own neighbour, how it makes the graph symmetric. These choices are
part of the weights (`KeepNearImages`, `MirrorEdges` exist for GemNet).

**No index in the arguments of a method.** Layers take the `Graph`, the `EdgeTuples` or
the `Structures` and the features, not `edge_index`, `senders`, `batch` or `ptr`. An index
passed around can be the one of another graph or another batch and nothing notices; the
object carries its own sizes and offsets. Compare the signatures of `M3GNetGraphConv` and
`ThreeBodyInteractions` in `models/m3gnet/model.py`.

**No dictionary as an argument or as a return value.** Tensors which travel together
(the features of the atoms, of the bonds and of the angles; the lengths, the angles and
their cutoffs) are the fields of a frozen dataclass, as `Features` and `Geometry` in
`models/layers/line_graph_network.py`, or separate arguments when there are few. A
dictionary with string keys is not typed, a wrong key only fails when it is read, and the
signature no longer says what the layer needs. Upstream code often passes dictionaries of
tensors around: do not carry them over. A dictionary stays right for what is a mapping by
nature: a state dict, the parsed `config.yaml`.

**No scatter by hand.** No `scatter`, `scatter_add_`, `index_add_`, `bincount` or fancy
indexing with an index in a model. Use the methods which name what is done:

| From → to | Method |
|---|---|
| atoms → edges | `graph.unpool_senders(x)`, `graph.unpool_receivers(x)` |
| edges → atoms | `graph.aggregate_edges_to_nodes(x, reduce, to="receiver" \| "sender")` |
| atoms → structures | `graph.aggregate_nodes_to_structures(x, reduce)` |
| edges → structures | `graph.aggregate_edges_to_structures(x, reduce)` |
| edges → triplets, pairs, quadruplets | `tuples.unpool_edges(x, position)` |
| triplets, pairs, quadruplets → edges | `tuples.aggregate_to_edges(x, reduce, position)` |
| a line graph as a graph | `tuples.as_graph()` |
| vectors, distances, angles | `edges_vectors(graph)`, `triplet_cosines`, `pair_cosines`, `dihedral_angles` |

Mind the direction: upstream "source/target", "i/j", "center/neighbour" must be mapped to
sender and receiver once, at the top, and checked on a graph. If an aggregation is missing
(a softmax over the neighbours, a reduction by element), add it as a method of `Graph` or
`EdgeTuples` with a doctest, then call it. Older modules of the library still call
`scatter` directly; do not take them as a model.

**Reuse before writing.** Before writing a basis, a cutoff, an MLP or a table of elements,
look for it: `torchms.models.layers` (`DimeNetPlusPlus`, `GemNetT`), its `basis` (`BesselBasis`, `SphericalBesselBasis`,
`GaussianBasis`, `AngleBasis`, `polynomial_cutoff`, `spherical_harmonics`, `legendre`,
`gaunt_coefficients`, `mlp`), `torchms.models.estimator.pretrained` (`element_table`,
`element_index`), `torchms.data.constants`, `torchms.graph`. When the original differs from
what exists by a constant or a normalisation, give the existing function an argument rather
than a twin. A new function several models could use goes to `models/layers/basis.py` or
`torchms.graph` with its own test, not in the port.

**No new library.** `pyproject.toml` is not touched. What the original takes from e3nn,
DGL, PyG, sympy, ASE or pymatgen is rewritten with torch and numpy. The models and the
layers do not import scipy: it is for a reference script or a test. If this is out of reach
for a part of the model, say so and port the variant which does not need it, as was done
for the small MACE.

The rules of `CLAUDE.md` apply as everywhere: no Python loop over structures, `device=`
on every new tensor, no `.to(device)` on an input.

## 7. Reference values and tests

The test of a port compares it with the original code. As the original is not installed,
the comparison is in two parts.

**`scripts/reference_<name>.py`** runs the original once, in an isolated environment, and
writes `tests/data/<name>_reference.npz`. Its docstring gives the exact command, with
pinned versions:

```bash
uv run --isolated --no-project --python 3.12 --with <package>==<version> \
    python scripts/reference_<name>.py
```

Build this environment early, before writing the port: it is the riskiest step for an old
project. Take the versions of its `env.yml` or `requirements.txt` (Python, torch, PyG),
with `--extra-index-url https://download.pytorch.org/whl/cpu` and `--find-links` of the
wheels of PyG for this torch, and pin the dependencies of dependencies which break at
import, one error after the other. A project which is not on PyPI is cloned at a fixed
commit, and its path given to the script: the docstring has both commands. Run upstream as
it is; the only code of the script around it is what loads the inputs and the weights.

It reads the 40 structures of `tests/data/orb_reference.npz` (`lattice`, fractional `x`,
`z`, `num_atoms`), the same for every port, and loads the same weights file as the port.
Save the prediction in float64 and also what locates a difference: number of edges of each
structure, number of angles or pairs, a sum of cosines, constants computed by the original
(roots, tables). Keep the file small, it is committed.

**`tests/test_<name>.py`**, after an existing one:

- with random weights (always run): shape and dtype of the output, a batch equals its
  structures alone, the invariances the model has (`AtomPermutation`, `LatticeRotation`,
  `AtomTranslation`, `LatticeBasisChange`), the energy of a supercell for an extensive
  output, an unknown element, `load_pretrained` refusing other arguments, missing and
  unknown parameters, structures without graph;
- pinned checkpoints: `https://` URL, 64 digits of SHA-256, paths in the cache;
- with the pretrained weights (skipped when they are not in the cache): the graphs match
  the reference exactly, then the predictions match.

The difference with the original is expected **under 1e-5** (per atom for an energy),
which is the rounding of float32. For outputs which are not of order 1 (scores of 30,
latent values of 7) it is 1e-5 of the largest expected value. For a model with several
stages, save and compare the output of each stage, each computed from the reference of the
stage before: the encoder, each head, the decoder, a few steps of a dynamics with its
starting point and its noise recorded from upstream. What is random is never compared
through a seed. Download the weights and run the pretrained tests
yourself; a port whose comparison was skipped is not tested. If the difference is larger:

1. compare the graphs first (edges per structure). Most differences are a graph: radius
   strict or not, self images, neighbours at the same distance cut in another order;
2. then bisect the network: extend the reference script to save an intermediate (basis,
   embedding, output of the first layer) and compare, until the first one which differs;
3. run the original in double precision if it can, to tell rounding from a bug.

Do not loosen the tolerance until the cause is found, and prove the cause instead of
naming it: a guess about the graph is checked by a run of upstream in the scratchpad with
the one thing changed, a guess about rounding by a run of upstream in double precision
(the port in float32 should be as near to it as upstream in float32 is).

A cause which belongs to the original and cannot be reproduced is kept as a tight test on
the structures without it, and written in the documentation with numbers. The usual one is
a limit of neighbours with neighbours at the same distance: in a crystal several
neighbours are, and an atom is always as far from its image at +v as from the one at -v.
The test finds these structures itself, from the graph of the port (the last neighbour
kept and the first one left out at the same length, see
`neighbours_at_the_same_distance` in `tests/test_cdvae.py`), asserts that enough
structures remain, and still checks loosely the other ones.

Tests with random weights need inputs the network can read: random weights which decode a
cell give a flat one. Set the buffers which scale an output to plausible values in the
fixture.

Tests have a timeout of 3 s by default: mark the slow ones
(`pytestmark = pytest.mark.timeout(30)`) rather than shrinking what they check.

## 8. Documentation and checks

- A page `docs/models/<name>.md`, as the existing ones: the table of facts (class, what it
  predicts, source and version, license, paper, weights, transform, largest difference),
  a snippet, the comparison with the original (version compared, largest and mean
  difference), behaviours kept, limits. Add it to the nav of `mkdocs.yml` and its rows to
  the two tables of `docs/models/index.md`. A new layer gets its own page too.
- A transform added to `torchms.graph` gets its row in the table of
  `docs/guide/graphs.md`.
- `docs/reference/models.md`: the `:::` entry of the new module. `docs/guide/models.md` if
  the model is used another way than the ones shown there.
- Run and fix: `uv run pytest`, `uv run pyright src`, `uv run black .`,
  `uv run pylint src`, `uv run poe docs-test`,
  `uv run --group docs mkdocs build --strict`.

## 9. Report

End with what the user needs to trust the port: repository and commit, weights and where
they come from, scope and what was left out, the measured differences with the original
(largest and mean, on how many structures, with which version), what was added to
`torchms.graph` or `models/layers/basis.py`, and anything which could not be reproduced. Say
plainly if a test was skipped or a tolerance is above 1e-5, and why.
