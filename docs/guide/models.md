# Using models

`torchms.models` has supervised models, which predict the energy of a structure,
generative models, which generate structures, and layers to build your own. This page shows how to use them; the [Models](../models/index.md) section
has a page for each one.

| | | Models |
|---|---|---|
| Supervised | total energy, pretrained | [orb-v3](../models/orb.md), [MACE](../models/mace.md), [M3GNet](../models/m3gnet.md), [CHGNet](../models/chgnet.md) |
| | formation energy, pretrained | [ALIGNN](../models/alignn.md) |
| Generative | crystals, pretrained | [CDVAE](../models/cdvae.md), [DiffCSP](../models/diffcsp.md), [MatterGen](../models/mattergen.md) |
| Layers | bases and networks | [bases](../models/bases.md), [DimeNet++](../models/dimenet.md), [GemNet-T](../models/gemnet.md), [CSPNet](../models/cspnet.md) |
| | networks of the energy, to train | [`LineGraphNetwork`](../models/line-graph-network.md), [`EquivariantNetwork`](../models/equivariant-network.md) |

To know how far a structure is from the convex hull, use the
[energy estimator](energy.md), which wraps orb-v3: it is one call.

## A supervised model

```{.python notest}
from torchms.data import load_dataset
from torchms.models import M3GNet

dataset = load_dataset("mp-20", "test", transform=M3GNet.transform())
model = M3GNet.from_pretrained(device="cuda")

energy = model(dataset[:32].to("cuda"))     # (32,), total energy in eV
```

`from_pretrained` downloads the weights on the first call, to `~/.cache/torchms`
(`$TORCHMS_CACHE`), checks them against a SHA-256 written in the library, and gives a model
in evaluation mode, without gradients. A model can have several weights, listed on its page:

```{.python notest}
model = OrbV3.from_pretrained("orb-v3-conservative-inf-omat", device="cuda")
```

The pretrained models are rewrites: they predict what the original code predicts, without
its package. Forces, stress and training are not included.

## Configuration of a model

Every value of a pretrained model (the radii of its graph, its hyperparameters, the name
of its weights) is in the `config.yaml` of its directory. The module of the model has a
frozen dataclass `Config` with the same fields, typed, which the model is built from:

```{.python notest}
from torchms.models import m3gnet

config = m3gnet.Config.load()               # the values of config.yaml, read once
config.graph.radius                         # 5.0
config.model.num_blocks                     # 3

model = m3gnet.M3GNet()                     # the same as m3gnet.M3GNet(config)
model.config is config                      # True
```

The class has the same name in every model, `orb.Config`, `cdvae.Config`,
`diffcsp.Config`, and nothing is kept in a global variable: a model and its layers read
the object they are given. To train another network than the published one, give another
configuration to the model and to its transform:

```{.python notest}
from dataclasses import replace

small = replace(config, model=replace(config.model, num_blocks=2))

model = m3gnet.M3GNet(small)
transform = m3gnet.M3GNet.transform(small)
```

The pretrained weights are the ones of `config.yaml`: a model with other sizes refuses
them. A file with a key the dataclass does not have, or without one it needs, is refused
when it is read.

## Graphs are attached by a transform

**A model does not build its graph.** It reads the graph and the tuples of edges attached
to the structures, and its class gives the transform which attaches them. Building a graph
is the expensive part of a prediction: as a transform it runs in the workers of a loader,
once per structure, or on the GPU for a whole batch, as you choose.

```{.python notest}
transform = M3GNet.transform()

dataset = load_dataset("mp-20", "test", transform=transform)   # when a structure is loaded
structures = transform(structures)                             # or on a batch
```

A model refuses structures without what it reads, and names the transform to apply. The
transform of each model is in the [table of the models](../models/index.md#graphs); see
[Graphs](graphs.md) for the transforms themselves.

[CDVAE](../models/cdvae.md) has two graphs: its encoder completes a graph of bonds you
attach, and its decoder builds its own at each step of the generation.

## A generative model

```{.python notest}
from torchms.models import CDVAE

model = CDVAE.from_pretrained(device="cuda")
structures = model.sample(64)               # 64 crystals, 5000 steps of dynamics
```

The weights of CDVAE were trained by a third party: read
[its page](../models/cdvae.md) before using what it generates.

[DiffCSP](../models/diffcsp.md) has two models. `DiffCSP` predicts a structure from
its composition: it is given the atomic numbers of the atoms and the number of atoms of
each structure. `DiffCSPGeneration` also generates the elements: it is only given the
numbers of atoms, which it can draw as the ones of MP-20.

```{.python notest}
import torch

from torchms.models import DiffCSP, DiffCSPGeneration

model = DiffCSP.from_pretrained(device="cuda")

z = torch.tensor([11, 17, 38, 22, 8, 8, 8], device="cuda")   # NaCl and SrTiO3
num_atoms = torch.tensor([2, 5], device="cuda")
structures = model.sample(z, num_atoms)     # 1000 steps of diffusion

generation = DiffCSPGeneration.from_pretrained(device="cuda")
structures = generation.sample(generation.sample_num_atoms(64))
```

[MatterGen](../models/mattergen.md) generates crystals of given numbers of atoms with
their elements, without condition. As DiffCSP, it builds the graph of its network at each
step of the sampling.

```{.python notest}
from torchms.models import MatterGen

model = MatterGen.from_pretrained(device="cuda")
structures = model.sample(model.sample_num_atoms(64))   # 1000 steps, 2000 calls of the network
```

`MatterGenChemicalSystem` generates them with given elements, one chemical system per crystal:

```{.python notest}
from torchms.models import MatterGenChemicalSystem

model = MatterGenChemicalSystem.from_pretrained(device="cuda")
guidance = model.guidance(["Li-O"] * 64)
structures = model.sample(model.sample_num_atoms(64), guidance=guidance)
```

`MatterGenSpaceGroup` and `MatterGenBandGap` are used the same way, with a space group or a
band gap in eV per crystal: `model.guidance([225] * 64)`, `model.guidance([3.0] * 64)`.
Both are followed, not imposed.

## Layers for your own model

`torchms.models.layers` has the bases of distances and of angles of the models, and two
networks which take their hyperparameters as arguments:

```python
from torchms.models.layers import BesselBasis, polynomial_cutoff

distance = torch.tensor([1.0, 3.0, 6.0])
features = BesselBasis(r_max=6.0, num_bases=8)(distance)       # (3, 8)
features = features * polynomial_cutoff(distance, 6.0, p=5).unsqueeze(-1)
```

As the models, a layer reads the graph of the structures and aggregates with its methods
(`graph.aggregate_edges_to_nodes`, `graph.aggregate_nodes_to_structures`): see
[Graphs](graphs.md) and the pages of the [layers](../models/index.md#layers).

### Networks of the energy to train

```{.python notest}
from torchms.data import load_dataset
from torchms.models.layers import (
    EquivariantNetwork,
    LineGraphNetwork,
    equivariant_network,
    line_graph_network,
)

structures = load_dataset("mp-20", "train")[:32]

model = LineGraphNetwork(line_graph_network.Config(order=2, radius=(5.0, 3.5)))
energy = model(model.transform()(structures))      # (32,)

model = EquivariantNetwork(equivariant_network.Config(correlation=3, radius=5.0))
energy = model(model.transform()(structures))      # (32,)
```

Both read the graph of their transform, as the pretrained models. It is a method of the
network, not of its class: the graph depends on its radius and, for `LineGraphNetwork`, on
its order.

Two networks written for this library, with terms on more than two atoms. They have no
pretrained weights and there is no training loop in the library. Their energy does not
depend on the description of the crystal and grows with the size of its cell, which is
checked in the tests with the transforms of [data augmentation](augmentation.md). Their
pages say what each one computes and
[which one to use](../models/line-graph-network.md#compared-with-equivariantnetwork).
