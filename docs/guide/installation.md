# Installation

TorchMS is a library: add it to the dependencies of your project, then import `torchms`. It
needs Python 3.10 or later and PyTorch 2.11 or later, and is installed from its git
repository.

=== "uv"

    ```bash
    uv add "torchms @ git+https://gitlab.univ-artois.fr/astrid.klipfel/torch-ms.git"
    ```

=== "pip"

    ```bash
    pip install "torchms @ git+https://gitlab.univ-artois.fr/astrid.klipfel/torch-ms.git"
    ```

Check the installation:

```python
import torch
from torchms.data import Structures

structures = Structures(
    lattice=torch.eye(3).unsqueeze(0) * 4.0,
    x=torch.zeros(1, 3),
    z=torch.tensor([29]),
    num_atoms=torch.tensor([1]),
)
print(structures)
```

## Loading from Hugging Face

The published datasets and the weights of the energy estimator are downloaded from Hugging
Face. This needs `huggingface_hub`, which comes with the `hub` extra:

=== "uv"

    ```bash
    uv add "torchms[hub] @ git+https://gitlab.univ-artois.fr/astrid.klipfel/torch-ms.git"
    ```

=== "pip"

    ```bash
    pip install "torchms[hub] @ git+https://gitlab.univ-artois.fr/astrid.klipfel/torch-ms.git"
    ```

Without it, the rest of the library works: `HDF5Dataset("file.hdf5")` reads a local file.

## A fixed version

The commands above install the last commit of `main`. For results that can be reproduced,
end the address with a tag, a branch or a commit:

```bash
uv add "torchms @ git+https://gitlab.univ-artois.fr/astrid.klipfel/torch-ms.git@3dd18de"
```

uv writes the commit in `uv.lock` in any case; `uv lock --upgrade-package torchms` moves to
the last one.

## GPU

TorchMS runs on the device of the tensors it is given, and installs nothing for the GPU by
itself: it uses the PyTorch of your environment. See the
[PyTorch instructions](https://pytorch.org/get-started/locally/) for a build matching your
CUDA version. The energy estimator runs on CPU but is much faster on a GPU: about
600 structures per second on a laptop GPU with 8 GB of memory.

## Command line

The library installs one command, `preprocessing`, which converts CIF files to the HDF5
format of the datasets (see [Preprocessing your own data](preprocessing.md)).

To change TorchMS itself, see [Development](../development.md).
