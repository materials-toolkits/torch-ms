"""MatterGen, a diffusion model which generates crystals, with pretrained weights.

Derived from MatterGen of Microsoft (MIT license, see LICENSE in this
directory): model.py has its denoiser, its three diffusions, of the lattice,
of the fractional coordinates and of the elements, and its sampling without
condition (MatterGen). Its models fine-tuned on a property guide the sampling
towards a chemical system (MatterGenChemicalSystem, in chemical_system.py), a
space group (MatterGenSpaceGroup, in space_group.py), a band gap
(MatterGenBandGap, in band_gap.py), a magnetic density
(MatterGenMagneticDensity, in magnetic_density.py), a bulk modulus
(MatterGenBulkModulus, in bulk_modulus.py), or a chemical system and the energy
above the hull together (MatterGenChemicalSystemEnergyAboveHull);
properties.py has the embeddings of their properties, diffusion.py the
arithmetic of a step. Their network, GemNet-T,
is in torchms.models.layers. The weights of the seven models are listed in
config.yaml, with their hyperparameters, which config.py reads into a Config.
"""

from torchms.models.mattergen.config import Config
from torchms.models.mattergen.band_gap import MatterGenBandGap
from torchms.models.mattergen.bulk_modulus import MatterGenBulkModulus
from torchms.models.mattergen.chemical_system import MatterGenChemicalSystem
from torchms.models.mattergen.chemical_system_energy_above_hull import (
    MatterGenChemicalSystemEnergyAboveHull,
)
from torchms.models.mattergen.diffusion import Denoised, Guidance, StepNoise
from torchms.models.mattergen.magnetic_density import MatterGenMagneticDensity
from torchms.models.mattergen.model import InvalidWeights, MatterGen
from torchms.models.mattergen.space_group import MatterGenSpaceGroup

__all__ = [
    "Config",
    "Denoised",
    "Guidance",
    "InvalidWeights",
    "MatterGen",
    "MatterGenBandGap",
    "MatterGenBulkModulus",
    "MatterGenChemicalSystem",
    "MatterGenChemicalSystemEnergyAboveHull",
    "MatterGenMagneticDensity",
    "MatterGenSpaceGroup",
    "StepNoise",
]
