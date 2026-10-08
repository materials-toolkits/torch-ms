#!/usr/bin/env python
"""Test the rate of hash collision for lematerial dataset using color_refinement.

Computes color_refinement() hashes for all structures in the lematerial dataset
and reports:
- Total number of structures
- Number of unique hashes (collision count)
- Collision statistics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import tqdm
from collections import Counter

from torchms.data import load_dataset
from torchms.graph.coloring import color_refinement
from torchms.graph.atomic_radius import atomic_radius
from torch.utils.data import DataLoader

from torchms.data.batching import Structures
from torchms.graph.atomic_radius import atomic_radius

lematerial = load_dataset("lematerial", "unique")

loader = DataLoader(lematerial, batch_size=128, shuffle=True)

total_hash = []

for batch in tqdm.tqdm(loader):
    batch = batch.to("cuda")
    graph = atomic_radius(batch)
    hash = color_refinement(batch,graph).tolist()
    hash = list(map(lambda x:"".join("{:08X}".format(y) for y in x),hash))
    total_hash.extend(hash)

collision = len(total_hash)-len(set(total_hash))
rate = collision/len(total_hash)

print(f"collision rate: {rate*100:.2f}%")
print(f"collision: {collision}")


