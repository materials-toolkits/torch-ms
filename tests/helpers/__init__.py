"""Helpers shared by the tests, importable as `helpers` (pytest pythonpath).

* paths: the directories of the tests and of their data;
* references: what an original code computed, recorded by
  tests/references/reference_*.py;
* weights: pretrained models, skipped when their weights are not in the cache;
* compare: tolerances against a reference;
* matching: the permutation between two orders of the same rows;
* gemnet: the graph of GemNet-T of an original code as torchms graphs;
* structures: the crystals of the tests of the models, random generators;
* models: running a model on structures without graph;
* graphs: the structures where the graph of an original is not defined.
"""
