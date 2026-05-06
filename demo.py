from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.cutoff import cutoff

structure = Structures.zincblende()

print(structure)

graph = cutoff(structure, 3.0)

print(graph.edges_distances())
print(graph.structures.x)
print(graph.edge_shift.t())
print(graph.edge_index)
