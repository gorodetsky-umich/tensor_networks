"""Class for benchmark details."""
from typing import List, Optional

import json
import numpy as np
from pydantic.dataclasses import dataclass

from pytens import Index, NodeName, Tensor, TensorNetwork

@dataclass
class Node:
    """Class for node representation in benchmarks"""
    name: NodeName
    indices: List[Index]
    value: Optional[str] = None

@dataclass
class Benchmark:
    """Class for benchmark data storage."""
    name: str
    nodes: List[Node]

    def to_network(self) -> TensorNetwork:
        """Convert a benchmark into a tensor network."""
        network = TensorNetwork()

        edges = {}
        for node in self.nodes:
            node_shape = tuple(i.size for i in node.indices)
            if node.value is None:
                node_value = np.random.randn(*node_shape)
            else:
                with open(node.value, "r", encoding="utf-8") as value_file:
                    node_value = np.array(json.load(value_file))

            network.add_node(node.name, Tensor(node_value, node.indices))

            for ind in node.indices:
                if ind.name not in edges:
                    edges[ind.name] = []

                edges[ind.name].append(node.name)

        for nodes in edges.values():
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i+1:]:
                    network.add_edge(n1, n2)

        return network

if __name__ == "__main__":
    with open("benchmarks/SVDinsTN/order_4_structure_1.json", encoding="utf-8") as f:
        benchmark_obj = json.load(f)
        b = Benchmark(**benchmark_obj)
        print(b.name)
        print(b.nodes)
