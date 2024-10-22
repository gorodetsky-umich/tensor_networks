"""Class for benchmark details."""
from typing import List, Optional
import os
import glob
import json
import random

import numpy as np
from pydantic import RootModel
from pydantic.dataclasses import dataclass
import matplotlib.pyplot as plt

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
    source: str

    nodes: Optional[List[Node]] = None
    value_file: Optional[str] = None

    def to_network(self) -> TensorNetwork:
        """Convert a benchmark into a tensor network."""
        network = TensorNetwork()

        edges = {}
        for node in self.nodes:
            node_shape = tuple(i.size for i in node.indices)
            if node.value is None:
                node_value = np.random.randn(*node_shape)
            else:
                with open(node.value, "rb") as value_file:
                    node_value = np.load("data/" + value_file).astype(np.float64)

            network.add_node(node.name, Tensor(node_value, node.indices))

            for ind in node.indices:
                if ind.name not in edges:
                    edges[ind.name] = set()

                edges[ind.name].add(node.name)

        for nodes in edges.values():
            nodes = list(nodes)
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i+1:]:
                    network.add_edge(n1, n2)

        # network.draw()
        # plt.show()
        return network

def convert_single_value_to_benchmark(bench_name: str, bench_source: str, data_file: str = None, data: np.ndarray = None):
    """Convert single numpy data files into a benchmark JSON file."""
    benchmark = Benchmark(bench_name, bench_source)
    benchmark.nodes = []

    benchmark_dir = f"benchmarks/{bench_source}/{bench_name}"
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    if os.path.exists(f"{benchmark_dir}/data.npy"):
        os.remove(f"{benchmark_dir}/data.npy")
    
    data_dir = f"data/{bench_source}/{bench_name}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    new_data_file = f"{data_dir}/data.npy"
    if data_file is not None:
        os.rename(data_file, new_data_file)
    else:
        np.save(new_data_file, data)

    benchmark.value_file = new_data_file

    with open(new_data_file, "rb") as npy_file:
        value = np.load(npy_file)
        indices = [Index(f"I{i}", sz) for i, sz in enumerate(value.shape)]
        benchmark.nodes.append(Node("G0", indices, new_data_file))

    with open(f"{benchmark_dir}/original.json", "w+", encoding="utf-8") as json_file:
        json_str = RootModel[Benchmark](benchmark).model_dump_json(indent=4)
        json_file.write(json_str)

def stack_data_to_benchmark(iters: int, source: str, stack_size: int, all_data: List[str], reshape=None):
    """Stack several data files into one and save it as a new benchmark"""
    for i in range(iters):
        name = f"stack_{stack_size}_test_{i}"
        data = []
        for f in random.sample(all_data, k=stack_size):
            data.append(np.load(f))

        data = np.stack(data, axis=0).reshape(reshape)
        convert_single_value_to_benchmark(name, f"{source}_stack", data=data)

def all_data_to_benchmarks(source: str, all_data: List[str]):
    """Stack all data files into one, reshape it, and save it as a new benchmark"""
    source = f"{source}_all"
    data = []
    for f in all_data:
        data.append(np.load(f))

    data = np.stack(data, axis=0)
    # data: (18*11*17) * 120 * 120 * 12
    name = "stack_all_test_0"
    convert_single_value_to_benchmark(name, source, data=data)

    data1 = data.reshape(18, 11, 17, 120, 120, 12)
    name = "stack_all_test_1"
    convert_single_value_to_benchmark(name, source, data=data1)

    data2 = data.reshape(18, 11 * 17, 120, 120, 12)
    name = "stack_all_test_2"
    convert_single_value_to_benchmark(name, source, data=data2)

    data3 = data.reshape(18 * 17, 11, 120, 120, 12)
    name = "stack_all_test_3"
    convert_single_value_to_benchmark(name, source, data=data3)

    data4 = data.reshape(3, 6*17*11, 120, 120, 12)
    name = "stack_all_test_4"
    convert_single_value_to_benchmark(name, source, data=data4)

    data5 = data.reshape(18, 17*11, 120, 120, 12).transpose(1,2,3,4,0).reshape(17*11, 120, 120, -1)
    name = "stack_all_test_5"
    convert_single_value_to_benchmark(name, source, data=data5)

def start_from_tt(data_dir: str, name: str, eps: float):
    """Generate benchmark files that start from tensor train results."""
    benchmark_dir = f"{data_dir}/{name}/"
    eps_version = "".join(format(eps,'.2f').split("."))
    benchmark_name = f"{name}_tt_eps_{eps_version}"
    core_files = sorted(list(glob.glob(f"{benchmark_dir}/trainedCores/*{eps_version}*.npy")))
    core_nodes = []
    for i, core_file in enumerate(core_files):
        core_i = np.load(core_file)
        indices = [Index(2*i+j, ind) for j,ind in enumerate(core_i.shape)]
        core_node = Node(f"G{i}", indices, core_file)
        core_nodes.append(core_node)

    benchmark = Benchmark(benchmark_name, "transformed", core_nodes)
    with open(f"{benchmark_dir}/tt_{eps_version}.json", "w+", encoding="utf-8") as json_file:
        json_str = RootModel[Benchmark](benchmark).model_dump_json(indent=4)
        json_file.write(json_str)

def start_from_ht(data_dir: str, name: str, eps: float):
    """Generate benchmark files that start from htucker results."""
    benchmark_dir = f"{data_dir}/{name}/"
    eps_version = "".join(format(eps,'.2f').split("."))
    benchmark_name = f"{name}_ht_eps_{eps_version}"
    edge_file = f"{benchmark_dir}/htucker_textfile/eps_{eps}_edges.txt"

    # reconstruct the tree structure
    parent = {}
    with open(edge_file, "r") as edge_fd:
        lines = edge_fd.readlines()
        for line in lines:
            line = line.strip()
            if line:
                u, v = line.split('\t')
                parent[v] = u

    index_idx = 0
    # starting from node 0 and do the traversal
    def traverse(nodes, pindices, node_name):
        node_file = f"{benchmark_dir}/htucker_nodes/eps_{eps}_node_{node_name}.npy"
        node_value = np.load(node_file)
        indices = []
        for sz in node_value.shape:
            if pindices is not None:
                psizes = [i.size for i in pindices]
                if sz in psizes:
                    indices.append(pindices[psizes.index(sz)])
                    pindices.pop(psizes.index(sz))
                    continue

            nonlocal index_idx
            indices.append(Index(index_idx, sz))
            index_idx += 1

        node = Node(f"G{node_name}", indices, node_file)
        nodes.append(node)

        for u, v in parent.items():
            if v == node_name:
                indices = traverse(nodes, indices, u)

        return pindices

    nodes = []
    traverse(nodes, None, "0")

    benchmark = Benchmark(benchmark_name, "transformed", nodes)
    with open(f"{benchmark_dir}/ht_{eps_version}.json", "w+", encoding="utf-8") as json_file:
        json_str = RootModel[Benchmark](benchmark).model_dump_json(indent=4)
        json_file.write(json_str)


if __name__ == "__main__":
    # Uncomment for converting single cores into benchmarks
    # for f in glob.glob("data/SVDinsTN/*/*.npy"):
    #     name = f.split('/')[-2]
    #     source = f.split('/')[-3]
    #     convert_single_value_to_benchmark(name, source, data_file=f)

    # Uncomment for converting BigEarthNet cores into benchmarks (random sample 10)
    # for f in random.sample(glob.glob("data/BigEarthNet-v1_0/*.npy"), k=10):
    #     name = f.split('/')[-1][:-4]
    #     source = f.split('/')[-2]
    #     convert_single_value_to_benchmark(name, source, data_file=f)

    # Uncomment for stacking BigEarthNet cores into medium and large benchmarks
    # path_to_bigearth = "data/BigEarthNet-v1_0"
    # all_data = glob.glob(f"{path_to_bigearth}/*.npy")
    # # medium benchmarks: 120 x 120 x 12 x 18 (random sample 18)
    # stack_data_to_benchmark(10, "BigEarthNet-v1_0", 18, all_data, (18, 120, 120, 12))
    # # large benchmarks: 120 x 120 x 12 x 11 x 17 (random sample 11 x 17)
    # stack_data_to_benchmark(10, "BigEarthNet-v1_0", 11 * 17, all_data, (11, 17, 120, 120, 12))
    # # benchmarks with all data: several different reshapes
    # all_data_to_benchmarks("BigEarthNet-v1_0", all_data)
    

    # for f in os.listdir("benchmarks/SVDinsTN/"):
    #     if ".DS_Store" in f:
    #         continue
    #     start_from_tt("benchmarks/SVDinsTN/", f, 0.1)
    # for f in glob.glob("benchmarks/SVDinsTN/order_4_structure_1/breakdown.json"):
    #     with open(f, "r") as fp:
    #         b = Benchmark(**json.load(fp))
    #         net = b.to_network()
    #         net_value = net.contract().value
    #         path_segments = f.split('/')
    #         path = "/".join(path_segments[:-1])
    #         np.save(path+"/data.npy", net_value)
    # with open("benchmarks/SVDinsTN/order_4_structure_1.json", encoding="utf-8") as f:
    #     benchmark_obj = json.load(f)
    #     b = Benchmark(**benchmark_obj)
    #     print(b.name)
    #     print(b.nodes)
