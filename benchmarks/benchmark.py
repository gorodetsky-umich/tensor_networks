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

    def to_network(self, normalize=False) -> TensorNetwork:
        """Convert a benchmark into a tensor network."""
        network = TensorNetwork()

        edges = {}
        for node in self.nodes:
            node_shape = tuple(i.size for i in node.indices)
            if node.value is None:
                node_value = np.random.randn(*node_shape)
            else:
                with open(node.value, "rb") as value_file:
                    node_value = np.load(value_file).astype(np.float32)
                    if normalize:
                        node_value = node_value / np.linalg.norm(node_value)

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

def start_from_ht(data_dir: str, source: str, name: str, eps: float):
    """Generate benchmark files that start from htucker results."""
    ht_data_dir = f"{data_dir}/{source}/{name}/"
    eps_version = "".join(format(eps,'.2f').split("."))
    benchmark_name = f"{name}_ht_eps_{eps_version}"
    edge_file = f"{ht_data_dir}/{eps_version}/htucker_nodes/edges.txt"

    # reconstruct the tree structure
    edges = {}
    with open(edge_file, "r") as edge_fd:
        lines = edge_fd.readlines()
        for line in lines:
            line = line.strip()
            if line:
                p, cs = line.split('\t')
                l, r = cs.split(',')
                edges[int(p)] = (int(l), int(r))

    nodes = []
    all_node_indices = set()
    for k, v in edges.items():
        all_node_indices.add(k)
        all_node_indices.add(v[0])
        all_node_indices.add(v[1])

    for i in all_node_indices:
        node_file = f"{ht_data_dir}/{eps_version}/htucker_nodes/node_{i}.npy"
        node_value = np.load(node_file)
        p = None
        for j, jc in edges.items():
            if i in jc:
                p = j
                break

        if i in edges and len(node_value.shape) == 2: # root node
            l, r = edges[i]
            l_size, r_size = node_value.shape
            indices = [Index(f"s_{i}_{l}", l_size), Index(f"s_{i}_{r}", r_size)]
        elif i in edges and len(node_value.shape) == 3: # internal nodes
            l, r = edges[i]
            l_size, r_size, p_size = node_value.shape
            indices = [Index(f"s_{i}_{l}", l_size), Index(f"s_{i}_{r}", r_size), Index(f"s_{p}_{i}", p_size)]
        elif i not in edges and len(node_value.shape) == 2: # leaf nodes
            c_size, p_size = node_value.shape
            indices = [Index(f"s_{i}", c_size), Index(f"s_{p}_{i}", p_size)]
        else:
            raise RuntimeError("node", i, "does not belong to any valid category")

        node = Node(f"G{i}", indices, node_file)
        nodes.append(node)

    benchmark = Benchmark(benchmark_name, source, nodes)
    with open(f"benchmarks/{source}/{name}/ht_{eps_version}.json", "w+") as json_file:
        json_str = RootModel[Benchmark](benchmark).model_dump_json(indent=4)
        json_file.write(json_str)


if __name__ == "__main__":
    """Main function that surpresses errors"""
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

    # Script for creating benchmark files from SVDinsTN dataset
    # import cv2
    # imgs = []
    # name = "truck"
    # for i, f in enumerate(glob.glob(f"/Users/zhgguo/Downloads/{name}_rectified/*.png")):
    #     if i % 17 >= 9:
    #         continue

    #     if i // 17 >= 9:
    #         break

    #     img = cv2.imread(f)
    #     img = cv2.resize(img, (40, 60))
    #     imgs.append(img)

    # stacked_x = np.stack(imgs, axis=0).reshape(9, 9, 40, 60, 3).transpose(2,3,4,0,1)
    # convert_single_value_to_benchmark(name, "SVDinsTN", data=stacked_x)

    # Script for creating benchmark files from tnGPS dataset
    # import cv2
    # for i, f in enumerate(glob.glob("/Users/zhgguo/Downloads/BSDS300/images/test/*.jpg")[:10]):
    #     img = cv2.imread(f)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img = cv2.resize(img, (256, 256))
    #     name = f"bsd_test_{i}"

    # import time
    # x = np.load("/Users/zhgguo/Documents/projects/tensor_networks/data/BigEarthNet-v1_0_all/stack_all_test_4/data.npy")
    # y = x.reshape(3*1122, -1)
    # start = time.time()
    # # q, r = np.linalg.qr(y)
    # u, s, v = np.linalg.svd(y, False, True)
    # # u = q @ u
    # print("svd time", time.time() - start)
    # print(s.shape)

    source = "BigEarthNet-v1_0_all"
    eps = 0.1
    for name in os.listdir(f"data/{source}"):
        if name == ".DS_Store":
            continue

        start_from_ht("data", source, name, eps)