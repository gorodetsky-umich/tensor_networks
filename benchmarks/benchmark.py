"""Class for benchmark details."""

from typing import List, Optional
import os
import glob
import random

import numpy as np
from pydantic import RootModel
from pydantic.dataclasses import dataclass
import networkx as nx
import h5py

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
                for n2 in nodes[i + 1 :]:
                    network.add_edge(n1, n2)

        # network.draw()
        # plt.show()
        return network


def convert_single_value_to_benchmark(
    bench_name: str, bench_source: str, data_file: str = None, data: np.ndarray = None
):
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


def stack_data_to_benchmark(
    iters: int, source: str, stack_size: int, all_data: List[str], reshape=None
):
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

    data4 = data.reshape(3, 6 * 17 * 11, 120, 120, 12)
    name = "stack_all_test_4"
    convert_single_value_to_benchmark(name, source, data=data4)

    data5 = (
        data.reshape(18, 17 * 11, 120, 120, 12)
        .transpose(1, 2, 3, 4, 0)
        .reshape(17 * 11, 120, 120, -1)
    )
    name = "stack_all_test_5"
    convert_single_value_to_benchmark(name, source, data=data5)


def start_from_tt(data_dir: str, name: str, eps: float):
    """Generate benchmark files that start from tensor train results."""
    benchmark_dir = f"{data_dir}/{name}/"
    eps_version = "".join(format(eps, ".2f").split("."))
    benchmark_name = f"{name}_tt_eps_{eps_version}"
    core_files = sorted(
        list(glob.glob(f"{benchmark_dir}/trainedCores/*{eps_version}*.npy"))
    )
    core_nodes = []
    for i, core_file in enumerate(core_files):
        core_i = np.load(core_file)
        indices = [Index(2 * i + j, ind) for j, ind in enumerate(core_i.shape)]
        core_node = Node(f"G{i}", indices, core_file)
        core_nodes.append(core_node)

    benchmark = Benchmark(benchmark_name, "transformed", core_nodes)
    with open(
        f"{benchmark_dir}/tt_{eps_version}.json", "w+", encoding="utf-8"
    ) as json_file:
        json_str = RootModel[Benchmark](benchmark).model_dump_json(indent=4)
        json_file.write(json_str)


def start_from_ht(data_dir: str, source: str, name: str, eps: float):
    """Generate benchmark files that start from htucker results."""
    ht_data_dir = f"{data_dir}/{source}/{name}/"
    eps_version = "".join(format(eps, ".2f").split("."))
    benchmark_name = f"{name}_ht_eps_{eps_version}"
    edge_file = f"{ht_data_dir}/{eps_version}/htucker_nodes/edges.txt"

    # reconstruct the tree structure
    edges = {}
    with open(edge_file, "r", encoding="utf-8") as edge_fd:
        lines = edge_fd.readlines()
        for line in lines:
            line = line.strip()
            if line:
                p, cs = line.split("\t")
                l, r = cs.split(",")
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

        if i in edges and len(node_value.shape) == 2:  # root node
            l, r = edges[i]
            l_size, r_size = node_value.shape
            indices = [Index(f"s_{i}_{l}", l_size), Index(f"s_{i}_{r}", r_size)]
        elif i in edges and len(node_value.shape) == 3:  # internal nodes
            l, r = edges[i]
            l_size, r_size, p_size = node_value.shape
            indices = [
                Index(f"s_{i}_{l}", l_size),
                Index(f"s_{i}_{r}", r_size),
                Index(f"s_{p}_{i}", p_size),
            ]
        elif i not in edges and len(node_value.shape) == 2:  # leaf nodes
            c_size, p_size = node_value.shape
            indices = [Index(f"s_{i}", c_size), Index(f"s_{p}_{i}", p_size)]
        else:
            raise RuntimeError("node", i, "does not belong to any valid category")

        node = Node(f"G{i}", indices, node_file)
        nodes.append(node)

    benchmark = Benchmark(benchmark_name, source, nodes)
    with open(
        f"benchmarks/{source}/{name}/ht_{eps_version}.json", "w+", encoding="utf-8"
    ) as json_file:
        json_str = RootModel[Benchmark](benchmark).model_dump_json(indent=4)
        json_file.write(json_str)


def gen_fctn(num_of_nodes: int, bond_size: int, free_bond: int):
    """Create random fully connected tensor networks."""
    benchmark_name = f"fctn_{num_of_nodes}_{free_bond}_{bond_size}"
    benchmark_dir = f"benchmarks/random/{benchmark_name}"
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    network = TensorNetwork()
    index_mapping = {}
    for i in range(num_of_nodes):
        indices = []
        for j in range(num_of_nodes):
            if i == j:
                indices.append(Index(f"I{i}", free_bond))
            elif j < i:
                indices.append(Index(f"r{j}{i}", index_mapping[(j, i)]))
            else:
                index_mapping[(i, j)] = random.randint(2, 4)
                indices.append(Index(f"r{i}{j}", index_mapping[(i, j)]))
                network.add_edge(f"G{i}", f"G{j}")

        shape = [i.size for i in indices]
        network.add_node(f"G{i}", Tensor(np.random.randn(*shape), indices))

    # save the single core data file
    datafile = f"data/random/{benchmark_name}/data.npy"
    np.save(datafile, network.contract().value)

    benchmark = Benchmark(
        benchmark_name, "random", [Node("G0", network.free_indices(), datafile)]
    )
    with open(
        f"benchmarks/random/{benchmark_name}/original.json", "w+", encoding="utf-8"
    ) as json_file:
        json_str = RootModel[Benchmark](benchmark).model_dump_json(indent=4)
        json_file.write(json_str)


def gen_tt(num_of_nodes: int, bond_size: int, free_bond: int):
    """Create random tensor train benchmarks."""
    benchmark_name = f"tt_{num_of_nodes}_{free_bond}_{bond_size}"
    benchmark_dir = f"benchmarks/random/{benchmark_name}"
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    nodes = []

    n0_indices = [Index("I0", free_bond), Index("r01", bond_size)]
    nodes.append(Node("G0", n0_indices))
    for i in range(1, num_of_nodes - 1):
        indices = [
            Index(f"r{i-1}{i}", bond_size),
            Index(f"I{i}", free_bond),
            Index(f"r{i}{i+1}", bond_size),
        ]
        nodes.append(Node(f"G{i}", indices))

    nm_indices = [
        Index(f"r{num_of_nodes-2}{num_of_nodes-1}", bond_size),
        Index(f"I{num_of_nodes-1}", free_bond),
    ]
    nodes.append(nm_indices)

    benchmark = Benchmark(benchmark_name, "random", nodes)
    with open(
        f"benchmarks/random/{benchmark_name}/original.json", "w+", encoding="utf-8"
    ) as json_file:
        json_str = RootModel[Benchmark](benchmark).model_dump_json(indent=4)
        json_file.write(json_str)

def random_tree(benchmark_name, num_of_nodes: int, free_indices: List[Index]):
    length = num_of_nodes - 2
    arr = [0] * length
 
    # Generate random array
    for i in range(length):
        arr[i] = random.randint(1, length + 1)

    vertex_set = [0] * num_of_nodes
 
    # Initialize the array of vertices
    for i in range(num_of_nodes):
        vertex_set[i] = 0
 
    # Number of occurrences of vertex in code
    for i in range(length):
        vertex_set[arr[i] - 1] += 1
 
    # construct the edge set
    edge_set = []
    j = 0
    # Find the smallest label not present in prufer[].
    for i in range(length):
        for j in range(num_of_nodes):
            # If j+1 is not present in prufer set
            if vertex_set[j] == 0:
                # Remove from Prufer set and print pair.
                vertex_set[j] = -1
                edge_set.append((j+1, arr[i]))
                vertex_set[arr[i] - 1] -= 1
                break
 
    j = 0
 
    # For the last element
    edge_pair = []
    for i in range(num_of_nodes):
        if vertex_set[i] == 0 and j == 0:
            edge_pair.append(i+1)
            j += 1
        elif vertex_set[i] == 0 and j == 1:
            edge_pair.append(i+1)
            edge_set.append(edge_pair)
            edge_pair = []

    g = nx.Graph(edge_set)
    # start from a root we collect all free nodes and find a mapping between nodes and free indices
    # each node should have at least two edges
    node_candidates = []
    for n in g.nodes():
        if len(list(nx.neighbors(g, n))) == 1:
            node_candidates.append(n)

    free_assignment = node_candidates + random.choices(list(g.nodes), k=len(free_indices)-len(node_candidates))
    random.shuffle(free_assignment)
    nodes = []
    edges = {}
    for n in g.nodes:
        n_name = f"G{n}"
        n_indices = []
        if n in free_assignment:
            for mi, m in enumerate(free_assignment):
                if m == n:
                    n_indices.append(free_indices[mi])

        for gn in nx.neighbors(g, n):
            l = min(gn, n)
            r = max(gn, n)
            if (l, r) not in edges:
                edges[(l, r)] = random.randint(2, 5)
            n_indices.append(Index(f"s_{l}_{r}", edges[(l, r)]))

        nodes.append(Node(n_name, n_indices))

    b = Benchmark(benchmark_name, "random", nodes)
    benchmark_dir = f"benchmarks/random/{benchmark_name}"
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)
    with open(
        f"{benchmark_dir}/random.json", "w+", encoding="utf-8"
    ) as json_file:
        json_str = RootModel[Benchmark](b).model_dump_json(indent=4)
        json_file.write(json_str)

    # b.to_network().draw()
    # import matplotlib.pyplot as plt
    # plt.show()
    

def pde_to_benchmark(benchmark_name, data_path, data_field, data_length):
    f = h5py.File(data_path)
    if data_field is None:
        datas = []
        for k in list(f.keys())[:data_length]:
            datas.append(f[k]["data"])

        data = np.stack(datas, axis=0)
    elif data_field in f:
        data = np.array(f[data_field])[:data_length]
    else:
        print(benchmark_name, list(f.keys()))
        for k in f.keys():
            print(f[k])

    indices = [Index(f"I{i}", sz) for i, sz in enumerate(data.shape)]

    benchmark_dir = f"benchmarks/PDEBench/{benchmark_name}"
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    data_file = f"data/PDEBench/{benchmark_name}"
    if not os.path.exists(data_file):
        os.makedirs(data_file)

    np.save(f"{data_file}/data.npy", data)
    b = Benchmark(benchmark_name, "PDEBench", [Node("G0", indices, f"{data_file}/data.npy")])
    with open(
        f"{benchmark_dir}/original.json", "w+", encoding="utf-8"
    ) as json_file:
        json_str = RootModel[Benchmark](b).model_dump_json(indent=4)
        json_file.write(json_str)

def cfd_pde_to_benchmark(benchmark_name, data_path, data_length):
    f = h5py.File(data_path)
    datas = []
    print(len(f["density"]))
    indices = random.sample(range(len(f["density"])), k=data_length)
    print(indices)
    for data_field in ["Vx", "Vy", "Vz", "density", "pressure"]:
        if data_field not in f:
            continue

        d = np.array(f[data_field])[indices]
        print(data_field, d.shape)
        datas.append(d)

    data = np.stack(datas, axis=1)

    indices = [Index(f"I{i}", sz) for i, sz in enumerate(data.shape)]

    benchmark_dir = f"benchmarks/PDEBench/{benchmark_name}"
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    data_file = f"data/PDEBench/{benchmark_name}"
    if not os.path.exists(data_file):
        os.makedirs(data_file)

    np.save(f"{data_file}/data.npy", data)
    b = Benchmark(benchmark_name, "PDEBench", [Node("G0", indices, f"{data_file}/data.npy")])
    with open(
        f"{benchmark_dir}/original.json", "w+", encoding="utf-8"
    ) as json_file:
        json_str = RootModel[Benchmark](b).model_dump_json(indent=4)
        json_file.write(json_str)

def cfd_pde_to_batches(benchmark_name, data_path, data_length):
    f = h5py.File(data_path)
    total_length = len(f["density"])
    remaining_indices = list(range(total_length))

    for batch_idx in range(total_length // data_length):
        datas = []
        indices = random.sample(remaining_indices, k=min(len(remaining_indices), data_length))
        remaining_indices = list(set(remaining_indices).difference(set(indices)))
        print(indices)
        for data_field in ["Vx", "Vy", "Vz", "density", "pressure"]:
            if data_field not in f:
                continue

            d = np.array(f[data_field])[indices]
            print(data_field, d.shape)
            datas.append(d)

        tensor = np.stack(datas, axis=1)
        tensor_indices = [Index(f"I{i}", sz) for i, sz in enumerate(tensor.shape)]

        benchmark_dir = f"benchmarks/PDEBench/{benchmark_name}_batches/{batch_idx}"
        if not os.path.exists(benchmark_dir):
            os.makedirs(benchmark_dir)

        data_file = f"data/PDEBench/{benchmark_name}_batches/{batch_idx}"
        if not os.path.exists(data_file):
            os.makedirs(data_file)

        np.save(f"{data_file}/data.npy", tensor)
        b = Benchmark(f"{benchmark_name}_batch_{batch_idx}", "PDEBench", [Node("G0", tensor_indices, f"{data_file}/data.npy")])
        with open(
            f"{benchmark_dir}/original.json", "w+", encoding="utf-8"
        ) as json_file:
            json_str = RootModel[Benchmark](b).model_dump_json(indent=4)
            json_file.write(json_str)

def bigearthnet_to_batches():
    from torchgeo.datasets import BigEarthNet

    # Initialize the dataset
    dataset = BigEarthNet(root="/Users/zhgguo/Downloads/BigEarthNet", bands="s2", download=False)
    total_length = len(dataset)
    print(total_length)
    remaining_indices = list(range(total_length))

    for batch_idx in range(total_length // 3000):
        indices = random.sample(remaining_indices, k=min(len(remaining_indices), 3000))
        remaining_indices = list(set(remaining_indices).difference(set(indices)))
        imgs = [dataset[x]["image"].numpy() for x in indices]
        tensor = np.stack(imgs, axis=0).reshape(5, 20, 30, 12, 120, 120)
        tensor_indices = [Index(f"I{i}", sz) for i, sz in enumerate(tensor.shape)]

        benchmark_dir = f"benchmarks/BigEarthNet/bigearthnet_batches/{batch_idx}"
        if not os.path.exists(benchmark_dir):
            os.makedirs(benchmark_dir)

        data_file = f"data/BigEarthNet/bigearthnet_batches/{batch_idx}"
        if not os.path.exists(data_file):
            os.makedirs(data_file)

        np.save(f"{data_file}/data.npy", tensor)
        b = Benchmark(f"bigearthnet_batch_{batch_idx}", "PDEBench", [Node("G0", tensor_indices, f"{data_file}/data.npy")])
        with open(
            f"{benchmark_dir}/original.json", "w+", encoding="utf-8"
        ) as json_file:
            json_str = RootModel[Benchmark](b).model_dump_json(indent=4)
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

    # Script for creating benchmark files from SVDinsTN dataset
    # import cv2
    # # name, start_num = "truck", 5785
    # # name, start_num = "bunny", 3352
    # name, start_num = "knight", 9566
    # for k in range(10):
    #     # random start point
    #     start_x = random.randint(0, 8)
    #     start_y = random.randint(0, 8)
    #     stacks = []
    #     for i in range(start_x, start_x + 9):
    #         imgs = []
    #         for j in range(start_y, start_y + 9):
    #             # f = glob.glob(f"/Users/zhgguo/Downloads/{name}_rectified/out_{i:02}_{j:02}_*.png")[0]
    #             f = glob.glob(f"/Users/zhgguo/Downloads/{name}/IMG_{start_num + i * 17 + j}.JPG")[0]
    #             # print(i, j, f)
    #             img = cv2.imread(f)
    #             img = cv2.resize(img, (60, 40), interpolation=cv2.INTER_AREA)
    #             # img = cv2.resize(img, (60, 40))
    #             # print(img.shape)
    #             imgs.append(img)

    #         stacked_x = np.stack(imgs, axis=0)
    #         stacks.append(stacked_x)
            
    #     stacked_y = np.stack(stacks, axis=0).transpose(2,3,4,0,1)
    #     convert_single_value_to_benchmark(f"{name}_{k}", "SVDinsTN", data=stacked_y)

    # Script for creating benchmark files for BigEarthNet dataset
    # from torchgeo.datasets import BigEarthNet

    # # Initialize the dataset
    # dataset = BigEarthNet(root="/Users/zhgguo/Downloads/BigEarthNet", bands="s2", download=True)

    # # sizes = [(30,), (10, 30), (5, 20, 30)]
    # sizes = [(i,) for i in range(5, 51, 5)]
    # for si, s in enumerate(sizes):
    #     for i in range(1):
    #         indices = random.sample(range(len(dataset)), k=int(np.prod(s)))
    #         imgs = [dataset[x]["image"].numpy() for x in indices]
    #         data = np.stack(imgs, axis=0).reshape(*s, 12, 120, 120)
    #         name = f"bigearthnet_sample_{si}"
    #         convert_single_value_to_benchmark(name, "BigEarthNet", data=data)

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

    # source = "BigEarthNet-v1_0_all"
    # eps = 0.1
    # for name in os.listdir(f"data/{source}"):
    #     if name == ".DS_Store":
    #         continue

    #     start_from_ht("data", source, name, eps)

    # for num_nodes in range(4, 11):
    #     for bidx in range(5):
    #         gen_fctn(num_nodes, bidx, 20 if num_nodes < 6 else 5)

    # for i in range(100):
    #     if i < 25:
    #         random_tree(f"random_test_{i}", 4, [Index("I0", 16), Index("I1", 18), Index("I2", 20), Index("I3", 22)])
    #     elif i < 50:
    #         random_tree(f"random_test_{i}", 5, [Index("I0", 16), Index("I1", 18), Index("I2", 20), Index("I3", 22)])
    #     elif i < 75:
    #         random_tree(f"random_test_{i}", 5, [Index("I0", 16), Index("I1", 18), Index("I2", 20), Index("I3", 22), Index("I4", 14)])
    #     else:
    #         random_tree(f"random_test_{i}", 6, [Index("I0", 16), Index("I1", 18), Index("I2", 20), Index("I3", 22), Index("I4", 14)])

    # source= "PDEBench"
    # pde = {
    #     "1D_Advection": ("/Users/zhgguo/Downloads/1D_Advection_Sols_beta0.1.hdf5", "tensor", 5000),
    #     "1D_Burgers": ("/Users/zhgguo/Downloads/1D_Burgers_Sols_Nu0.001.hdf5", "tensor", 5000),
    #     "1D_Diff_Sorp": ("/Users/zhgguo/Downloads/1D_diff-sorp_NA_NA.h5", None, 5000),
    #     "1D_CFD_Vx": ("/Users/zhgguo/Downloads/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5", "Vx", 5000),
    #     "1D_CFD_density": ("/Users/zhgguo/Downloads/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5", "density", 5000),
    #     "1D_CFD_pressure": ("/Users/zhgguo/Downloads/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5", "pressure", 5000),
    #     "1D_Diffusion_Reaction": ("/Users/zhgguo/Downloads/ReacDiff_Nu0.5_Rho1.0.hdf5", "tensor", 5000),
    #     "2D_CFD_Vx": ("/Users/zhgguo/Downloads/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5", "Vx", 5000),
    #     "2D_CFD_Vy": ("/Users/zhgguo/Downloads/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5", "Vy", 5000),
    #     "2D_CFD_density": ("/Users/zhgguo/Downloads/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5", "density", 5000),
    #     "2D_CFD_pressure": ("/Users/zhgguo/Downloads/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5", "pressure", 5000),
    #     "2D_Diffusion_Reaction": ("/Users/zhgguo/Downloads/2D_diff-react_NA_NA.h5", None, 100),
    #     "2D_NS_Incom_Inhom": ("/Users/zhgguo/Downloads/ns_incom_inhom_2d_512-100.h5", "velocity", 5000),
    #     "2D_DarcyFlow": ("/Users/zhgguo/Downloads/2D_DarcyFlow_beta0.01_Train.hdf5", "tensor", 5000),
    #     "2D_ShallowWater": ("/Users/zhgguo/Downloads/2D_rdb_NA_NA.h5", None, 100),
    #     "3D_CFD_Vx": ("/Users/zhgguo/Downloads/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", "Vx", 100),
    #     "3D_CFD_Vy": ("/Users/zhgguo/Downloads/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", "Vy", 100),
    #     "3D_CFD_Vz": ("/Users/zhgguo/Downloads/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", "Vz", 100),
    #     "3D_CFD_density": ("/Users/zhgguo/Downloads/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", "density", 100),
    #     "3D_CFD_pressure": ("/Users/zhgguo/Downloads/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", "pressure", 100),
    # }
    # for name, data in pde.items():
    #     pde_to_benchmark(name, *data)

    # source= "PDEBench"
    # pde = {
    #     # "1D_CFD": ("/Users/zhgguo/Downloads/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5", 100),
    #     # "2D_CFD": ("/Users/zhgguo/Downloads/164686", 10),
    #     "3D_CFD": ("/Users/zhgguo/Downloads/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", 10),
    # }
    # for name, data in pde.items():
    #     for i in range(1, 11):
    #         cfd_pde_to_benchmark(f"{name}_sample_{i}", data[0], i)
    # cfd_pde_to_batches("3D_CFD", "/Users/zhgguo/Downloads/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", 10)
    bigearthnet_to_batches()