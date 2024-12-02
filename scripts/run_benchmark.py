"""Script for benchmark running"""

from typing import Any, Dict
import json
import os
import pickle

import matplotlib.pyplot as plt
from svdinstn_decomposition import FCTN

from benchmarks.benchmark import Benchmark
from pytens.search.search import SearchEngine
from pytens.algs import TensorNetwork, Index, Tensor
from htucker.algs import HTucker, createDimensionTree, TuckerCore


def eps_to_str(eps: float) -> str:
    """Convert an epsilon number to a formatted string."""
    return "".join(f"{eps:.2f}".split("."))


class Runner:
    """Benchmark runner."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.engine = SearchEngine(params)

    def _to_ht(self, core: Tensor) -> TensorNetwork:
        core_indices = core.indices
        ht_net = TensorNetwork()
        ht = HTucker()
        ht.initialize(core.value)
        dimension_tree = createDimensionTree(
            ht.original_shape, numSplits=2, minSplitSize=1
        )
        dimension_tree.get_items_from_level()
        ht.rtol = self.params["eps"]
        ht.compress_leaf2root(core.value, dimension_tree)

        # save the result as a new tensor network
        dims_to_idx = {
            f"transfer:{ht.root.core_idx}": 0,
            f"transfer:{ht.root.children[0].core_idx}": 1,
            f"transfer:{ht.root.children[1].core_idx}": 2,
        }
        root_indices = [
            Index("s_0_1", ht.root.core.shape[0]),
            Index("s_0_2", ht.root.core.shape[1]),
        ]
        ht_net.add_node("G0", Tensor(ht.root.core, root_indices))
        for n in ht.transfer_nodes:
            p = f"transfer:{n.parent.core_idx}"
            if p not in dims_to_idx:
                dims_to_idx[p] = len(dims_to_idx)
            p = dims_to_idx[p]

            curr = f"transfer:{n.core_idx}"
            if curr not in dims_to_idx:
                dims_to_idx[curr] = len(dims_to_idx)
            curr = dims_to_idx[curr]

            if isinstance(n.children[0], TuckerCore):
                lc = f"transfer:{n.children[0].core_idx}"
            else:
                lc = f"leaf:{n.children[0].leaf_idx}"

            if lc not in dims_to_idx:
                dims_to_idx[lc] = len(dims_to_idx)
            lc = dims_to_idx[lc]

            if isinstance(n.children[1], TuckerCore):
                rc = f"transfer:{n.children[1].core_idx}"
            else:
                rc = f"leaf:{n.children[1].leaf_idx}"

            if rc not in dims_to_idx:
                dims_to_idx[rc] = len(dims_to_idx)
            rc = dims_to_idx[rc]

            # print(dims_to_idx)
            dims = n.core.shape
            n_indices = [
                Index(f"s_{curr}_{lc}", dims[0]),
                Index(f"s_{curr}_{rc}", dims[1]),
                Index(f"s_{p}_{curr}", dims[2]),
            ]
            ht_net.add_node(f"G{curr}", Tensor(n.core, n_indices))
            ht_net.add_edge(f"G{curr}", f"G{p}")

        for l in ht.leaves:
            p = f"transfer:{l.parent.core_idx}"
            if p not in dims_to_idx:
                dims_to_idx[p] = len(dims_to_idx)
            p = dims_to_idx[p]

            curr = f"leaf:{l.leaf_idx}"
            if curr not in dims_to_idx:
                dims_to_idx[curr] = len(dims_to_idx)
            curr = dims_to_idx[curr]

            dims = l.core.shape
            l_indices = [core_indices[l.leaf_idx], Index(f"s_{p}_{curr}", dims[1])]
            ht_net.add_node(f"L{curr}", Tensor(l.core, l_indices))
            ht_net.add_edge(f"L{curr}", f"G{p}")

        return ht_net

    def run(self, benchmark: Benchmark, repeat: int = 1):
        """Run a benchmark for the given repeated times."""
        if self.params["engine"] == "dfs":
            search_engine = self.engine.dfs
        elif self.params["engine"] == "bfs":
            search_engine = self.engine.bfs
        elif self.params["engine"] == "mcts":
            search_engine = self.engine.mcts
        elif self.params["engine"] == "beam":
            search_engine = self.engine.beam
        elif self.params["engine"] == "svdinstn":
            search_engine = FCTN
        else:
            raise RuntimeError("unrecognized search engine")

        for i in range(repeat):
            net = benchmark.to_network()

            if "core" in self.params["start_from"]:
                core = net.contract()
                net = TensorNetwork()
                net.add_node("G0", core)

            # if start from hierarchical tucker, we call ht
            if "ht" in self.params["start_from"]:
                tic = time.time()
                net = self._to_ht(net.contract())
                toc = time.time()
                ht_construct_time = toc - tic
            else:
                ht_construct_time = 0

            net.draw()
            # plt.show()

            eps_str = eps_to_str(self.params["eps"])
            # log_name = f"{self.params['engine']}_{eps_str}{ht_tag}_ops_{self.params['max_ops']}_split_{self.params['split_errors']}"
            output_dir = f"output/{benchmark.source}/{benchmark.name}/{eps_str}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            plt.savefig(f"{output_dir}/start_from.png")
            plt.close()

            net.compress()
            net.draw()
            plt.savefig(f"{output_dir}/start_from_compressed.png")
            plt.close()

            all_stats = []
            with open(
                f"{output_dir}/{self.params['log_name']}_{i}.log", "w", encoding="utf-8"
            ) as f:
                if self.params["engine"] == "svdinstn":
                    fctn = search_engine(
                        net,
                        timeout=self.params["timeout"],
                        gamma=self.params["gamma"],
                        eps=self.params["eps"],
                    )
                    fctn.initialize()
                    fctn.decompose()
                    fctn.to_tensor_network()
                    stats = fctn.stats
                elif self.params["engine"] == "beam":
                    data_name = benchmark.name.split("_ht_")[0]
                    data_file = f"data/{benchmark.source}/{data_name}/data.npy"
                    if os.path.exists(data_file):
                        target_tensor = np.load(
                            f"data/{benchmark.source}/{data_name}/data.npy"
                        )
                    else:
                        target_tensor = None

                    stats = search_engine(net, target_tensor)
                else:
                    stats = search_engine(net)

                f.write(
                    f"{i},{stats['time']},{stats['reconstruction_error']},{stats['cr_core']},{stats['cr_start']},{ht_construct_time}\n"
                )
                bn = stats.pop("best_network")
                with open(f"{output_dir}/{self.params['log_name']}.pkl", "wb") as f:
                    pickle.dump(bn, f)

                bn.draw()
                plt.savefig(f"{output_dir}/{self.params['log_name']}_result.png")
                plt.close()
                all_stats.append(stats)

        with open(f"{output_dir}/{log_name}_{i}_all.log", "w", encoding="utf-8") as f:
            json.dump(all_stats, f)


if __name__ == "__main__":
    import glob
    import argparse
    import time
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Path pattern of the selected benchmarks",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["bfs", "dfs", "mcts", "beam", "svdinstn"],
        help="Type of the search engine",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeats to run for each benchmark",
    )
    parser.add_argument("--eps", type=float, help="Error target")
    parser.add_argument(
        "--max_ops",
        type=int,
        default=5,
        help="Maximum number of operations to search for",
    )
    parser.add_argument(
        "--beam_size", type=int, help="Specify the beam size during beam search"
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Whether to perform pruning during BFS or DFS",
    )
    parser.add_argument(
        "--consider_ranks",
        action="store_true",
        help="Whether to consider edge ranks during pruning",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Whether to optimize the found structure by global optimization",
    )
    parser.add_argument(
        "--no-heuristic", action="store_true", help="Disable prune of no truncation"
    )
    parser.add_argument(
        "--split_errors",
        type=int,
        default=0,
        help="Consider all possible ranks in each split action",
    )
    parser.add_argument(
        "--guided",
        action="store_true",
        help="Whether to use neural network to guide the beam search",
    )
    parser.add_argument(
        "--start_from",
        choices=["ht", "tt", "core"],
        default="core",
        nargs="+",
        help="Choose which algorithm result to start from",
    )
    parser.add_argument("--timeout", type=float, help="Timeout limit")
    parser.add_argument(
        "--gamma", type=float, default=1e-3, help="Gamma value used in SVDinsTN"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to perform verbose logging"
    )
    parser.add_argument(
        "--collect_only",
        action="store_true",
        help="Collect log data into one single file",
    )

    args = parser.parse_args()

    runner = Runner(args.__dict__)

    print(
        f"{'Name':35}\t{'Time':>10}\t{'HT Time':>10}\t{'RE':>10}\t{'CR_core':>10}\t{'CR_start':>10}\t{'Best_time':>10}\t{'Max_ops':>10}"
    )
    for benchmark_file in glob.glob(args.pattern):
        with open(benchmark_file, "r", encoding="utf-8") as benchmark_fd:
            benchmark_dict = json.load(benchmark_fd)
            b = Benchmark(**benchmark_dict)
            print(f"{b.name:35}\t", end="")
            np.random.seed(int(time.time()))

            if "ht" in args.start_from:
                ht_tag = "_ht"
            else:
                ht_tag = ""

            log_name = f"{args.engine}_{eps_to_str(args.eps)}{ht_tag}_ops_{args.max_ops}_split_{args.split_errors}"
            args["log_name"] = log_name

            if not args.collect_only:
                runner.run(b, repeat=args.repeat)

            out_dir = f"output/{b.source}/{b.name}/{eps_to_str(args.eps)}"
            if not os.path.exists(out_dir):
                print("Directory", out_dir, "does not exist")

            slog_name = f"{out_dir}/{log_name}_0.log"

            with open(slog_name, "r", encoding="utf-8") as slog_file:
                line = slog_file.read().strip()
                _, t, re, cr_core, cr_start, ht_time = line.split(",")

            alog_name = f"{out_dir}/{log_name}_0_all.log"
            if os.path.exists(alog_name):
                with open(alog_name, "r", encoding="utf-8") as alog_file:
                    all_log = json.load(alog_file)[0]
                    # get the time that reaches the best network
                    if "best_cost" in all_log and all_log["best_cost"]:
                        best_cost = all_log["best_cost"][-1][1]
                        for ts, c in all_log["best_cost"]:
                            if c == best_cost:
                                first_best_cost = ts
                                break

                        max_ops = all_log["ops"][-1][1]
                    else:
                        best_cost = 0
                        first_best_cost = 0
                        max_ops = 0
            else:
                first_best_cost = None
                max_ops = None

            print(
                f"{float(t):>10.5f}\t{float(ht_time):>10.5f}\t{float(re):>10.5f}\t{float(cr_core):>10.5f}\t{float(cr_start):>10.5f}\t{float(first_best_cost):>10.5f}\t{max_ops:>10}"
            )
