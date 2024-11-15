"""Script for benchmark running"""

from typing import Any, Dict
import json
import os
import pickle

import matplotlib.pyplot as plt

from benchmarks.benchmark import Benchmark
from pytens.search import SearchEngine
from svdinstn_decomposition import FCTN

def eps_to_str(eps: float) -> str:
    return "".join(f"{eps:.2f}".split('.'))

class Runner:
    """Benchmark runner."""
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.engine = SearchEngine(params)

    def run(self, benchmark: Benchmark, repeat: int = 1):
        """Run a benchmark for the given repeated times."""
        net = benchmark.to_network()
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

        net.draw()

        eps_str = eps_to_str(self.params['eps'])
        log_name = f"{self.params['engine']}_{eps_str}_ops_{self.params['max_ops']}_split_{self.params['split_errors']}"
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
        with open(f"{output_dir}/{log_name}.log", "w", encoding="utf-8") as f:
            for i in range(repeat):
                if self.params["engine"] == "svdinstn":
                    fctn = search_engine(net, timeout=self.params["timeout"],
                                         gamma=self.params["gamma"],
                                         eps=self.params["eps"])
                    fctn.initialize()
                    fctn.decompose()
                    fctn.to_tensor_network()
                    stats = fctn.stats
                elif self.params["engine"] == "beam":
                    data_name = benchmark.name.split("_ht_")[0]
                    target_tensor = np.load(f"data/{benchmark.source}/{data_name}/data.npy")
                    stats = search_engine(net, target_tensor)
                else: 
                    stats = search_engine(net)
                
                f.write(f"{i},{stats['time']},{stats['reconstruction_error']},{stats['cr_core']},{stats['cr_start']}\n")
                bn = stats.pop("best_network")
                with open(f"{output_dir}/{log_name}.pkl", "wb") as f:
                    pickle.dump(bn, f)

                bn.draw()
                plt.savefig(f"{output_dir}/{log_name}_result.png")
                plt.close()
                all_stats.append(stats)

        with open(f"{output_dir}/{log_name}_all.log", "w", encoding="utf-8") as f:
            json.dump(all_stats, f)

if __name__ == "__main__":
    import glob
    import argparse
    import time
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True, help="Path pattern of the selected benchmarks")
    parser.add_argument("--engine", type=str, choices=["bfs", "dfs", "mcts", "beam", "svdinstn"], help="Type of the search engine")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeats to run for each benchmark")
    parser.add_argument("--eps", type=float, help="Error target")
    parser.add_argument("--max_ops", type=int, default=5, help="Maximum number of operations to search for")
    parser.add_argument("--beam_size", type=int, help="Specify the beam size during beam search")
    parser.add_argument("--prune", action="store_true", help="Whether to perform pruning during BFS or DFS")
    parser.add_argument("--consider_ranks", action="store_true", help="Whether to consider edge ranks during pruning")
    parser.add_argument("--optimize", action="store_true", help="Whether to optimize the found structure by global optimization")
    parser.add_argument("--no-heuristic", action="store_true", help="Disable prune of no truncation")
    parser.add_argument("--single_core_start", action="store_true", help="Start from a single core instead of the given network")
    parser.add_argument("--split_errors", type=int, default=0, help="Consider all possible ranks in each split action")
    parser.add_argument("--guided", action="store_true", help="Whether to use neural network to guide the beam search")
    parser.add_argument("--timeout", type=float, help="Timeout limit")
    parser.add_argument("--gamma", type=float, default=1e-3, help="Gamma value used in SVDinsTN")
    parser.add_argument("--verbose", action="store_true", help="Whether to perform verbose logging")
    parser.add_argument("--collect_only", action="store_true", help="Collect log data into one single file")

    args = parser.parse_args()

    runner = Runner(args.__dict__)

    all_stats = []
    print(f"{'Name':35}\t{'Time':>10}\t{'RE':>10}\t{'CR_core':>10}\t{'CR_start':>10}\t{'Best_time':>10}\t{'Max_ops':>10}")
    for benchmark_file in glob.glob(args.pattern):
        with open(benchmark_file, "r", encoding="utf-8") as benchmark_fd:
            benchmark_dict = json.load(benchmark_fd)
            b = Benchmark(**benchmark_dict)
            np.random.seed(int(time.time()))
            if not args.collect_only:
                runner.run(b, repeat=args.repeat)

            out_dir = f"output/{b.source}/{b.name}/{eps_to_str(args.eps)}"
            if not os.path.exists(out_dir):
                print("Directory", out_dir, "does not exist")

            log_name = f"{args.engine}_{eps_to_str(args.eps)}_ops_{args.max_ops}_split_{args.split_errors}"
            slog_name = f"{out_dir}/{log_name}.log"
            with open(slog_name, "r", encoding="utf-8") as slog_file:
                line = slog_file.read().strip()
                _, t, re, cr_core, cr_start = line.split(",")

            alog_name = f"{out_dir}/{log_name}_all.log"
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

            print(f"{b.name:35}\t{float(t):>10.5f}\t{float(re):>10.5f}\t{float(cr_core):>10.5f}\t{float(cr_start):>10.5f}\t{float(first_best_cost):>10.5f}\t{max_ops:>10}")
