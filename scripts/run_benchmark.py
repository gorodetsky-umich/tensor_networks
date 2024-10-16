"""Script for benchmark running"""

from typing import Any, Dict
import json
import matplotlib.pyplot as plt

from benchmarks.benchmark import Benchmark
from pytens.search import SearchEngine

class Runner:
    """Benchmark runner."""
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.engine = SearchEngine(params)

    def run(self, benchmark: Benchmark, repeat: int = 1):
        """Run a benchmark for the given repeated times."""
        name = benchmark.name
        net = benchmark.to_network()
        if self.params["engine"] == "exhaustive":
            search_engine = self.engine.exhaustive
        else:
            raise RuntimeError("unrecognized search engine")

        log_name = f"{name}_{self.params['engine']}"
        all_stats = []
        with open(f"{log_name}.log", "w", encoding="utf-8") as f:
            for i in range(repeat):
                stats = search_engine(net)
                f.write(f"{i},{stats['time']},{stats['reconstruction_error']}\n")
                bn = stats.pop("best_network")
                bn.draw()
                plt.savefig(f"{log_name}_result.png")
                plt.close()
                all_stats.append(stats)

        with open(f"{log_name}_all.log", "w", encoding="utf-8") as f:
            json.dump(all_stats, f)

if __name__ == "__main__":
    import os
    import glob
    import argparse
    import time
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Directory to the benchmarks")
    parser.add_argument("--engine", type=str, default="exhaustive", help="Type of the search engine")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeats to run for each benchmark")
    parser.add_argument("--verbose", action="store_true", help="Whether to perform verbose logging")

    args = parser.parse_args()

    if not os.path.exists(args.benchmark_dir):
        raise ValueError(f"{args.benchmark_dir} does not exist.")

    runner = Runner(args.__dict__)

    for benchmark_file in glob.glob(f"{args.benchmark_dir}/*.json"):
        with open(benchmark_file, "r", encoding="utf-8") as benchmark_fd:
            benchmark_dict = json.load(benchmark_fd)
            b = Benchmark(**benchmark_dict)
            np.random.seed(int(time.time()))
            runner.run(b, repeat=args.repeat)
