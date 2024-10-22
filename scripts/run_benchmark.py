"""Script for benchmark running"""

from typing import Any, Dict
import json
import os
import matplotlib.pyplot as plt

from benchmarks.benchmark import Benchmark
from pytens.search import SearchEngine

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
        if self.params["engine"] == "exhaustive":
            search_engine = self.engine.exhaustive
        else:
            raise RuntimeError("unrecognized search engine")

        # net.draw()

        eps_str = eps_to_str(self.params['eps'])
        log_name = f"{self.params['engine']}_{eps_str}"
        output_dir = f"output/{benchmark.source}/{benchmark.name}/{eps_str}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # plt.savefig(f"{output_dir}/start_from.png")
        # plt.close()

        all_stats = []
        with open(f"{output_dir}/{log_name}.log", "w", encoding="utf-8") as f:
            for i in range(repeat):
                stats = search_engine(net)
                f.write(f"{i},{stats['time']},{stats['reconstruction_error']},{stats['cr_core']},{stats['cr_start']}\n")
                bn = stats.pop("best_network")
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
    parser.add_argument("--engine", type=str, default="exhaustive", help="Type of the search engine")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeats to run for each benchmark")
    parser.add_argument("--eps", type=float, help="Error target")
    parser.add_argument("--verbose", action="store_true", help="Whether to perform verbose logging")
    parser.add_argument("--collect_only", action="store_true", help="Collect log data into one single file")

    args = parser.parse_args()

    runner = Runner(args.__dict__)

    all_stats = []
    print(f"{'Name':35}\t{'Time':>10}\t{'RE':>10}\t{'CR_core':>10}\t{'CR_start':>10}\t{'Best time':>10}\t{'Max ops':>10}")
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

            slog_name = f"{out_dir}/{args.engine}_{eps_to_str(args.eps)}.log"
            with open(slog_name, "r", encoding="utf-8") as slog_file:
                line = slog_file.read().strip()
                _, t, re, cr_core, cr_start = line.split(",")

            alog_name = f"{out_dir}/{args.engine}_{eps_to_str(args.eps)}_all.log"
            if os.path.exists(alog_name):
                with open(alog_name, "r", encoding="utf-8") as alog_file:
                    all_log = json.load(alog_file)[0]
                    # get the time that reaches the best network
                    best_cost = all_log["best_cost"][-1][1]
                    for ts, c in all_log["best_cost"]:
                        if c == best_cost:
                            first_best_cost = ts
                            break

                    max_ops = all_log["ops"][-1][1]
            else:
                first_best_cost = None
                max_ops = None

            print(f"{b.name:35}\t{float(t):>10.5f}\t{float(re):>10.5f}\t{float(cr_core):>10.5f}\t{float(cr_start):>10.5f}\t{float(first_best_cost):>10.5f}\t{max_ops:>10}")
