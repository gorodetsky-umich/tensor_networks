"""Script for neural network training."""

import json
import torch

from benchmarks.benchmark import Benchmark
from pytens.search.nn import RLTrainer

def eps_to_str(eps: float) -> str:
    """Convert an epsilon to a formatted string."""
    return "".join(f"{eps:.2f}".split('.'))

if __name__ == "__main__":
    import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True, help="Path pattern of the selected benchmarks")
    parser.add_argument("--eps", type=float, help="Error target")
    parser.add_argument("--max_ops", type=int, default=5, help="Maximum number of operations to search for")
    parser.add_argument("--no-heuristic", action="store_true", help="Disable prune of no truncation")
    parser.add_argument("--split_errors", type=int, default=0, help="Consider all possible ranks in each split action")
    parser.add_argument("--gamma", type=float, default=1e-3, help="Gamma value used in SVDinsTN")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    nets = []
    for benchmark_file in glob.glob(args.pattern):
        with open(benchmark_file, "r", encoding="utf-8") as benchmark_fd:
            benchmark_dict = json.load(benchmark_fd)
            b = Benchmark(**benchmark_dict)
            # Note that we only do the training for networks starting from single cores,
            # otherwise this logic is incorrect
            net = b.to_network(normalize=True)
            net.compress()
            nets.append(net)

    trainer = RLTrainer(args.__dict__)
    if args.test:
        with open("models/value.pkl", "rb") as value_model:
            trainer.value_net.load_state_dict(torch.load(value_model, weights_only=True))

        with open("models/action.pkl", "rb") as action_model:
            trainer.op_picker.load_state_dict(torch.load(action_model, weights_only=True))
        
        with open("models/state.pkl", "rb") as state_model:
            trainer.state_to_torch.load_state_dict(torch.load(state_model, weights_only=True))

        trainer.beam_rollout(nets[0])
    else:
        with open("models/value.pkl", "rb") as value_model:
            trainer.value_net.load_state_dict(torch.load(value_model, weights_only=True))

        with open("models/action.pkl", "rb") as action_model:
            trainer.op_picker.load_state_dict(torch.load(action_model, weights_only=True))

        with open("models/state.pkl", "rb") as state_model:
            trainer.state_to_torch.load_state_dict(torch.load(state_model, weights_only=True))

        trainer.train(nets)
