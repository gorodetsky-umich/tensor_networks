"""Class for beam search."""

import time
import heapq

from typing import List, Optional
import torch

from pytens.search.state import SearchState, Split
from pytens.search.nn import RLTrainer

class BeamSearch:
    """Beam search with a given beam size."""

    def __init__(self, params):
        self.params = params
        self.heap: List[SearchState] = None
        self.initial_cost = 0
        self.best_network = None
        self.stats = {
            "split": 0,
            "merge": 0,
            "split time": 0,
            "count": 0,
            "compression": [],
        }

    def search(self, initial_state: SearchState, guided: bool = False):
        """Perform the beam search from the given initial state."""
        self.initial_cost = initial_state.network.cost()
        self.best_network = initial_state.network
        self.heap = [(-self.initial_cost, initial_state)] # the initial state has a very bad score
        # trainer = RLTrainer(self.params)
        tic = time.time()
        trainer = None
        if guided:
            trainer = RLTrainer(self.params)
            with open("models/value.pkl", "rb") as value_model:
                trainer.value_net.load_state_dict(torch.load(value_model, weights_only=True))

            with open("models/action.pkl", "rb") as action_model:
                trainer.op_picker.load_state_dict(torch.load(action_model, weights_only=True))
            
            with open("models/state.pkl", "rb") as state_model:
                trainer.state_to_torch.load_state_dict(torch.load(state_model, weights_only=True))

        for _ in range(self.params["max_ops"]):
            # start = time.time()
            # maintain a set of networks of at most k
            self.step(tic, trainer)
            # print("one step time", time.time() - start)

        # self.stats["unique"] = len(self.stats["unique"])
        # print(self.stats)

    def get_score(self, st: SearchState, trainer: Optional[RLTrainer] = None):
        """Get the prediction score for a given state."""
        if trainer is None:
            return -st.network.cost()
        else:
            st_encoding = trainer.state_to_torch(st)
            return trainer.value_net(st_encoding)

    def step(self, tic, trainer: Optional[RLTrainer] = None):
        """Make a step in a beam search."""
        next_level = []
        
        while len(self.heap) > 0:
            _, state = heapq.heappop(self.heap)
            for ac in state.get_legal_actions():
                action_start = time.time()
                for new_state in state.take_action(ac, split_errors=self.params["split_errors"], no_heuristic=self.params["no_heuristic"]):
                    if new_state.is_noop:
                        continue

                    self.stats["count"] += 1
                    # self.stats["unique"].add(new_state.network.canonical_structure())
                    new_score = self.get_score(new_state, trainer)
                    if len(next_level) < self.params["beam_size"]:
                        heapq.heappush(next_level, (new_score, new_state))
                    elif self.get_score(next_level[0][1], trainer) < new_score: # we decide whether to add a network by its value network
                        heapq.heappushpop(next_level, (new_score, new_state))

                    if new_state.network.cost() < self.best_network.cost():
                        self.best_network = new_state.network

                    self.stats["compression"].append((time.time() - tic, self.initial_cost / new_state.network.cost()))

                if isinstance(ac, Split):
                    self.stats["split time"] += time.time() - action_start
                    self.stats["split"] += 1
                else:
                    self.stats["merge"] += 1

        self.heap = next_level