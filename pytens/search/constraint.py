"""Linear constraints for finding best rank assignment."""

import copy
import itertools
import logging
import os
from typing import List, Optional, Sequence, Dict

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from pytens.algs import Index, Tensor, TreeNetwork
from pytens.cross.funcs import TensorFunc, FuncTensorNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.state import OSplit, SearchState
from pytens.search.utils import DataTensor

BAD_SCORE = 9999999999999

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ILPSolver:
    """An ILP solver to find near-optimal rank assignments."""

    def __init__(self, config: SearchConfig):
        self.config = config
        # Create empty environment, set options and start
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("TimeLimit", 60)
        env.start()
        self.env = env
        self.model = gp.Model("A model", env=env)
        self.vars = gp.tupledict()

    def add_var(self, ind: Index):
        """Add variables for a given rank i"""
        # for a given edge, we add binary variables i0, i1, .., in
        indices = [(ind.name, j) for j in ind.value_choices]
        # print(indices)
        # print(ind, len(indices), ind.size[1] - ind.size[0])
        self.vars.update(self.model.addVars(indices, vtype=GRB.BINARY))

    def add_constraint(self, inds: List[Index], pfsums, delta: float):
        """Given n ranks to be solved, generate all constraints

        Constr1: sum_j xij = 1
        Constr2: sum_ij xij*pij <= delta**2

        Arguments:
            n - Number of ranks to be resolved
            pfsums - Prefix sums of singular values for corresponding edges
            delta - The maximum error can be accumulated
        """
        coeff = {}
        for ind in inds:
            self.model.addConstr(self.vars.sum(ind.name, "*") == 1)

            # print(ind)
            # print(ind, len(pfsums[ind.name]))
            assert len(pfsums[ind.name]) == len(ind.value_choices)
            for sz, p in zip(ind.value_choices, pfsums[ind.name]):
                coeff[(ind.name, sz)] = p

        # print(self.vars, delta ** 2)
        self.model.addConstr(self.vars.prod(coeff) <= delta**2)

    def set_objective(
        self, free_indices: List[Index], nodes: List[Tensor], upper: int
    ):
        """Set the objective for the solver."""
        # max_cost = np.prod([i.size for i in free_indices])
        cost = gp.LinExpr()
        for node in nodes:
            var_inds = []

            node_cost = 1
            for ind in node.indices:
                if ind in free_indices:
                    node_cost *= ind.size
                else:
                    var_inds.append(ind)

            all_var_cost = 0
            if len(var_inds) > 1:
                var_sizes = [ind.value_choices for ind in var_inds]
                for v_sizes in itertools.product(*var_sizes):
                    # we need to add a temporary variable to
                    # turn this term into a linear term
                    y = self.model.addVar(vtype=GRB.BINARY)
                    var_sum = 0
                    var_cost = y
                    for ind, v in zip(var_inds, v_sizes):
                        self.model.addConstr(y <= self.vars[(ind.name, v)])
                        var_sum += self.vars[(ind.name, v)]
                        var_cost *= v

                    self.model.addConstr(y >= var_sum - len(var_inds) + 1)
                    all_var_cost += var_cost

            elif len(var_inds) == 1:
                ind = var_inds[0]
                var_cost = 0
                for v in ind.value_choices:
                    var_cost += v * self.vars[(ind.name, v)]

                all_var_cost += var_cost

            node_cost *= all_var_cost
            cost += node_cost
            # print(cost)

        self.model.addConstr(cost <= upper)
        self.model.setObjective(cost, GRB.MINIMIZE)


class ConstraintSearch:
    """Search rank assignments by constraint solving."""

    def __init__(self, config: SearchConfig):
        self.config = config

        self.split_actions = {}
        self.first_steps = {}
        self.temp_files = []
        self.delta = 0.0

    def abstract(self, s):
        """Separate the given set of singular values into chunks."""
        prev = 0
        prev_sum = 0
        cnt = 0
        s_sizes = [1]
        s_sums = [s[-1] ** 2]

        chunk_size = self.config.synthesizer.bin_size * self.delta**2
        truncation_values = [
            x for x in np.cumsum(np.flip(s) ** 2) if x <= self.delta**2
        ]
        for sv in truncation_values[1:]:
            if sv < prev + chunk_size:
                prev_sum = sv
                cnt += 1
            else:
                prev += chunk_size
                if cnt != 0:
                    s_sums.append(prev_sum)
                    s_sizes.append(cnt)
                prev_sum = sv
                cnt = 1

        if cnt != 0:
            s_sizes.append(cnt)
            s_sums.append(prev_sum)

        # the final sizes need to be accumulated
        final_sizes = [max(len(s) - x, 1) for x in np.cumsum(np.array(s_sizes))]

        # print(s_sizes, list(zip(final_sizes, s_sums)))
        return s_sums, final_sizes

    def _recompute(self, file_name):
        return self.config.preprocess.force_recompute or not os.path.exists(
            file_name
        )

    def _preprocess_cross(
        self, data_tensor: TensorFunc, comb: Sequence[Index]
    ):
        bin_size = self.config.synthesizer.bin_size
        err = self.delta * bin_size
        if isinstance(data_tensor, FuncTensorNetwork):
            net = copy.deepcopy(data_tensor.net)
            net = net.swap(comb)
            # reset the network internal indices to 1
            free_indices = list(net.free_indices())
            for node in net.network:
                tensor = net.node_tensor(node)
                shape = []
                for ind in tensor.indices:
                    if ind in free_indices:
                        shape.append(ind.size)
                    else:
                        shape.append(1)
                tensor.update_val_size(np.zeros(shape))

            ranks_and_errors = net.cross(data_tensor, err).ranks_and_errors
        else:
            net = TreeNetwork()
            net.add_node(
                "G",
                Tensor(
                    np.empty([0 for _ in data_tensor.indices]), data_tensor.indices
                ),
            )
            (_, s, v), _ = OSplit(comb).svd(net, compute_data=False)
            net.merge(v, s, compute_data=False)
            ranks_and_errors = net.cross(data_tensor, err).ranks_and_errors

        sizes, sums = zip(*reversed(ranks_and_errors))
        # print(sizes, sums)

        # pick 10 according to the error change
        bin_num = int(1 / bin_size)
        final_sums, final_sizes = [], []
        prev_idx = -1
        for bin_idx in range(1, bin_num + 1):
            eps = err * bin_idx
            if eps < sums[0] or eps > sums[-1]:
                continue

            min_idx = np.searchsorted(sums, eps) - 1
            if min_idx == prev_idx:
                continue

            final_sums.append(sums[min_idx] ** 2)
            final_sizes.append(sizes[min_idx])
            prev_idx = min_idx

        if prev_idx != len(sizes) - 1:
            final_sums.append(sums[prev_idx + 1] ** 2)
            final_sizes.append(sizes[prev_idx + 1])

        # print(final_sums, final_sizes)

        self.split_actions[OSplit(comb)] = (final_sums, final_sizes)

    def preprocess_tt(self, result: Dict[Sequence[Index], np.ndarray]):
        for comb, s in result.items():
            ac = OSplit(comb)
            sums, sizes = self.abstract(s)
            self.split_actions[ac] = (sums, sizes)

    def preprocess_comb(
        self,
        data_tensor: DataTensor,
        comb: Sequence[Index],
        compute_uv: bool = False,
    ):
        """Precompute the singluar values for a given index combination."""
        logger.debug("preprocess %s", comb)
        logger.debug("%s", data_tensor)

        if isinstance(data_tensor, TensorFunc):
            self._preprocess_cross(data_tensor, comb)
            return

        ac = OSplit(comb)
        if ac in self.split_actions:
            return

        ac.delta = 0.0
        file_name = os.path.join(
            self.config.output.output_dir, f"{len(self.first_steps)}.npz"
        )
        if not self._recompute(file_name):
            data = np.load(file_name)
            s = data["s"]
            self.first_steps[ac] = file_name
        else:
            net = copy.deepcopy(data_tensor)
            # (u, s, v), _ = ac.svd(net, compute_uv=compute_uv)
            # u = net.value(u)
            # s = np.diag(net.value(s))
            # v = net.value(v)
            # if compute_uv:
            #     # save to file to avoid memory explosion
            #     if not os.path.exists(self.config.output.output_dir):
            #         os.makedirs(self.config.output.output_dir)

            #     np.savez(file_name, u=u, s=s, v=v)
            #     self.first_steps[OSplit(comb)] = file_name
            #     self.temp_files.append(file_name)
            s = ac.svals(net)

        sums, sizes = self.abstract(s)
        logger.debug("preprocess: %s, %s", comb, s)
        logger.debug("abstract results: %s, %s", sums, sizes)
        self.split_actions[OSplit(comb)] = (sums, sizes)

    def solve(self, st: SearchState, upper: int) -> Optional[SearchState]:
        """Compute cost for a given set of splits."""
        solver = ILPSolver(self.config)

        pfsums = {}
        # extract nodes from the current network
        relabel_map = {}
        for idx, ac in enumerate(st.past_actions):
            if not isinstance(ac, OSplit):
                index_ac = ac.to_osplit(st, idx)
            else:
                index_ac = ac

            # print(index_ac)
            ac_sums, ac_sizes = self.split_actions[index_ac]
            # print(index_ac, ac_sums, ac_sizes, st.links[idx])
            pfsums[st.links[idx]] = ac_sums
            # we need to substitute the links to all
            relabel_map[st.links[idx]] = tuple(ac_sizes)

        st.network.rerange_indices(relabel_map)
        indices = st.network.all_indices()
        free_indices = st.network.free_indices()
        var_indices = []
        # st.network.draw()
        # plt.show()
        # print(st.network.all_indices())
        rerange_map = {}
        for ind in indices:
            rerange_map[ind.name] = ind.value_choices
            if ind not in free_indices:
                var_indices.append(ind)
                solver.add_var(ind)
        solver.add_constraint(var_indices, pfsums, self.delta)

        nodes = [st.network.node_tensor(n) for n in st.network.network.nodes]
        solver.set_objective(free_indices, nodes, upper)
        solver.model.optimize()

        if solver.model.Status == GRB.INFEASIBLE:
            solver.model.dispose()
            solver.env.dispose()
            return None

        relabel_map = {}
        for ind in var_indices:
            for j in ind.value_choices:
                if solver.vars[(ind.name, j)].x == 1:
                    relabel_map[ind.name] = int(j)

        st.network.relabel_indices(relabel_map)
        st.network.rerange_indices(rerange_map)
        solver.model.dispose()
        solver.env.dispose()
        return st
