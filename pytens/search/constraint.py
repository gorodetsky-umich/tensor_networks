"""Linear constraints for finding best rank assignment."""

from typing import List, Sequence, Optional, Tuple, Dict
import itertools
import os

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from pytens.algs import Tensor, Index, TreeNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.state import OSplit, SearchState
from pytens.cross.cross import TensorFunc

BAD_SCORE = 9999999999999


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
        indices = [(ind.name, j) for j in ind.size]
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
            assert len(pfsums[ind.name]) == len(ind.size)
            for sz, p in zip(ind.size, pfsums[ind.name]):
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
                var_sizes = [ind.size for ind in var_inds]
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
                for v in ind.size:
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
        self.delta = 0

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
        final_sizes = [len(s) - x for x in np.cumsum(np.array(s_sizes))]

        # print(list(zip(final_sizes, s_sums)))
        return s_sums, final_sizes

    def _preprocess_cross(
        self,
        target_tensor: Tensor,
        positions: Sequence[int],
        comb: Sequence[Index],
        cross_func: TensorFunc,
    ):
        # print(comb)
        net = TreeNetwork()
        net.add_node("G", target_tensor)
        bin_size = self.config.synthesizer.bin_size
        err = self.config.engine.eps * bin_size
        _, _, st = net.cross(cross_func, {}, "G", positions[: len(comb)], err)
        sizes, sums = zip(*st.ranks_and_errors)
        # print(st.ranks_and_errors)

        # for each error, find the smallest rank when there are several close

        # pick 10 according to the error change
        sizes = np.array(sizes)
        sums = np.array(sums)
        bin_num = int(1 / bin_size)
        final_sums, final_sizes = [], []
        for bin_idx in range(1, bin_num + 1):
            eps = err * bin_idx
            deviation = eps - sums
            # print(eps, deviation)
            # find the indices where after it the deviation are mostly positive
            cnts = np.convolve(
                deviation >= 0,
                np.ones(min(3, len(deviation)), dtype=int),
                "same",
            )
            pos_cnt = cnts >= 0
            within_range = np.logical_and(deviation >= 0, deviation <= err)
            valid_errors = np.where(np.logical_and(within_range, pos_cnt))[0]
            # print(eps, sums[valid_errors], sizes[valid_errors])
            if len(valid_errors) > 0:
                min_idx = np.argmin(sizes[valid_errors])
                final_sums.append(sums[valid_errors][min_idx] ** 2)
                final_sizes.append(sizes[valid_errors][min_idx])
                # print(final_sums, final_sizes)

        # print(comb, final_sums, final_sizes)
        self.split_actions[OSplit(comb)] = (final_sums, final_sizes)

    def preprocess_comb(
        self,
        target_tensor: Tensor,
        comb: Sequence[Index],
        compute_uv: bool = False,
        cross_func: Optional[TensorFunc] = None,
    ):
        """Precompute the singluar values for a given index combination."""
        free_indices = target_tensor.indices
        right_indices = [i for i in free_indices if i not in comb]
        positions = []
        for i in list(comb) + right_indices:
            positions.append(target_tensor.indices.index(i))

        tensor_val = target_tensor.value.transpose(positions)
        left_size = np.prod([x.size for x in comb])
        file_name = (
            f"{self.config.output.output_dir}/{len(self.first_steps)}.npz"
        )
        # if OSplit(comb) in self.split_actions:
        #     print("skipping", [str(ac) for ac in comb])
        #     return
        # else:
        #     print("not skipping", [str(ac) for ac in comb])

        if cross_func is not None:
            self._preprocess_cross(target_tensor, positions, comb, cross_func)
            return

        if compute_uv:
            u, s, v = np.linalg.svd(
                tensor_val.reshape(left_size, -1), False, True
            )
            # save to file to avoid memory explosion
            if not os.path.exists(self.config.output.output_dir):
                os.makedirs(self.config.output.output_dir)

            np.savez(file_name, u=u, s=s, v=v)
            self.first_steps[OSplit(comb)] = file_name
            self.temp_files.append(file_name)
        else:
            if not self.config.preprocess.force_recompute and os.path.exists(
                file_name
            ):
                data = np.load(file_name)
                s = data["s"]
                self.first_steps[OSplit(comb)] = file_name
            else:
                s = np.linalg.svd(
                    tensor_val.reshape(left_size, -1), False, False
                )
            sums, sizes = self.abstract(s)
            self.split_actions[OSplit(comb)] = (sums, sizes)

    def preprocess(
        self,
        target_tensor: Tensor,
        acs: Optional[Sequence[OSplit]] = None,
        compute_uv: bool = False,
        delta: Optional[float] = None,
        cross_func: Optional[TensorFunc] = None,
    ):
        """Compute the mapping between splits and singular values.

        Build the abstractions.
        """
        free_indices = target_tensor.indices
        if delta is not None:
            self.delta = delta
        elif cross_func is not None:
            self.delta = self.config.engine.eps
        else:
            x_norm = np.linalg.norm(target_tensor.value)
            self.delta = float(self.config.engine.eps * x_norm)

        if acs is not None:
            for ac in acs:
                comb = ac.indices
                self.preprocess_comb(
                    target_tensor, comb, cross_func=cross_func
                )
        else:
            for comb in SearchState.all_index_combs(free_indices):
                self.preprocess_comb(
                    target_tensor,
                    comb,
                    compute_uv=compute_uv,
                    cross_func=cross_func,
                )
                # self.first_steps[OSplit(comb)] = (u, s, v)

    # we can integrate this with A*, beam search, or other things
    # let's try A* first
    def get_cost(
        self, st: SearchState, upper: int
    ) -> Tuple[Dict[int, int], int]:
        """Compute cost for a given set of splits."""
        solver = ILPSolver(self.config)

        pfsums = {}
        # extract nodes from the current network
        relabel_map = {}
        for idx, ac in enumerate(st.past_actions):
            # print(ac)
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

        st.network.relabel_indices(relabel_map)
        indices = st.network.all_indices()
        free_indices = st.network.free_indices()
        var_indices = []
        # st.network.draw()
        # plt.show()
        # print(st.network.all_indices())
        for ind in indices:
            if ind not in free_indices:
                var_indices.append(ind)
                solver.add_var(ind)
        solver.add_constraint(var_indices, pfsums, self.delta)

        nodes = [
            data["tensor"] for _, data in st.network.network.nodes(data=True)
        ]
        solver.set_objective(free_indices, nodes, upper)
        solver.model.optimize()

        if solver.model.Status == GRB.INFEASIBLE:
            solver.model.dispose()
            solver.env.dispose()
            return {}, BAD_SCORE

        relabel_map, error_map = {}, {}
        for ind in var_indices:
            for jj, j in enumerate(ind.size):
                if solver.vars[(ind.name, j)].x == 1:
                    relabel_map[ind.name] = j
                    error_map[ind.name] = jj

        st.network.relabel_indices(relabel_map)
        # st.network.compress()
        result = {}
        for ind, ind_size in relabel_map.items():
            for k, v in enumerate(st.links):
                if v == ind:
                    result[k] = ind_size
                    break

        solver.model.dispose()
        solver.env.dispose()
        return result, st.network.cost()
