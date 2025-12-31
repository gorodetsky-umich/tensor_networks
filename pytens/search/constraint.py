"""Linear constraints for finding best rank assignment."""

import copy
import itertools
import logging
import os
from typing import List, Optional, Sequence

import gurobipy as gp
import numpy as np
import tntorch
import torch
from gurobipy import GRB

from pytens.algs import Index, Tensor, TensorTrain
from pytens.cross.funcs import FuncTensorNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.hierarchical.utils import tntorch_wrapper
from pytens.search.state import OSplit, SearchState
from pytens.search.utils import DataTensor, reshape_func
from pytens.types import IndexMerge, SVDAlgorithm

BAD_SCORE = 9999999999999

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

        logger.trace("adding coeffs: %s", coeff)
        logger.trace("allowed delta: %s", delta ** 2)

        # rescale the numbers to avoid overflow
        numbers = list(coeff.values()) + [delta ** 2]
        scale = max(numbers) - min(numbers)
        if scale == 0:
            scale = 1.0

        for k in coeff:
            coeff[k] /= scale

        self.model.addConstr(self.vars.prod(coeff) <= delta**2 / scale, name="total_error")
        # self.model.update()

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

    def abstract(self, s, include_last: bool = False):
        """Separate the given set of singular values into chunks."""
        prev = 0
        prev_sum = 0
        cnt = 0
        if len(s) == 0:
            return None

        if include_last:
            s_sizes = [0, 1]
            s_sums = [0, s[-1] ** 2]
        else:
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

        if cnt != 0 and cnt != 1:
            s_sizes.append(cnt)
            s_sums.append(prev_sum)

        # the final sizes need to be accumulated
        final_sizes = [
            max(len(s) - x, 1) for x in np.cumsum(np.array(s_sizes))
        ]

        # print(s_sizes, list(zip(final_sizes, s_sums)))
        return s_sums, final_sizes

    def _recompute(self, file_name):
        return self.config.preprocess.force_recompute or not os.path.exists(
            file_name
        )

    def _preprocess_cross(
        self, data_tensor: DataTensor, comb: Sequence[Index]
    ):
        bin_size = self.config.synthesizer.bin_size
        err = self.delta * bin_size
        assert isinstance(data_tensor, TensorTrain)
        # create a merge func that merges comb into one index and the rest into the other
        data_indices = data_tensor.free_indices()

        comb_size = int(np.prod([i.size for i in comb]))
        comb_merge = IndexMerge(
            indices=comb,
            result=Index(
                "_".join([str(i.name) for i in comb]),
                comb_size,
                range(comb_size),
            ),
        )

        other_indices = [ind for ind in data_indices if ind not in comb]
        other_size = int(np.prod([i.size for i in other_indices]))
        other_merge = IndexMerge(
            indices=other_indices,
            result=Index(
                "_".join([str(i.name) for i in other_indices]),
                other_size,
                range(other_size),
            ),
        )
        merge_func = reshape_func(
            [comb_merge, other_merge],
            FuncTensorNetwork(data_tensor.free_indices(), data_tensor),
        )

        # get ranks and errors for the comb
        domains = [torch.arange(s) for s in [comb_size, other_size]]
        _, info = tntorch.cross(
            tntorch_wrapper(merge_func),
            domains,
            eps=err,
            kickrank=1,
            max_iter=self.config.preprocess.max_rank,
            verbose=False,
            return_info=True,
        )
        errors = info["epss"]
        # TODO: process the errors
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

    # def preprocess_tt(self, result: Dict[Sequence[Index], np.ndarray]):
    #     for comb, s in result.items():
    #         ac = OSplit(comb)
    #         sums, sizes = self.abstract(s)
    #         self.split_actions[ac] = (sums, sizes)

    def preprocess_comb(
        self,
        data_tensor: DataTensor,
        comb: Sequence[Index],
        compute_uv: bool = False,
        cross: bool = False,
    ):
        """Precompute the singluar values for a given index combination."""
        logger.debug("preprocess %s", comb)
        logger.debug("%s", data_tensor)

        if cross:
            self._preprocess_cross(
                FuncTensorNetwork(data_tensor.free_indices(), data_tensor),
                comb,
            )
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
            s = ac.svals(
                net,
                max_rank=self.config.preprocess.max_rank,
                rand=self.config.preprocess.rand_svd,
                eps=self.config.engine.eps,
                algo=SVDAlgorithm.SVD,
            )

        logger.debug("get singular values with sum: %s, net norm: %s", sum(s ** 2), net.norm() ** 2)
        res = self.abstract(s, True)
        if res is not None:
            sums, sizes = res
            logger.debug("preprocess: %s, %s", comb, s)
            logger.debug("abstract results: %s, %s", sums, sizes)
            self.split_actions[OSplit(comb)] = (sums, sizes)
        else:
            logger.debug("no truncation for %s", comb)
            self.split_actions[OSplit(comb)] = ([], [])

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

        logger.trace("constraints to be solved:")
        if logger.level == logging.TRACE:
            for constr in solver.model.getConstrs():
                # Get the value of the LHS expression in the current solution
                lhs = solver.model.getRow(constr)
                # Get the RHS value
                rhs = constr.RHS
                # Get the constraint sense
                sense = constr.Sense
                logger.trace("Constraint: %s, %s %s %s", constr.ConstrName, lhs, sense, rhs)

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

        logger.trace("feasible rank assignment: %s", relabel_map)
        if logger.level == logging.TRACE:
            for constr in solver.model.getConstrs():
                # Get the value of the LHS expression in the current solution
                lhs_value = solver.model.getRow(constr).getValue()
                # Get the RHS value
                rhs_value = constr.RHS
                # Get the constraint sense
                sense = constr.Sense
                logger.trace("Constraint: %s, %s %s %s", constr.ConstrName, lhs_value, sense, rhs_value)

        st.network.relabel_indices(relabel_map)
        st.network.rerange_indices(rerange_map)
        solver.model.dispose()
        solver.env.dispose()
        return st
