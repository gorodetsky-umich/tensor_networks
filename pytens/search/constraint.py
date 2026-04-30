"""Linear constraints for finding best rank assignment."""

import copy
import itertools
import logging
import os
from typing import List, Optional, Sequence

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from pytens.algs import Index, Tensor
from pytens.search.configuration import SearchConfig
from pytens.search.state import OSplit, SearchState
from pytens.search.utils import DataTensor
from pytens.types import AlgoParams, SVDAlgorithm, SVDParams

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

        logger.debug("adding coeffs: %s", coeff)
        logger.debug("allowed delta: %s", delta**2)

        # rescale the numbers to avoid overflow
        numbers = [v for v in coeff.values() if v > 1e-8] + [delta**2]
        scale = (max(numbers) ** 0.5) * (min(numbers) ** 0.5)
        if scale == 0:
            scale = 1.0

        for k in coeff:
            coeff[k] /= scale

        self.model.addConstr(
            self.vars.prod(coeff) <= delta**2 / scale, name="total_error"
        )
        # self.model.update()

    def set_objective(
        self,
        free_indices: List[Index],
        nodes: List[Tensor],
        upper: Optional[int],
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

        if upper is not None:
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

        s_sizes, s_sums = [], []
        if include_last:
            s_sizes.append(0)
            s_sums.append(0)

        if len(s) > 1:
            s_sizes.append(1)
            s_sums.append(s[-1] ** 2)

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

        if cnt not in (0, 1):
            s_sizes.append(cnt)
            s_sums.append(prev_sum)

        # the final sizes need to be accumulated
        final_sizes = []
        for x in np.cumsum(np.array(s_sizes)):
            final_sizes.append(max(len(s) - x, 1))

        # print(s_sizes, list(zip(final_sizes, s_sums)))
        return s_sums, final_sizes

    def _recompute(self, file_name):
        return self.config.preprocess.force_recompute or not os.path.exists(
            file_name
        )

    def preprocess_comb(
        self,
        data_tensor: DataTensor,
        comb: Sequence[Index],
        _compute_uv: bool = False,
        cross: bool = False,
    ):
        """Precompute the singluar values for a given index combination."""
        logger.debug("preprocess %s", comb)
        logger.debug("%s", data_tensor)

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
            rand_seed = 42 if self.config.preprocess.rand_svd else None
            s = ac.svals(
                net,
                algo_params=AlgoParams(
                    algo=SVDAlgorithm.SVD if not cross else SVDAlgorithm.CROSS,
                    eps=self.config.engine.eps,
                ),
                svd_params=SVDParams(
                    max_rank=self.config.preprocess.max_rank,
                    orthonormal=None,
                    random_seed=rand_seed,
                ),
            )

        res = self.abstract(s, True)
        if res is not None:
            sums, sizes = res
            logger.debug("preprocess: %s, %s", comb, s)
            logger.debug("abstract results: %s, %s", sums, sizes)
            self.split_actions[OSplit(comb)] = (sums, sizes)
        else:
            logger.debug("no truncation for %s", comb)
            self.split_actions[OSplit(comb)] = ([], [])

    @staticmethod
    def _log_constraints(solver: ILPSolver, solved: bool) -> None:
        """Log constraint values for debugging."""
        for constr in solver.model.getConstrs():
            if solved:
                lhs = solver.model.getRow(constr).getValue()
            else:
                lhs = solver.model.getRow(constr)
            rhs = constr.RHS
            sense = constr.Sense
            logger.debug(
                "Constraint: %s, %s %s %s",
                constr.ConstrName,
                lhs,
                sense,
                rhs,
            )

    def solve(
        self, st: SearchState, upper: Optional[int]
    ) -> Optional[SearchState]:
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

            ac_sums, ac_sizes = self.split_actions[index_ac]
            pfsums[st.links[idx]] = ac_sums
            # we need to substitute the links to all
            relabel_map[st.links[idx]] = tuple(ac_sizes)

        st.network.rerange_indices(relabel_map)
        indices = st.network.all_indices()
        free_indices = st.network.free_indices()
        var_indices = []
        rerange_map = {}
        for ind in indices:
            rerange_map[ind.name] = ind.value_choices
            if ind not in free_indices:
                var_indices.append(ind)
                solver.add_var(ind)
        solver.add_constraint(var_indices, pfsums, self.delta)

        nodes = [st.network.node_tensor(n) for n in st.network.network.nodes]
        solver.set_objective(free_indices, nodes, upper)

        if logger.level == logging.DEBUG:
            logger.debug("constraints to be solved:")
            self._log_constraints(solver, solved=False)

        solver.model.optimize()

        logger.debug("solving result: %s", solver.model.Status)
        if solver.model.Status == GRB.INFEASIBLE:
            solver.model.dispose()
            solver.env.dispose()
            return None

        relabel_map = {}
        for ind in var_indices:
            for j in ind.value_choices:
                if solver.vars[(ind.name, j)].x == 1:
                    relabel_map[ind.name] = int(j)

        logger.debug("feasible rank assignment: %s", relabel_map)
        if logger.level == logging.DEBUG:
            self._log_constraints(solver, solved=True)

        st.network.relabel_indices(relabel_map)
        st.network.rerange_indices(rerange_map)
        solver.model.dispose()
        solver.env.dispose()

        logger.debug(
            "Get cost %s for network %s", st.network.cost(), st.network
        )
        return st
