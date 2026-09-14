"""Rank assignment by integer linear programming.

Given a tree structure produced by a sequence of splits, every internal edge
still needs a rank. For each edge we know the singular values of the
corresponding matricization, so truncating it to a candidate rank costs a
known amount of squared error. The ILP picks one candidate rank per edge
such that the total squared error stays within the budget, and the network
cost (the number of stored entries) is minimized.

The model has, for every internal edge ``e`` and candidate rank ``r``, a
binary variable ``x[e, r]`` and

* one-hot choice:      ``sum_r x[e, r] == 1``               for every ``e``
* error budget:        ``sum_{e, r} err[e, r] * x[e, r] <= delta**2``
* objective:           ``sum_nodes free_size(node) * prod_{e in node} rank(e)``

where ``rank(e) = sum_r r * x[e, r]``. A node touching several internal edges
makes the objective a product of choices, which is linearized with one binary
``y`` per rank combination.
"""

import copy
import functools
import itertools
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from pytens.algs import Index, Tensor, TensorNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.state import ISplit, OSplit, SearchState
from pytens.search.types import Action
from pytens.types import AlgoParams, IndexName, SVDAlgorithm, SValsParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Assignment of a rank to every internal edge, by edge (index) name.
RankAssignment = Dict[IndexName, int]


@functools.lru_cache(maxsize=None)
def _gurobi_env() -> gp.Env:
    """The process-wide Gurobi environment, started on first use.

    Starting an environment is far more expensive than building a model, so
    every ILP shares this one.
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("TimeLimit", 60)
    env.start()
    return env


@dataclass
class RankChoices:
    """Candidate ranks for one edge, with the truncation error of each.

    Attributes:
        ranks: Candidate ranks, largest first.
        errors: Squared error of truncating the edge to the rank at the same
            position, i.e. the sum of the discarded squared singular values.
    """

    ranks: List[int]
    errors: List[float]

    @staticmethod
    def empty() -> "RankChoices":
        """No truncation is possible on this edge."""
        return RankChoices([], [])


def bin_singular_values(
    svals: Sequence[float],
    delta: float,
    bin_size: float,
    include_last: bool = False,
) -> Optional[RankChoices]:
    """Turn a spectrum into a small set of candidate ranks.

    Every prefix of the (reversed) spectrum that fits into the error budget
    ``delta**2`` is a valid truncation. Consecutive truncations whose
    accumulated error falls into the same bin of width
    ``bin_size * delta**2`` are collapsed into one candidate, keeping the
    largest truncation of the bin, so the ILP sees at most ``1 / bin_size``
    candidates per edge.

    Arguments:
        svals: Singular values in descending order.
        delta: Error budget of the whole search step.
        bin_size: Bin width as a fraction of ``delta**2``.
        include_last: Also offer "keep everything" (zero error) as a choice.

    Returns:
        The candidate ranks and their errors, or ``None`` for an empty
        spectrum.
    """
    n = len(svals)
    if n == 0:
        return None

    budget = delta**2
    bin_width = bin_size * budget

    # error[k] is the squared error of discarding the k smallest values
    error = np.concatenate([[0.0], np.cumsum(np.flip(svals) ** 2)])
    max_discard = int(np.searchsorted(error, budget, side="right")) - 1

    # candidates are numbers of discarded values, smallest first
    discards: List[int] = []
    if include_last:
        discards.append(0)
    if n > 1:
        discards.append(1)

    # Group the deeper truncations into bins of increasing error and keep
    # the deepest truncation of every bin. A bin closes at the first
    # truncation whose error reaches the threshold, and that truncation
    # opens the next bin; the last bin only counts if it has two members.
    threshold = bin_width
    bin_members: List[int] = []
    for k in range(2, max_discard + 1):
        if error[k] >= threshold:
            if bin_members:
                discards.append(bin_members[-1])
            threshold += bin_width
            bin_members = []
        bin_members.append(k)

    if len(bin_members) >= 2:
        discards.append(bin_members[-1])

    ranks = [max(n - k, 1) for k in discards]
    errors = [float(error[k]) for k in discards]
    return RankChoices(ranks, errors)


class RankILP:
    """One ILP instance choosing a rank for every internal edge."""

    def __init__(self, env: gp.Env):
        self.model = gp.Model("rank assignment", env=env)
        # x[edge name, rank] binary choice variables
        self.choice: gp.tupledict = gp.tupledict()

    def add_edge(self, edge: Index) -> None:
        """Register an internal edge whose candidate ranks are ``edge.space``.

        Exactly one candidate has to be chosen.
        """
        keys = [(edge.name, rank) for rank in edge.space]
        self.choice.update(self.model.addVars(keys, vtype=GRB.BINARY))
        self.model.addConstr(self.choice.sum(edge.name, "*") == 1)

    def add_error_budget(
        self,
        edges: Sequence[Index],
        errors: Dict[IndexName, Sequence[float]],
        delta: float,
    ) -> None:
        """Bound the total truncation error by ``delta**2``.

        Arguments:
            edges: Internal edges that were registered with ``add_edge``.
            errors: Per edge name, the truncation error of every candidate
                rank in the same order as ``edge.space``.
            delta: The error budget.
        """
        coeff = {}
        for edge in edges:
            assert len(errors[edge.name]) == len(edge.space)
            for rank, err in zip(edge.space, errors[edge.name]):
                coeff[(edge.name, rank)] = err

        budget = delta**2
        logger.debug("adding coeffs: %s", coeff)
        logger.debug("allowed delta: %s", budget)

        # rescale the numbers to avoid overflow
        numbers = [v for v in coeff.values() if v > 1e-8] + [budget]
        scale = (max(numbers) ** 0.5) * (min(numbers) ** 0.5)
        if scale == 0:
            scale = 1.0

        for key in coeff:
            coeff[key] /= scale

        self.model.addConstr(
            self.choice.prod(coeff) <= budget / scale, name="total_error"
        )

    def set_cost_objective(
        self,
        free_indices: Sequence[Index],
        nodes: Sequence[Tensor],
        upper: Optional[int],
    ) -> None:
        """Minimize the number of stored entries over all nodes.

        Arguments:
            free_indices: Indices whose sizes are fixed.
            nodes: The tensors of the network; every index of a node is
                either free or a registered edge.
            upper: If given, only solutions with cost at most ``upper`` are
                feasible.
        """
        cost = gp.LinExpr()
        for node in nodes:
            fixed_size = 1
            edges = []
            for ind in node.indices:
                if ind in free_indices:
                    fixed_size *= ind.size
                else:
                    edges.append(ind)

            cost += fixed_size * self._edge_size(edges)

        if upper is not None:
            self.model.addConstr(cost <= upper)

        self.model.setObjective(cost, GRB.MINIMIZE)

    def _edge_size(self, edges: Sequence[Index]) -> gp.LinExpr:
        """Linear expression for the product of the chosen ranks."""
        if len(edges) == 0:
            return gp.LinExpr()

        if len(edges) == 1:
            edge = edges[0]
            return gp.LinExpr(
                list(edge.space),
                [self.choice[(edge.name, rank)] for rank in edge.space],
            )

        # Each combination of ranks gets a binary y that is forced to 1
        # when all of its ranks are chosen. Since y carries a positive cost
        # in a minimization, y >= sum(x) - (k - 1) alone pins it to the
        # product of the choices at the optimum.
        combos = list(itertools.product(*[edge.space for edge in edges]))
        ys = self.model.addVars(len(combos), vtype=GRB.BINARY)
        slack = len(edges) - 1
        self.model.addConstrs(
            ys[i]
            >= gp.quicksum(
                self.choice[(edge.name, rank)]
                for edge, rank in zip(edges, combo)
            )
            - slack
            for i, combo in enumerate(combos)
        )
        return gp.LinExpr(
            [math.prod(combo) for combo in combos], list(ys.values())
        )

    def solve(self, edges: Sequence[Index]) -> Optional[RankAssignment]:
        """Optimize and read back the chosen rank of every edge.

        Returns ``None`` when no assignment satisfies the constraints. The
        model is disposed either way.
        """
        if logger.level == logging.DEBUG:
            logger.debug("constraints to be solved:")
            self._log_constraints(solved=False)

        self.model.optimize()
        logger.debug("solving result: %s", self.model.Status)

        assignment: Optional[RankAssignment] = None
        if self.model.Status != GRB.INFEASIBLE:
            assignment = {}
            for edge in edges:
                for rank in edge.space:
                    if self.choice[(edge.name, rank)].x == 1:
                        assignment[edge.name] = int(rank)

            logger.debug("feasible rank assignment: %s", assignment)
            if logger.level == logging.DEBUG:
                self._log_constraints(solved=True)

        self.model.dispose()
        return assignment

    def _log_constraints(self, solved: bool) -> None:
        """Log constraint values for debugging."""
        for constr in self.model.getConstrs():
            row = self.model.getRow(constr)
            lhs = str(row.getValue()) if solved else str(row)
            logger.debug(
                "Constraint: %s, %s %s %s",
                constr.ConstrName,
                lhs,
                constr.Sense,
                constr.RHS,
            )


class ConstraintSearch:
    """Search rank assignments by constraint solving.

    ``preprocess_comb`` computes and bins the spectrum of every split the
    enumeration may use; ``solve`` then builds and solves one ILP for a
    concrete sequence of splits.

    Attributes:
        split_actions: Candidate ranks of every preprocessed split.
        first_steps: Files holding the full SVD of a split, when available.
        temp_files: Files to remove after the run.
        delta: Error budget of the current search step.
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self.split_actions: Dict[Action, RankChoices] = {}
        self.first_steps: Dict[Action, str] = {}
        self.temp_files: List[str] = []
        self.delta = 0.0

    def preprocess_comb(
        self,
        data_tensor: TensorNetwork,
        comb: Sequence[Index],
        # precompute UV for ablation (back compatibility)
        _compute_uv: bool = False,
        cross: bool = False,
    ) -> None:
        """Precompute the candidate ranks of splitting off ``comb``."""
        logger.debug("preprocess %s", comb)
        logger.debug("%s", data_tensor)

        action = OSplit(comb)
        if action in self.split_actions:
            return

        action.delta = 0.0
        svals = self._split_svals(data_tensor, action, cross)
        choices = bin_singular_values(
            list(svals),
            self.delta,
            self.config.synthesizer.bin_size,
            include_last=True,
        )
        if choices is None:
            logger.debug("no truncation for %s", comb)
            choices = RankChoices.empty()
        else:
            logger.debug("preprocess: %s, %s", comb, svals)
            logger.debug("abstract results: %s", choices)

        self.split_actions[action] = choices

    def _split_svals(
        self, data_tensor: TensorNetwork, action: OSplit, cross: bool
    ) -> np.ndarray:
        """Singular values of a split, loaded from disk when precomputed."""
        file_name = os.path.join(
            self.config.output.output_dir, f"{len(self.first_steps)}.npz"
        )
        precomputed = not self.config.preprocess.force_recompute and (
            os.path.exists(file_name)
        )
        if precomputed:
            self.first_steps[action] = file_name
            return np.asarray(np.load(file_name)["s"])

        algo = SVDAlgorithm.CROSS if cross else SVDAlgorithm.MERGE
        rand_seed = 42 if self.config.preprocess.rand_svd else None
        return action.svals(
            copy.deepcopy(data_tensor),
            algo_params=AlgoParams(algo=algo, eps=self.config.engine.eps),
            svd_params=SValsParams(
                max_rank=self.config.preprocess.max_rank,
                orthonormal=None,
                random_seed=rand_seed,
            ),
        )

    def solve(
        self, st: SearchState, upper: Optional[int]
    ) -> Optional[SearchState]:
        """Assign the cheapest feasible ranks to the links of ``st``.

        On success the ranks are written into ``st.network`` and ``st`` is
        returned; ``None`` means no assignment fits the error budget (or the
        ``upper`` cost bound).
        """
        choices = self._link_choices(st)

        # every link carries its candidate ranks as its space
        st.network.rerange_indices(
            {link: tuple(c.ranks) for link, c in choices.items()}
        )
        spaces: Dict[IndexName, Sequence[float]] = {}
        for ind in st.network.all_indices():
            spaces[ind.name] = ind.space

        free_indices = st.network.free_indices()
        edges = [
            ind for ind in st.network.all_indices() if ind not in free_indices
        ]
        nodes = [st.network.node_tensor(n) for n in st.network.network.nodes]

        ilp = RankILP(_gurobi_env())
        for edge in edges:
            ilp.add_edge(edge)
        ilp.add_error_budget(
            edges, {link: c.errors for link, c in choices.items()}, self.delta
        )
        ilp.set_cost_objective(free_indices, nodes, upper)
        assignment = ilp.solve(edges)

        if assignment is None:
            return None

        st.network.relabel_indices(assignment)
        st.network.rerange_indices(spaces)
        logger.debug(
            "Get cost %s for network %s", st.network.cost(), st.network
        )
        return st

    def _link_choices(self, st: SearchState) -> Dict[IndexName, RankChoices]:
        """Candidate ranks of every link created by the past actions."""
        choices = {}
        for idx, action in enumerate(st.past_actions):
            if isinstance(action, ISplit):
                split = action.to_osplit(st, idx)
            elif isinstance(action, OSplit):
                split = action
            else:
                raise TypeError(f"Unsupported action type: {type(action)}")

            choices[st.links[idx]] = self.split_actions[split]

        return choices
