"""Linear constraints for finding best rank assignment."""

from typing import List, Dict
import itertools
import time
import os

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

from pytens.algs import Tensor, Index
from pytens.search.state import SplitIndex, SearchState

BAD_SCORE = 9999999999999

class ILPSolver:
    """An ILP solver to find near-optimal rank assignments."""
    def __init__(self, params: Dict):
        self.params = params
        self.model = gp.Model("A model")
        self.vars = gp.tupledict()

    def add_var(self, ind: Index):
        """Add variables for a given rank i"""
        # for a given edge, we add binary variables i0, i1, .., in
        indices = [(ind.name, j) for j in ind.size]
        # print(indices)
        # print(ind, len(indices), ind.size[1] - ind.size[0])
        self.vars.update(self.model.addVars(indices, vtype=GRB.BINARY))

    def add_constraint(self, inds: List[Index], prefix_sums, delta: float):
        """Given n ranks to be solved, generate all constraints

        Constr1: sum_j xij = 1
        Constr2: sum_ij xij*pij <= delta**2

        Arguments:
            n - Number of ranks to be resolved
            prefix_sums - Prefix sums of the singular values for corresponding edge
            delta - The maximum error can be accumulated
        """
        coeff = {}
        for ind in inds:
            self.model.addConstr(self.vars.sum(ind.name, '*') == 1)

            # print(ind)
            # print(ind, len(prefix_sums[ind.name]))
            assert len(prefix_sums[ind.name]) == len(ind.size)
            for sz, p in zip(ind.size, prefix_sums[ind.name]):
                coeff[(ind.name, sz)] = p

        self.model.addConstr(self.vars.prod(coeff) <= delta ** 2)

    def set_objective(self, free_indices: List[Index], nodes: List[Tensor], upper: int):
        """Set the objective for the solver."""
        # max_cost = np.prod([i.size for i in free_indices])
        cost = 0
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
                    # we need to add a temporary variable to turn this term into a linear term
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
    """Search structures ordered by costs that are computed from constraint solving."""
    def __init__(self, params):
        self.params = params

        self.split_actions = {}
        self.first_steps = {}
        self.delta = 0

    def abstract(self, s):
        """Separate the given set of singular values into chunks."""
        prev = 0
        prev_sum = 0
        cnt = 0
        s_sizes = [1]
        s_sums = [s[-1]**2]

        chunk_size = self.params["bin_size"] * self.delta ** 2
        truncation_values = [x for x in np.cumsum(np.flip(s) ** 2) if x <= self.delta ** 2]
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
        final_sizes = [len(s)-x for x in np.cumsum(np.array(s_sizes))]

        # print(list(zip(final_sizes, s_sums)))
        return s_sums, final_sizes

    def preprocess_comb(self, target_tensor: Tensor, comb: List[int], compute_uv = False):
        """Precompute the singluar values for a given combination of indices."""
        free_indices = target_tensor.indices
        right_indices = [i for i in free_indices if i not in comb]
        positions = [target_tensor.indices.index(i) for i in list(comb) + right_indices]
        tensor_val = target_tensor.value.transpose(positions)
        left_size = np.prod([x.size for x in comb])
        # u, s, v = np.linalg.svd(tensor_val.reshape(left_size, -1), False, True)
        if compute_uv:
            u, s, v = np.linalg.svd(tensor_val.reshape(left_size, -1), False, True)
            # save to file to avoid memory explosion
            file_name = f"{self.params['output_dir']}/{len(self.first_steps)}.npz"
            np.savez(file_name, u=u,s=s,v=v)
            self.first_steps[SplitIndex(comb)] = file_name
        else:
            file_name = f"{self.params['output_dir']}/{len(self.first_steps)}.npz"
            if os.path.exists(file_name):
                data = np.load(file_name)
                s = data['s']
                self.first_steps[SplitIndex(comb)] = file_name
            else:
                s = np.linalg.svd(tensor_val.reshape(left_size, -1), False, False)
            sums, sizes = self.abstract(s)
            self.split_actions[SplitIndex(comb)] = (sums, sizes)

    def preprocess(self, target_tensor: Tensor, acs = None, compute_uv = False):
        """Compute the mapping between splits and singular values.
        
        Build the abstractions.
        """
        free_indices = target_tensor.indices
        x_norm = np.linalg.norm(target_tensor.value)
        self.delta = self.params["eps"] * x_norm
        if acs is not None:
            for ac in acs:
                comb = ac.indices
                self.preprocess_comb(target_tensor, comb)
        else:
            for k in range(1, len(free_indices) // 2 + 1):
                combs = list(itertools.combinations(free_indices, k))
                if len(free_indices) % 2 == 0 and k == len(free_indices) // 2:
                    combs = combs[:len(combs) // 2]

                for comb in combs:
                    self.preprocess_comb(target_tensor, comb, compute_uv=compute_uv)
                    # self.first_steps[SplitIndex(comb)] = (u, s, v)

    # we can integrate this with A*, beam search, or other things
    # let's try A* first
    def get_cost(self, st: SearchState, upper: int):
        """Compute cost for a given set of splits."""
        solver = ILPSolver({})
        solver.model.params.OutputFlag = 0
        solver.model.params.TimeLimit = 60
        
        prefix_sums = {}
        # extract nodes from the current network
        relabel_map = {}
        for idx, ac in enumerate(st.past_actions):
            # print(ac)
            if not isinstance(ac, SplitIndex):
                index_ac = ac.to_index(st, idx)
            else:
                index_ac = ac
                
            # print(index_ac)
            ac_sums, ac_sizes = self.split_actions[index_ac]
            prefix_sums[st.links[idx]] = ac_sums
            # we need to substitute the links to all 
            relabel_map[st.links[idx]] = tuple(ac_sizes)
        
        st.network.relabel_indices(relabel_map)
        indices = st.network.all_indices()
        free_indices = st.network.free_indices()
        var_indices = []
        # st.network.draw()
        # plt.show()
        for ind in indices:
            if ind not in free_indices:
                var_indices.append(ind)
                solver.add_var(ind)
        solver.add_constraint(var_indices, prefix_sums, self.delta)

        nodes = []
        for _, data in st.network.network.nodes(data=True):
            nodes.append(data["tensor"])

        solver.set_objective(free_indices, nodes, upper)
        solver.model.optimize()

        try:
            relabel_map = {}
            for ind in var_indices:
                for j in ind.size:
                    if solver.vars[(ind.name, j)].x == 1:
                        relabel_map[ind.name] = j

            st.network.relabel_indices(relabel_map)
            # st.network.compress()
            # if st.network.cost() < 380000:
            #     st.network.draw()
            #     plt.savefig(f"{'_'.join([''.join([ind.name for ind in ac.indices]) for ac in st.past_actions])}.png")
            #     plt.close()
            result = {}
            for ind, ind_size in relabel_map.items():
                for k, v in enumerate(st.links):
                    if v == ind:
                        result[k] = ind_size
                        break
            return result, st.network.cost()
        except AttributeError:
            return {}, BAD_SCORE

def test_case_1():
    """Test case for tensor trains with 10-20-8 indices."""
    i = 10
    j = 20
    k = 8
    x = np.random.randn(i, j, k)
    x_indices = [Index("i", i), Index("j", j), Index("k", k)]

    delta = np.linalg.norm(x) * 0.5

    r1_vals = np.linalg.svdvals(x.reshape(i, j * k))
    r1_ps = [x for x in np.cumsum(np.flip(r1_vals) ** 2) if x <= delta ** 2]
    r2_vals = np.linalg.svdvals(x.reshape(i * j, k))
    r2_ps = [x for x in np.cumsum(np.flip(r2_vals) ** 2) if x <= delta ** 2]
    print(r1_ps, r2_ps, delta**2)

    r1 = Index("r1", (min(i, j*k) - len(r1_ps), min(i, j*k)+1))
    x1 = Tensor(None, [Index("i", i), r1])
    r2 = Index("r2", (min(i*j, k) - len(r2_ps), min(i*j, k)+1))
    x2 = Tensor(None, [r1, Index("j", j), r2])
    x3 = Tensor(None, [r2, Index("k", k)])

    solver = ILPSolver({})
    prefix_sums = {"r1": r1_ps, "r2": r2_ps}
    solver.add_var(r1)
    solver.add_var(r2)
    solver.add_constraint([r1,r2], prefix_sums, delta)
    solver.set_objective(x_indices, [x1, x2, x3])
    solver.model.optimize()

    for ind in [r1, r2]:
        for j in range(*ind.size):
            print(ind, j, solver.vars[(ind.name, j)])

def test_case_2():
    """Test case for real data with tensor trains format."""
    x = np.load("data/BigEarthNet-v1_0_stack/stack_18_test_2/data.npy")
    x_indices = [
        Index("I0", 18),
        Index("I1", 120),
        Index("I2", 120),
        Index("I3", 12),
    ]
    delta = np.linalg.norm(x) * 0.1

    # split into tt
    s0 = np.linalg.svdvals(x.reshape(18, -1))
    r0_ps = [s for s in np.cumsum(np.flip(s0) ** 2) if s <= delta ** 2]
    r0 = Index("r0", (18-len(r0_ps), 18))

    s1 = np.linalg.svdvals(x.reshape(18*120, -1))
    r1_ps = [s for s in np.cumsum(np.flip(s1) ** 2) if s <= delta ** 2]
    r1 = Index("r1", (12*120-len(r1_ps), 12*120))

    s2 = np.linalg.svdvals(x.reshape(18*120*120, -1))
    r2_ps = [s for s in np.cumsum(np.flip(s2) ** 2) if s <= delta ** 2]
    r2 = Index("r2", (12-len(r2_ps), 12))

    nodes = [
        Tensor(None, [x_indices[0], r0]),
        Tensor(None, [r0, x_indices[1], r1]),
        Tensor(None, [r1, x_indices[2], r2]),
        Tensor(None, [r2, x_indices[3]]),
    ]

    solver = ILPSolver({})
    prefix_sums = {"r0": r0_ps, "r1": r1_ps, "r2": r2_ps}
    solver.add_var(r0)
    solver.add_var(r1)
    solver.add_var(r2)
    solver.add_constraint([r0, r1, r2], prefix_sums, delta)
    solver.set_objective(x_indices, nodes)
    solver.model.params.OutputFlag = 0
    solver.model.optimize()

    for ind in [r0, r1, r2]:
        for j in range(*ind.size):
            if solver.vars[(ind.name, j)].x == 1:
                print(ind, j)

def test_case_3():
    """Test case for real data with tree structures. (and abstraction!)"""
    x = np.load("data/BigEarthNet-v1_0_stack/stack_18_test_2/data.npy")
    x_indices = [
        Index("I0", 18),
        Index("I1", 120),
        Index("I2", 120),
        Index("I3", 12),
    ]
    x_norm = np.linalg.norm(x)
    delta = x_norm * 0.1

    def abstract(s):
        prev = 0
        prev_sum = 0
        cnt = 0
        s_sizes = [0]
        s_sums = [0]
        for sv in np.cumsum(np.flip(s) ** 2):
            if sv < prev + 0.0001 * x_norm ** 2:
                prev_sum = sv
                cnt += 1
            else:
                prev = prev + 0.0001 * x_norm ** 2
                s_sums.append(prev_sum)
                s_sizes.append(cnt)
                prev_sum = sv
                cnt = 1

        if cnt != 0:
            s_sizes.append(cnt)
            s_sums.append(prev_sum)

        return s_sums, s_sizes

    # split into ht
    # s0 = np.linalg.svdvals(x.reshape(18, -1))
    # # r0_ps = [s for s in np.cumsum(np.flip(s0) ** 2) if s <= delta ** 2]
    # r0_ps, r0_sizes = abstract(s0)
    # r0_sizes = [18-x for x in np.flip(np.cumsum(np.array(r0_sizes)))]
    # r0 = Index("r0", r0_sizes)

    s1 = np.linalg.svdvals(x.transpose(1,0,2,3).reshape(120, -1))
    # r1_ps = [s for s in np.cumsum(np.flip(s1) ** 2) if s <= delta ** 2]
    # r1 = Index("r1", (120-len(r1_ps), 120+1))
    r1_ps, r1_sizes = abstract(s1)
    r1_sizes = [120-x for x in np.flip(np.cumsum(np.array(r1_sizes)))]
    r1 = Index("r1", r1_sizes)

    s2 = np.linalg.svdvals(x.transpose(2,0,1,3).reshape(120, -1))
    # r2_ps = [s for s in np.cumsum(np.flip(s2) ** 2) if s <= delta ** 2]
    # r2 = Index("r2", (120-len(r2_ps), 120+1))
    r2_ps, r2_sizes = abstract(s2)
    r2_sizes = [120-x for x in np.flip(np.cumsum(np.array(r2_sizes)))]
    r2 = Index("r2", r2_sizes)

    # s3 = np.linalg.svdvals(x.transpose(3,0,1,2).reshape(12, -1))
    # # r3_ps = [s for s in np.cumsum(np.flip(s3) ** 2) if s <= delta ** 2]
    # # r3 = Index("r3", (12-len(r3_ps), 12+1))
    # r3_ps, r3_sizes = abstract(s3)
    # r3_sizes = [12-x for x in np.flip(np.cumsum(np.array(r3_sizes)))]
    # r3 = Index("r3", r3_sizes)

    s4 = np.linalg.svdvals(x.transpose(0,3,1,2).reshape(18*12, -1))
    # r4_ps = [s for s in np.cumsum(np.flip(s4) ** 2) if s <= delta ** 2]
    # r4 = Index("r4", (18*12-len(r4_ps), 18*12+1))
    r4_ps, r4_sizes = abstract(s4)
    r4_sizes = [18*12-x for x in np.flip(np.cumsum(np.array(r4_sizes)))]
    r4 = Index("r4", r4_sizes)

    nodes = [
        # Tensor(None, [x_indices[0], r0]),
        Tensor(None, [x_indices[1], r1]),
        Tensor(None, [x_indices[2], r2]),
        # Tensor(None, [x_indices[3], r3]),
        # Tensor(None, [r0, r1, r4]),
        # Tensor(None, [r4, r2, r3]),
        Tensor(None, [r1, r2, r4]),
        Tensor(None, [x_indices[0], x_indices[3], r4]),
    ]

    solver = ILPSolver({})
    # prefix_sums = {"r0": r0_ps, "r1": r1_ps, "r2": r2_ps, "r3": r3_ps, "r4": r4_ps}
    # ranks = [r0, r1, r2, r3, r4]
    prefix_sums = {"r1": r1_ps, "r2": r2_ps, "r4": r4_ps}
    ranks = [r1, r2, r4]
    for r in ranks:
        solver.add_var(r)
    solver.add_constraint(ranks, prefix_sums, delta)
    solver.set_objective(x_indices, nodes)
    # solver.model.params.OutputFlag = 0
    solver.model.optimize()

    for ind in ranks:
        for j in ind.size:
            if solver.vars[(ind.name, j)].x == 1:
                print(ind.name, j)


if __name__ == "__main__":
    # construct a tensor network and try tensor train solving
    import numpy as np

    test_case_3()
