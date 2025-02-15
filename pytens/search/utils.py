"""Utility function for structure search."""

import glob
import os

import numpy as np

from pytens.search.state import SearchState
from pytens.algs import TensorNetwork, Tensor

EMPTY_SEARCH_STATS = {
    "networks": [],
    "best_networks": [],
    "best_cost": [],
    "costs": [],
    "errors": [],
    "ops": [],
    "unique": {},
    "count": 0,
}


def approx_error(tensor: Tensor, net: TensorNetwork) -> float:
    """Compute the reconstruction error.

    Given a tensor network TN and the target tensor X,
    it returns ||X - TN|| / ||X||.
    """
    target_free_indices = tensor.indices
    net_free_indices = net.free_indices()
    net_value = net.contract().value
    perm = [net_free_indices.index(i) for i in target_free_indices]
    net_value = net_value.transpose(perm)
    error = float(
        np.linalg.norm(net_value - tensor.value) / np.linalg.norm(tensor.value)
    )
    return error


def log_stats(
    search_stats: dict,
    target_tensor: np.ndarray,
    ts: float,
    st: SearchState,
    bn: TensorNetwork,
):
    """Log statistics of a given state."""
    search_stats["ops"].append((ts, len(st.past_actions)))
    search_stats["costs"].append((ts, st.network.cost()))
    err = approx_error(target_tensor, st.network)
    search_stats["errors"].append((ts, err))
    search_stats["best_cost"].append((ts, bn.cost()))
    ukey = st.network.canonical_structure()
    search_stats["unique"][ukey] = search_stats["unique"].get(ukey, 0) + 1


def remove_temp_dir(temp_dir):
    """Remove temporary npz files"""
    try:
        for npfile in glob.glob(f"{temp_dir}/*.npz"):
            os.remove(npfile)

        if len(os.listdir(temp_dir)) == 0:
            os.rmdir(temp_dir)

    except FileNotFoundError:
        pass
