# Tree cross approximation

`CrossApproximation.cross` updates a connected tree network in place. Function
indices must match the network's physical indices. `validation`, `initialization`,
and `known` contain discrete indices, with columns in `tensor_func.indices` order.

```python
config = CrossConfig(
    max_iters=25,
    max_rank=64,
    batch_size=8192,
    cache_size=100_000,  # opt in for deterministic targets
    atol=0.0,
)
result = CrossApproximation(function, config).cross(network, eps=1e-6)
if not result.converged:
    print(result.stop_reason, result.history[-1].validation_error)
```

Accuracy is checked after every complete sweep using
`RMSE(target - estimate) <= atol + eps * RMS(target)`. Validation points are
sampled once unless supplied explicitly, and target values are evaluated once.
This is sampled evidence of accuracy, not a guarantee on unseen tensor entries.
`validation_error` is relative RMS error for nonzero targets and absolute RMSE
when every validation target value is zero.

The default convergence mode is `VALID_ERROR`. `NORM` additionally computes the
full iterate-change norm, at the cost of a network copy and contractions; it
never treats a stationary inaccurate approximation as successful. `max_iters`
is now a positive, exact sweep limit (default 25); `None` and zero are invalid.
Rank growth and the MAXVOL/DEIM selection algorithms are otherwise unchanged.
When fixed ranks and validation predictions remain stationary for
`stagnation_patience` sweeps, the run stops without claiming convergence.

`result.converged` indicates whether validation passed. `stop_reason` is one of
`validation`, `max_iters`, `rank_limit`, or `stagnated`. `history` contains one
record per sweep, including the edge ranks in tree preorder (excluding the root),
validation error, RMSE, optional norm change, and cumulative oracle evaluations.
The compatibility field `ranks_and_errors` now contains one
`(maximum_edge_rank, validation_error)` pair per sweep, preserving repeated ranks.

`batch_size` bounds oracle batches, validation contractions, and the temporary
coordinate arrays for sampled matrices. It does not bound tensor-core storage or
the sampled value matrices needed by QR/SVD.

Caching is disabled by default. With `cache_size > 0`, batched lookups use sorted
NumPy key/value arrays with LRU eviction; duplicate misses within a batch are
evaluated once. Keys are exact mixed-radix uint64 indices when the domain fits,
or fixed-width coordinate bytes for larger domains. They are not probabilistic
hashes. All-hit batches skip miss deduplication and cache rebuilding. Recency uses
the last occurrence of each point in the requested batch, including duplicates.
The cache is cleared at the beginning of every run, so reusing an engine after
changing its function does not reuse stale results. Targets must remain
deterministic and unchanged during each cached run. The legacy `CachedFunc` class
still tracks unique calls; actual memoization is provided by this run-local
evaluation layer. Keep caching off for very cheap vectorized functions: even
batched cache lookups have a cost, and fewer evaluations do not guarantee less
wall-clock time.

`result.evaluations` reports requested points, actual evaluated points, cache
hits (including within-batch reuse), and oracle evaluation time, including
validation. These are point counts, not Python-call counts.

`TTCrossRunner`, `HTCrossRunner`, and `TuckerCrossRunner` accept an optional
`CrossConfig` at construction. Their `run` methods still return a network;
`last_result` exposes status and statistics. They issue a `RuntimeWarning` when
the tolerance is unmet. An explicit `run(..., kickrank=...)` overrides the config.
