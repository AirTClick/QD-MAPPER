"""pyribs-compliant emitters."""
import gin
import ribs

import itertools

import numpy as np

from ribs.archives import AddStatus
from ribs.emitters._emitter_base import EmitterBase
from env_search_PBS.emitters.opt._cma_es import CMAEvolutionStrategy


from env_search_PBS.emitters.map_elites_baseline_emitter import MapElitesBaselineEmitter
from env_search_PBS.emitters.random_emitter import RandomEmitter

__all__ = [
    "GaussianEmitter",
    "ImprovementEmitter",
    "MapElitesBaselineEmitter",
    "RandomEmitter",
]


@gin.configurable
class GaussianEmitter(ribs.emitters.GaussianEmitter):
    """gin-configurable version of pyribs GaussianEmitter."""


@gin.configurable
class ImprovementEmitter(ribs.emitters.ImprovementEmitter):
    """gin-configurable version of pyribs ImprovementEmitter."""
    def __init__(self,
                 archive,
                 x0,
                 sigma0,
                 selection_rule="filter",
                 restart_rule="no_improvement",
                 weight_rule="truncation",
                 bounds=None,
                 batch_size=None,
                 seed=None):
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = sigma0
        EmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

        if restart_rule not in ["basic", "no_improvement"]:
            raise ValueError(f"Invalid restart_rule {restart_rule}")
        self._restart_rule = restart_rule

        opt_seed = None if seed is None else self._rng.integers(10_000)
        # print("Solution dim: ", self._solution_dim)
        # breakpoint()
        self.opt = CMAEvolutionStrategy(sigma0, batch_size, self._solution_dim,
                                        weight_rule, opt_seed,
                                        self.archive.dtype)
        self.opt.reset(self._x0)
        self._num_parents = (self.opt.batch_size //
                             2 if selection_rule == "mu" else None)
        self._batch_size = self.opt.batch_size
        self._restarts = 0  # Currently not exposed publicly.

    def ask(self):
        # Return addition None to cope with parent sol API of
        # MapElitesBaselineWarehouseEmitter
        return super().ask(), None