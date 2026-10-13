from dataclasses import dataclass, field

from chemprop.data.datasets import MoleculeDataset
from chemprop.featurizers.base import Featurizer
from chemprop.featurizers.molgraph.cache import MolGraphCache, MolGraphCacheOnTheFly
from chemprop_contrib.polymers.data.datapoints import PolymerDatapoint
from chemprop_contrib.polymers.data.molgraph import WeightedMolGraph
from chemprop_contrib.polymers.featurizers import PolymerMolGraphFeaturizer
from chemprop_contrib.polymers.polymer_types import Polymer


@dataclass
class PolymerDataset(MoleculeDataset):
    r"""A :class:`PolymerDataset` composed of :class:`PolymerDatapoint`\s

    A :class:`PolymerDataset` produces featurized data for input to a
    :class:`wMPNN` model. Typically, data featurization is performed on-the-fly
    and parallelized across multiple workers via the :class:`~torch.utils.data
    DataLoader` class. However, for small datasets, it may be more efficient to
    featurize the data in advance and cache the results. This can be done by
    setting ``PolymerDataset.cache=True``.

    Parameters
    ----------
    data : Iterable[PolymerDatapoint]
        the data from which to create a dataset
    featurizer : Featurizer[Polymer, WeightedMolGraph]
        the featurizer with which to generate WeightedMolGraphs of the polymers
    n_workers : int, optional
        number of workers to use for cache calculation
    """

    data: list[PolymerDatapoint]
    featurizer: Featurizer[Polymer, WeightedMolGraph] = field(
        default_factory=PolymerMolGraphFeaturizer
    )
    n_workers: int = 0

    def _init_cache(self):
        """initialize the cache"""
        if self.cache:
            self.mg_cache = MolGraphCache(
                self.polymers, self.V_fs, self.E_fs, self.featurizer, n_workers=self.n_workers
            )
        else:
            self.mg_cache = MolGraphCacheOnTheFly(
                self.polymers, self.V_fs, self.E_fs, self.featurizer
            )

    @property
    def polymers(self) -> list[Polymer]:
        """the polymers associated with the dataset"""
        return [(d.mol, d.fragment_weights, d.edge_rules) for d in self.data]
