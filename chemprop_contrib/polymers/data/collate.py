from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from chemprop.data.collate import BatchMolGraph, TrainingBatch
from chemprop.data.datasets import Datum
from chemprop_contrib.polymers.data.molgraph import WeightedMolGraph


@dataclass(repr=False, eq=False, slots=True)
class BatchWeightedMolGraph(BatchMolGraph):
    V_w: Tensor = field(init=False)
    """the atom feature weight matrix"""
    E_w: Tensor = field(init=False)
    """the bond feature weight matrix"""
    degree_of_poly: Tensor = field(init=False)
    """the degree of polymerisation matrix in the form 1 + log(Xn),
    default 1 for a non-polymer molecule"""

    def __post_init__(self, mgs: Sequence[WeightedMolGraph]):
        super(BatchWeightedMolGraph, self).__post_init__(mgs)
        Vw = []
        Ew = []
        degree_of_polys = []
        for i, mg in enumerate(mgs):
            Vw.append(mg.V_w)
            Ew.append(mg.E_w)
            degree_of_polys.append(mg.degree_of_poly)
        self.V_w = torch.from_numpy(np.concatenate(Vw)).float()
        self.E_w = torch.from_numpy(np.concatenate(Ew)).float()
        self.degree_of_poly = torch.tensor(degree_of_polys).float()

    def to(self, device: str | torch.device):
        super(BatchWeightedMolGraph, self).to(device)
        self.V_w = self.V_w.to(device)
        self.E_w = self.E_w.to(device)
        self.degree_of_poly = self.degree_of_poly.to(device)


def collate_polymer_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return TrainingBatch(
        BatchWeightedMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )
