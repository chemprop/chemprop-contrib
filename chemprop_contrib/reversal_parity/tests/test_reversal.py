from dataclasses import dataclass, field

from chemprop.data.collate import BatchMolGraph
import pytest
import torch

from chemprop_contrib.reversal_parity.featurizers import (
    ParityCGRFeaturizer,
    reverse_bmg,
    stack_reversed_bmg,
)

from .conftest import encode, make_rxn, parity_smis, reverse_rxn_smi, shared_element_smis

ZERO_EDGE_SMI = "[CH4:1]>>[CH3:1].[H:2]"
_TENSOR_FIELDS = ("V", "E", "edge_index", "rev_edge_index", "batch")


def _forward(smis=parity_smis, balanced=False, share_elements=False):
    """A featurizer and one batch holding every reaction in ``smis``."""
    feat = ParityCGRFeaturizer(balanced=balanced, share_elements=share_elements)
    return feat, BatchMolGraph([feat(make_rxn(smi)) for smi in smis])


def _reverse(feat, bmg):
    return reverse_bmg(bmg, feat.odd_atom_start, feat.odd_bond_start)


def test_shares_topology_and_does_not_mutate_input():
    feat, bmg = _forward()
    V, E = bmg.V.clone(), bmg.E.clone()
    out = _reverse(feat, bmg)
    torch.testing.assert_close(bmg.V, V, rtol=0, atol=0)
    torch.testing.assert_close(bmg.E, E, rtol=0, atol=0)
    assert out.edge_index is bmg.edge_index
    assert out.rev_edge_index is bmg.rev_edge_index
    assert out.batch is bmg.batch
    assert len(out) == len(bmg)


def test_is_an_involution():
    feat, bmg = _forward()
    out = _reverse(feat, _reverse(feat, bmg))
    torch.testing.assert_close(out.V, bmg.V, rtol=0, atol=0)
    torch.testing.assert_close(out.E, bmg.E, rtol=0, atol=0)


def test_handles_zero_edge_reaction():
    feat, bmg = _forward([ZERO_EDGE_SMI])
    assert bmg.E.shape == (0, feat.bond_fdim)
    out = _reverse(feat, bmg)
    assert out.E.shape == (0, feat.bond_fdim)
    assert out.V.shape == bmg.V.shape


@pytest.mark.parametrize("share_elements", [False, True])
@pytest.mark.parametrize("balanced", [False, True])
def test_matches_refeaturized_reverse(balanced, share_elements):
    smis = shared_element_smis if share_elements else parity_smis
    feat, bmg = _forward(smis, balanced, share_elements)
    _, refeaturized = _forward([reverse_rxn_smi(s) for s in smis], balanced, share_elements)
    # one row per reaction
    torch.testing.assert_close(encode(_reverse(feat, bmg)), encode(refeaturized), rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------------------------
# stack_reversed_bmg


def _stack(feat, bmg):
    return stack_reversed_bmg(bmg, feat.odd_atom_start, feat.odd_bond_start)


def _flip(feat, mg):
    V, E = mg.V.copy(), mg.E.copy()
    V[:, feat.odd_atom_start :] *= -1
    E[:, feat.odd_bond_start :] *= -1
    return mg._replace(V=V, E=E)


@pytest.mark.parametrize("balanced", [False, True])
def test_stack_matches_chemprop_collate_of_forward_then_reversed_graphs(balanced):
    # chemprop's own collate is the reference for the index offsets
    feat = ParityCGRFeaturizer(balanced=balanced, share_elements=False)
    mgs = [feat(make_rxn(s)) for s in parity_smis]
    out = _stack(feat, BatchMolGraph(mgs))
    ref = BatchMolGraph(mgs + [_flip(feat, mg) for mg in mgs])
    assert len(out) == len(ref) == 2 * len(mgs)
    for name in _TENSOR_FIELDS:
        torch.testing.assert_close(getattr(out, name), getattr(ref, name), rtol=0, atol=0)


def test_stack_does_not_mutate_input():
    feat, bmg = _forward()
    before = {name: getattr(bmg, name).clone() for name in _TENSOR_FIELDS}
    _stack(feat, bmg)
    assert len(bmg) == len(parity_smis)
    for name, value in before.items():
        torch.testing.assert_close(getattr(bmg, name), value, rtol=0, atol=0)


def test_stack_rejects_batch_with_unknown_field():
    # a field the stacking does not know about would be silently left out of the result
    @dataclass(repr=False, eq=False, slots=True)
    class ExtraFieldBatch(BatchMolGraph):
        extra: torch.Tensor = field(init=False)

    feat = ParityCGRFeaturizer(share_elements=False)
    bmg = ExtraFieldBatch([feat(make_rxn(ZERO_EDGE_SMI))])
    bmg.extra = torch.zeros(1)
    with pytest.raises(TypeError, match="extra"):
        _stack(feat, bmg)


def test_stack_rejects_batch_with_ad_hoc_attribute():
    class DictBatch(BatchMolGraph):  # no __slots__, so instances can carry extra attributes
        pass

    feat = ParityCGRFeaturizer(share_elements=False)
    bmg = DictBatch([feat(make_rxn(ZERO_EDGE_SMI))])
    bmg.atom_weights = torch.ones(2)
    with pytest.raises(TypeError, match="atom_weights"):
        _stack(feat, bmg)
