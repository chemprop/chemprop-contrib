from chemprop.featurizers.molgraph.reaction import CondensedGraphOfReactionFeaturizer
import numpy as np
import pytest

from chemprop_contrib.reversal_parity.featurizers import ParityCGRFeaturizer

from .conftest import alchemical_smis, make_rxn, reverse_rxn_smi, rxn_smis

RING = "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1"
SINGLE = f"{RING}[OH:7]>>{RING}[CH3:7]"
HYBRID = f"{RING}[OH]>>{RING}[CH3]"


def test_dimensions():
    ours = ParityCGRFeaturizer(share_elements=False)
    F, B = len(ours.atom_featurizer), len(ours.bond_featurizer)
    assert ours.atom_fdim == 2 * F
    assert ours.bond_fdim == 2 * B
    assert ours.odd_atom_start == F
    assert ours.odd_bond_start == B
    # the bond layout is the same width as core REAC_DIFF; atoms gain the element-block diff
    core = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    k = len(ours.atom_featurizer.atomic_nums) + 1
    assert ours.atom_fdim == core.atom_fdim + k
    assert ours.bond_fdim == core.bond_fdim
    # with shared elements the element block is written once, exactly as in core REAC_DIFF
    shared = ParityCGRFeaturizer(share_elements=True)
    assert (shared.atom_fdim, shared.bond_fdim) == (core.atom_fdim, core.bond_fdim)
    assert (shared.odd_atom_start, shared.odd_bond_start) == (F, B)


@pytest.mark.parametrize("balanced", [False, True])
def test_agrees_with_reac_diff_on_element_preserving_reactions(balanced):
    ours = ParityCGRFeaturizer(balanced=balanced, share_elements=False)
    core = CondensedGraphOfReactionFeaturizer(
        mode_="REAC_DIFF_BALANCE" if balanced else "REAC_DIFF"
    )
    F, Bb = ours.odd_atom_start, ours.odd_bond_start
    k = len(ours.atom_featurizer.atomic_nums) + 1

    for smi in rxn_smis:
        mg_ours = ours(make_rxn(smi))
        mg_core = core(make_rxn(smi))
        m, d = mg_ours.V[:, :F], mg_ours.V[:, F:]
        close = dict(atol=1e-6, err_msg=smi)

        # non-element columns: core emits (x_r, d) and leaves the absent side zero, as we do
        np.testing.assert_allclose(m[:, k:] - 0.5 * d[:, k:], mg_core.V[:, k:F], **close)
        np.testing.assert_allclose(d[:, k:], mg_core.V[:, F:], **close)
        # element block: core writes the present side's element once, i.e. max(x_r, x_p),
        # which is m + |d|/2 in our basis (an absent side is zero here, not element-only)
        np.testing.assert_allclose(m[:, :k] + 0.5 * np.abs(d[:, :k]), mg_core.V[:, :k], **close)

        E = mg_ours.E.copy()
        E[:, :Bb] -= 0.5 * E[:, Bb:]
        np.testing.assert_allclose(E, mg_core.E, **close)

        np.testing.assert_array_equal(mg_ours.edge_index, mg_core.edge_index, err_msg=smi)
        np.testing.assert_array_equal(mg_ours.rev_edge_index, mg_core.rev_edge_index, err_msg=smi)


@pytest.mark.parametrize("balanced", [False, True])
def test_shared_elements_is_reac_diff_in_the_md_basis(balanced):
    ours = ParityCGRFeaturizer(balanced=balanced, share_elements=True)
    core = CondensedGraphOfReactionFeaturizer(
        mode_="REAC_DIFF_BALANCE" if balanced else "REAC_DIFF"
    )
    F, k = ours.odd_atom_start, len(ours.atom_featurizer.atomic_nums) + 1

    for smi in rxn_smis:
        V, V_core = ours(make_rxn(smi)).V, core(make_rxn(smi)).V
        m, d = V[:, :F], V[:, F:]
        close = dict(atol=1e-6, err_msg=smi)
        # core emits (x_r, d) with the element once in x_r; ours emits (m, d), m = x_r + d/2
        np.testing.assert_allclose(m[:, :k], V_core[:, :k], **close)
        np.testing.assert_allclose(m[:, k:] - 0.5 * d, V_core[:, k:F], **close)
        np.testing.assert_allclose(d, V_core[:, F:], **close)
        # reversal is still a sign flip on the odd block (atoms come in the other side's order)
        V_rev = ours(make_rxn(reverse_rxn_smi(smi))).V
        np.testing.assert_allclose(_sorted_rows(np.hstack((m, -d))), _sorted_rows(V_rev), **close)


def test_elements_are_shared_by_default():
    feat = ParityCGRFeaturizer()
    core = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    assert (feat.atom_fdim, feat.bond_fdim) == (core.atom_fdim, core.bond_fdim)
    with pytest.raises(ValueError, match="share_elements"):
        feat(make_rxn(SINGLE))


def test_shared_elements_rejects_element_changes():
    feat = ParityCGRFeaturizer(share_elements=True)
    with pytest.raises(ValueError, match="share_elements"):
        feat(make_rxn(SINGLE))
    # an element that vanishes or appears unmapped is not a change of a mapped atom
    assert feat(make_rxn(HYBRID)).V.shape[1] == feat.atom_fdim


def _sorted_rows(A):
    return A[np.lexsort(np.round(A, 6).T[::-1])]


def _atom(mol, symbol):
    (atom,) = [a for a in mol.GetAtoms() if a.GetSymbol() == symbol and not a.GetIsAromatic()]
    return atom


def _sd(feat, x_r, x_p):
    return np.concatenate([0.5 * (x_r + x_p), x_p - x_r])


def test_single_topology_transmutes_one_node():
    feat = ParityCGRFeaturizer(share_elements=False)
    rct, pdt = make_rxn(SINGLE)
    V = feat((rct, pdt)).V
    assert len(V) == 7
    f = feat.atom_featurizer
    i = _atom(rct, "O").GetIdx()
    np.testing.assert_allclose(V[i], _sd(feat, f(_atom(rct, "O")), f(_atom(pdt, "C"))))
    # the element change lives in the odd block
    k = len(f.atomic_nums) + 1
    d_elem = V[i, feat.odd_atom_start : feat.odd_atom_start + k]
    assert sorted(d_elem[d_elem != 0]) == [-1.0, 1.0]


@pytest.mark.parametrize("balanced", [False, True])
def test_hybrid_topology_uses_a_zero_dummy_atom(balanced):
    feat = ParityCGRFeaturizer(balanced=balanced, share_elements=False)
    rct, pdt = make_rxn(HYBRID)
    V = feat((rct, pdt)).V
    assert len(V) == 8
    f = feat.atom_featurizer
    o, c = f(_atom(rct, "O")), f(_atom(pdt, "C"))
    i_o, i_c = _atom(rct, "O").GetIdx(), rct.GetNumAtoms()  # product-only atoms come last
    if balanced:
        # *_BALANCE semantics: an atom absent from one side is copied across unchanged
        np.testing.assert_allclose(V[i_o], _sd(feat, o, o))
        np.testing.assert_allclose(V[i_c], _sd(feat, c, c))
    else:
        zero = f(None)
        assert not zero.any()
        np.testing.assert_allclose(V[i_o], _sd(feat, o, zero))
        np.testing.assert_allclose(V[i_c], _sd(feat, zero, c))


def test_net_element_change_is_topology_independent():
    feat = ParityCGRFeaturizer(share_elements=False)
    F, k = feat.odd_atom_start, len(feat.atom_featurizer.atomic_nums) + 1
    # same molecules, same (implicit) hydrogen convention, different topology
    single, hybrid = (feat(make_rxn(smi)).V[:, F : F + k].sum(0) for smi in (SINGLE, HYBRID))
    np.testing.assert_allclose(single, hybrid)
    assert np.abs(single).sum() == 2  # -O, +C


def test_accepts_element_changing_maps():
    feat = ParityCGRFeaturizer(share_elements=False)
    for smi in [*alchemical_smis, "[CH4:1]>>[NH3:1]"]:
        mg = feat(make_rxn(smi))
        assert mg.V.shape[1] == feat.atom_fdim


def test_rejects_explicit_mode():
    with pytest.raises(ValueError, match="does not accept"):
        ParityCGRFeaturizer(mode_="REAC_PROD")


def test_balanced_and_mode_are_keyword_only():
    feat = ParityCGRFeaturizer(share_elements=False)
    with pytest.raises(TypeError):
        ParityCGRFeaturizer(feat.atom_featurizer, feat.bond_featurizer, True)
    assert ParityCGRFeaturizer(feat.atom_featurizer, feat.bond_featurizer, balanced=True).balanced
