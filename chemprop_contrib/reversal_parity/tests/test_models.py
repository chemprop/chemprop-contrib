import csv
from pathlib import Path

from chemprop.data import ReactionDataset, ReactionDatapoint, build_dataloader
from chemprop.data.collate import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.nn.predictors import (
    BinaryClassificationFFN,
    EvidentialFFN,
    MveFFN,
    QuantileFFN,
    RegressionFFN,
)
from chemprop.nn.transforms import GraphTransform, ScaleTransform, UnscaleTransform
import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from chemprop_contrib.reversal_parity.featurizers import ParityCGRFeaturizer, reverse_bmg
from chemprop_contrib.reversal_parity.models import ParityEquivariantMPNN
from chemprop_contrib.reversal_parity.parity import EVEN, FREE, ODD
from chemprop_contrib.reversal_parity.targets import ParityTargetTransform

from .conftest import (
    alchemical_smis,
    make_rxn,
    parity_smis,
    reverse_rxn_smi,
    rxn_smis,
    shared_element_smis,
)

D_H = 16


def _target(n_tasks=3, odd=(0,), even=(1,), **kwargs):
    return ParityTargetTransform(n_tasks, odd=odd, even=even, **kwargs)


# the fixture reactions have no atom with two bonds, so D-MPNN message updates (W_h) are inert
# on them; this one has a three-atom chain and exercises every encoder weight
CHAIN_SMI = "[CH3:1][CH2:2][OH:3]>>[CH2:1]=[CH2:2].[OH2:3]"


def _featurize(smis, feat=None):
    feat = feat or ParityCGRFeaturizer(share_elements=False)
    return feat, BatchMolGraph([feat(make_rxn(s)) for s in smis])


def _reverse(feat, bmg):
    return reverse_bmg(bmg, feat.odd_atom_start, feat.odd_bond_start)


def _model(
    feat=None,
    predictor=None,
    target=None,
    batch_norm=False,
    d_vd=None,
    d_xd=0,
    graph_transform=None,
    dropout=0.0,
    **kwargs,
):
    feat = feat or ParityCGRFeaturizer(share_elements=False)
    target = _target() if target is None else target
    torch.manual_seed(0)
    mp = BondMessagePassing(
        d_v=feat.atom_fdim,
        d_e=feat.bond_fdim,
        d_h=D_H,
        d_vd=d_vd,
        graph_transform=graph_transform,
        dropout=dropout,
    )
    predictor = predictor or RegressionFFN(
        n_tasks=target.n_tasks, input_dim=mp.output_dim + d_xd, hidden_dim=D_H, dropout=dropout
    )
    return ParityEquivariantMPNN.from_featurizer(
        feat, mp, MeanAggregation(), predictor, target, batch_norm=batch_norm, **kwargs
    )


def _batch(bmg, n_tasks=3, V_d=None, X_d=None, targets=None):
    B = len(bmg)
    targets = torch.randn(B, n_tasks) if targets is None else targets
    return (
        bmg,
        V_d,
        X_d,
        targets,
        torch.ones(B, 1),
        torch.zeros(B, n_tasks, dtype=torch.bool),
        torch.zeros(B, n_tasks, dtype=torch.bool),
    )


class _MetricSpy(torch.nn.Module):
    """Module wrapper so it can replace an entry of ``model.metrics`` (a ``ModuleList``)."""

    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.preds = []

    def forward(self, preds, *args):
        self.preds.append(preds)
        return self.metric(preds, *args)


def _record_logs(model, monkeypatch):
    # LightningModule.log warns when no Trainer is attached; record the calls instead so the
    # step methods can be unit-tested and their logging checked.
    calls = []
    monkeypatch.setattr(model, "log", lambda name, value, **kw: calls.append((name, value, kw)))
    return calls


# ---------------------------------------------------------------------------------------------
# R3 / R4 in eval()


@torch.no_grad()
def test_odd_task_is_exactly_antisymmetric():
    feat, bmg = _featurize(parity_smis)
    model = _model(feat).eval()
    f = model(bmg)[:, 0]
    torch.testing.assert_close(model(_reverse(feat, bmg))[:, 0], -f, rtol=0, atol=1e-5)

    _, bmg_rev = _featurize([reverse_rxn_smi(s) for s in parity_smis], feat)
    torch.testing.assert_close(model(bmg_rev)[:, 0], -f, rtol=0, atol=1e-5)


@torch.no_grad()
def test_even_task_is_exactly_symmetric():
    feat, bmg = _featurize(parity_smis)
    model = _model(feat).eval()
    f = model(bmg)[:, 1]
    torch.testing.assert_close(model(_reverse(feat, bmg))[:, 1], f, rtol=0, atol=1e-5)

    _, bmg_rev = _featurize([reverse_rxn_smi(s) for s in parity_smis], feat)
    torch.testing.assert_close(model(bmg_rev)[:, 1], f, rtol=0, atol=1e-5)


@torch.no_grad()
def test_free_task_equals_forward_pass():
    feat, bmg = _featurize(parity_smis)
    model = _model(feat).eval()
    expected = model.predictor(model.fingerprint(bmg))[:, 2]
    torch.testing.assert_close(model(bmg)[:, 2], expected, rtol=0, atol=1e-5)
    # and it is not constrained: every reversed reaction gives a different value
    assert not torch.isclose(model(_reverse(feat, bmg))[:, 2], expected, atol=1e-5).any()


@torch.no_grad()
@pytest.mark.parametrize("balanced", [False, True])
def test_parity_is_exact_with_shared_elements(balanced):
    feat = ParityCGRFeaturizer(balanced=balanced, share_elements=True)
    _, bmg = _featurize(shared_element_smis, feat)
    _, bmg_rev = _featurize([reverse_rxn_smi(s) for s in shared_element_smis], feat)
    model = _model(feat).eval()
    f = model(bmg)
    # the sign flip and the featurized reversed reactions agree, with the narrower odd block
    for f_rev in (model(_reverse(feat, bmg)), model(bmg_rev)):
        torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
        torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)
    # the predictions are not trivially zero, and the free task still tells the directions apart
    assert (f[:, 0].abs() > 1e-5).any()
    assert not torch.isclose(model(bmg_rev)[:, 2], f[:, 2], atol=1e-5).any()


@torch.no_grad()
def test_parity_is_exact_with_dropout_inactive_and_approximate_with_it_active():
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    rev = _reverse(feat, bmg)
    model = _model(feat, dropout=0.5)

    # eval(): dropout is inactive, so every prediction is exactly odd or even
    model.eval()
    f, f_rev = model(bmg), model(rev)
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)
    torch.testing.assert_close(model(bmg), f, rtol=0, atol=0)

    # train(): the forward and reversed copies get independent masks, so a single draw is not
    model.train()
    torch.manual_seed(0)
    f, f_rev = model(bmg), model(rev)
    assert not torch.allclose(f_rev[:, 0], -f[:, 0], atol=1e-5)
    assert not torch.allclose(f_rev[:, 1], f[:, 1], atol=1e-5)


# ---------------------------------------------------------------------------------------------
# R8: training and validation paths


def test_training_step_applies_the_parity_combination(monkeypatch):
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    model = _model(feat).train()
    logs = _record_logs(model, monkeypatch)
    spy = _MetricSpy(model.predictor.criterion)
    monkeypatch.setattr(model.predictor, "criterion", spy)

    targets = torch.randn(len(bmg), 3)
    targets[0, 1] = float("nan")
    loss = model.training_step(_batch(bmg, targets=targets), 0)
    model.training_step(_batch(_reverse(feat, bmg), targets=targets), 0)

    f, f_rev = spy.preds
    assert f.shape == (len(bmg), 3)
    # no batch norm and no dropout, so train mode is exact as well
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)
    # the objective sees the combined output, not the forward pass alone
    raw = model.predictor.train_step(model.fingerprint(bmg))
    assert not torch.allclose(f[:, 0], raw[:, 0], atol=1e-5)
    torch.testing.assert_close(f[:, 2], raw[:, 2], rtol=0, atol=1e-5)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert [name for name, *_ in logs] == ["train_loss", "train_loss"]
    assert logs[0][2]["batch_size"] == len(bmg)

    # gradients reach every parameter through the combined objective
    model.zero_grad()
    model.training_step(_batch(bmg, targets=targets), 0).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_validation_step_applies_the_parity_combination(monkeypatch):
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat)
    model.on_validation_model_eval()
    logs = _record_logs(model, monkeypatch)
    spy = _MetricSpy(model.metrics[-1])
    model.metrics[-1] = spy

    with torch.no_grad():
        model.validation_step(_batch(bmg), 0)
        model.validation_step(_batch(_reverse(feat, bmg)), 0)

    f, f_rev = spy.preds
    assert f.shape == (len(bmg), 3)
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)
    names = [name for name, *_ in logs]
    assert "val_loss" in names and any(n.startswith("val/") for n in names)


# ---------------------------------------------------------------------------------------------
# public contract of fingerprint / encoding


@torch.no_grad()
@pytest.mark.parametrize("batch_norm", [False, True])
def test_paired_fingerprint_equals_two_separate_encoder_passes(batch_norm):
    # the single stacked pass must compute exactly what encoding x and x̄ separately computes
    d_vd, d_xd = 3, 2
    feat, bmg = _featurize(rxn_smis + alchemical_smis + [CHAIN_SMI])
    model = _model(feat, batch_norm=batch_norm, d_vd=d_vd, d_xd=d_xd).double().eval()
    bmg.V, bmg.E = bmg.V.double(), bmg.E.double()
    gen = torch.Generator().manual_seed(0)
    V_d = torch.randn(bmg.V.shape[0], d_vd, generator=gen, dtype=torch.float64)
    X_d = torch.randn(len(bmg), d_xd, generator=gen, dtype=torch.float64)

    H_f = model.agg(model.message_passing(bmg, V_d), bmg.batch)
    H_r = model.agg(model.message_passing(_reverse(feat, bmg), V_d), bmg.batch)
    H = model.bn(torch.cat([H_f, H_r]))
    expected = torch.cat((H, model.X_d_transform(X_d).repeat(2, 1)), dim=1)

    actual = model._paired_fingerprint(bmg, V_d, X_d)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


@torch.no_grad()
def test_fingerprint_and_encoding_are_unchanged():
    # both describe the forward reactions alone, one row each, as in a plain MPNN
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat, batch_norm=True).eval()
    fp = model.fingerprint(bmg)
    assert fp.shape == (len(bmg), D_H)
    torch.testing.assert_close(fp, MPNN.fingerprint(model, bmg), rtol=0, atol=0)
    for i in (0, 1, -1):
        torch.testing.assert_close(
            model.encoding(bmg, i=i), MPNN.encoding(model, bmg, i=i), rtol=0, atol=0
        )
        assert model.encoding(bmg, i=i).shape[0] == len(bmg)


# ---------------------------------------------------------------------------------------------
# R7 validation and the target transform


def test_rejects_predictor_with_output_transform():
    transform = UnscaleTransform(mean=[0.0] * 3, scale=[1.0] * 3)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H, output_transform=transform)
    with pytest.raises(ValueError, match="output_transform"):
        _model(predictor=predictor)


def test_rejects_target_that_is_not_a_parity_target_transform():
    with pytest.raises(TypeError, match="ParityTargetTransform"):
        _model(
            predictor=RegressionFFN(n_tasks=3, input_dim=D_H),
            target=UnscaleTransform(mean=[0.0] * 3, scale=[1.0] * 3),
        )


@torch.no_grad()
def test_target_transform_unscales_after_the_combination():
    target = _target(mean=[0.0, 5.0, 7.0], scale=[2.0, 3.0, 4.0])
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat, target=target).eval()

    g_f = model.predictor.ffn(model.fingerprint(bmg))
    g_r = model.predictor.ffn(model.fingerprint(_reverse(feat, bmg)))
    out = model(bmg)
    torch.testing.assert_close(out[:, 0], 0.5 * (g_f - g_r)[:, 0] * 2.0, rtol=0, atol=1e-5)
    torch.testing.assert_close(out[:, 1], 0.5 * (g_f + g_r)[:, 1] * 3.0 + 5.0, rtol=0, atol=1e-5)
    torch.testing.assert_close(out[:, 2], g_f[:, 2] * 4.0 + 7.0, rtol=0, atol=1e-5)
    torch.testing.assert_close(model(_reverse(feat, bmg))[:, 0], -out[:, 0], rtol=0, atol=1e-5)


def _scale_transform(d):
    g = torch.Generator().manual_seed(d)
    return ScaleTransform(
        mean=torch.rand(d, generator=g).tolist(), scale=(0.5 + torch.rand(d, generator=g)).tolist()
    )


def test_parity_is_exact_with_atom_and_bond_feature_scaling():
    # the graph transform runs inside the encoder, after reverse_bmg, so it is part of g: parity
    # holds even with nonzero means on the odd (difference) columns
    feat, bmg = _featurize(rxn_smis)
    gt = GraphTransform(_scale_transform(feat.atom_fdim), _scale_transform(feat.bond_fdim))
    assert torch.all(gt.V_transform.mean[0, feat.odd_atom_start :] != 0)
    model = _model(feat, graph_transform=gt).eval()
    _, bmg_rev = _featurize([reverse_rxn_smi(s) for s in rxn_smis], feat)
    out, out_rev = model(bmg), model(bmg_rev)
    torch.testing.assert_close(out_rev[:, 0], -out[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(out_rev[:, 1], out[:, 1], rtol=0, atol=1e-5)
    # the transform is active, not inert
    plain = _model(feat).eval()
    assert not torch.allclose(plain(bmg), out)


def test_from_featurizer_rejects_stock_cgr_featurizer():
    from chemprop.featurizers import CondensedGraphOfReactionFeaturizer

    stock = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    mp = BondMessagePassing(d_v=stock.atom_fdim, d_e=stock.bond_fdim, d_h=D_H)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(TypeError, match="ParityCGRFeaturizer"):
        ParityEquivariantMPNN.from_featurizer(stock, mp, MeanAggregation(), predictor, _target())


@pytest.mark.parametrize("which", ["d_v", "d_e"])
def test_from_featurizer_rejects_encoder_width_mismatch(which):
    feat = ParityCGRFeaturizer(share_elements=False)
    d_v = feat.atom_fdim - (1 if which == "d_v" else 0)
    d_e = feat.bond_fdim - (1 if which == "d_e" else 0)
    mp = BondMessagePassing(d_v=d_v, d_e=d_e, d_h=D_H)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(ValueError, match=f"{which} = .* but the featurizer"):
        ParityEquivariantMPNN.from_featurizer(feat, mp, MeanAggregation(), predictor, _target())


def test_start_index_upper_bound_skipped_without_message_passing_dims(monkeypatch):
    feat = ParityCGRFeaturizer(share_elements=False)
    mp = BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=D_H)
    monkeypatch.setattr(type(mp), "hparams", property(lambda self: {}))
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    model = ParityEquivariantMPNN.from_featurizer(feat, mp, MeanAggregation(), predictor, _target())
    assert model.odd_atom_start == feat.odd_atom_start
    with pytest.raises(ValueError, match="odd_atom_start must satisfy 0 < odd_atom_start < d_v,"):
        ParityEquivariantMPNN(mp, MeanAggregation(), predictor, 0, feat.odd_bond_start, _target())


def test_rejects_disallowed_predictor():
    class Bare(torch.nn.Module):  # a custom predictor need not have an output_transform
        pass

    for predictor in (BinaryClassificationFFN(n_tasks=3, input_dim=D_H), Bare()):
        with pytest.raises(ValueError, match=type(predictor).__name__):
            _model(predictor=predictor)


@pytest.mark.parametrize("which", ["atom", "bond"])
def test_rejects_invalid_start_index(which):
    feat = ParityCGRFeaturizer(share_elements=False)
    mp = BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=D_H)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    starts = dict(odd_atom_start=feat.odd_atom_start, odd_bond_start=feat.odd_bond_start)
    name = f"odd_{which}_start"
    bad_starts = {
        "must satisfy": (0, -1, getattr(feat, f"{which}_fdim")),  # out of range
        "must be an integer": (10.5, True, None, "a", float("inf")),
    }
    for reason, values in bad_starts.items():
        for bad in values:
            kwargs = {**starts, name: bad, "target_transform": _target()}
            with pytest.raises(ValueError, match=f"{name} {reason}"):
                ParityEquivariantMPNN(mp, MeanAggregation(), predictor, **kwargs)


def test_rejects_target_with_wrong_number_of_tasks():
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(ValueError, match="n_tasks"):
        _model(predictor=predictor, target=ParityTargetTransform(2, odd=[0]))


# ---------------------------------------------------------------------------------------------
# extra descriptors, batch norm


@torch.no_grad()
def test_parity_is_exact_with_extra_atom_and_datapoint_descriptors():
    d_vd, d_xd = 3, 4
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat, d_vd=d_vd, d_xd=d_xd).eval()
    V_d = torch.randn(bmg.V.shape[0], d_vd)
    X_d = torch.randn(len(bmg), d_xd)

    f = model(bmg, V_d, X_d)
    f_rev = model(_reverse(feat, bmg), V_d, X_d)
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)
    # both kinds of descriptors reach the prediction
    assert not torch.allclose(model(bmg, torch.zeros_like(V_d), X_d), f, atol=1e-5)
    assert not torch.allclose(model(bmg, V_d, torch.zeros_like(X_d)), f, atol=1e-5)


def test_antisymmetry_holds_with_batch_norm():
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat, batch_norm=True)

    # Train mode: both passes are normalized together, so the forward and reversed inputs see
    # identical batch statistics and the combination stays exact. No output/graph transforms are
    # attached, so their train-mode short-circuit cannot make this pass for unrelated reasons.
    model.train()
    with torch.no_grad():
        f = model(bmg)
        f_rev = model(_reverse(feat, bmg))
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)

    # Eval mode with non-trivial running statistics
    with torch.no_grad():
        model.bn.running_mean.uniform_(-1, 1)
        model.bn.running_var.uniform_(0.5, 2)
    model.eval()
    with torch.no_grad():
        f = model(bmg)
        f_rev = model(_reverse(feat, bmg))
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------------------------
# R5 / R6: uncertainty predictors


def _odd_even_outputs(cls):
    """Eval-mode outputs ``(f, f_rev)``, each ``(B, 2, n_targets)``, for tasks ``[ODD, EVEN]``."""
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    predictor = cls(n_tasks=2, input_dim=D_H, hidden_dim=D_H)
    model = _model(feat, predictor=predictor, target=_target(2)).eval()
    with torch.no_grad():
        return model(bmg), model(_reverse(feat, bmg))


def _assert_component_parity(f, f_rev):
    # component 0 carries the task's parity (odd for task 0, even for task 1); every later
    # component is even, so a positive uncertainty stays positive under reversal
    torch.testing.assert_close(f_rev[:, 0, 0], -f[:, 0, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1, 0], f[:, 1, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[..., 1:], f[..., 1:], rtol=0, atol=1e-5)


def test_mve_variance_stays_positive_for_odd_task():
    f, f_rev = _odd_even_outputs(MveFFN)
    assert f.shape == (len(rxn_smis) + 1, 2, 2)
    for out in (f, f_rev):
        assert (out[..., 1] > 0).all()
    _assert_component_parity(f, f_rev)


def test_evidential_params_stay_positive_for_odd_task():
    f, f_rev = _odd_even_outputs(EvidentialFFN)
    assert f.shape == (len(rxn_smis) + 1, 2, 4)
    for out in (f, f_rev):
        v, alpha, beta = out[..., 1], out[..., 2], out[..., 3]
        assert (v > 0).all() and (alpha > 1).all() and (beta > 0).all()
    _assert_component_parity(f, f_rev)


def test_quantile_mean_is_odd_and_interval_is_even():
    # the interval's sign is not guaranteed for an untrained model, so only its parity is checked
    f, f_rev = _odd_even_outputs(QuantileFFN)
    assert f.shape == (len(rxn_smis) + 1, 2, 2)
    _assert_component_parity(f, f_rev)


def test_mve_loss_is_finite_for_odd_task(monkeypatch):
    # a negative variance would make the NLL NaN; every task here is odd, so the variance
    # column must come out of the combination positive
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    predictor = MveFFN(n_tasks=2, input_dim=D_H, hidden_dim=D_H)
    model = _model(feat, predictor=predictor, target=_target(2, odd=(0, 1), even=())).train()
    _record_logs(model, monkeypatch)

    assert (model._train_preds(bmg)[..., 1] > 0).all()
    loss = model.training_step(_batch(bmg, n_tasks=2), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------------------------
# R8 / R9: fixture data, training, checkpoint round-trip

FIXTURE_CSV = Path(__file__).with_name("test_data.csv")
FIXTURE_PARITY = [FREE, ODD, EVEN]  # CSV column order: ea, y_odd, y_even
FIXTURE_COLS = ["ea", "y_odd", "y_even"]
N_TRAIN, N_VAL = 24, 8


def _fixture_rows():
    with open(FIXTURE_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _datasets(feat, n_train=N_TRAIN, n_val=N_VAL, spec=None, as_reverse=False):
    """Fixture datasets; ``as_reverse`` replaces y_odd with ea - y_odd, the reverse value of a
    pair whose forward value is ea and whose difference is y_odd."""
    rows = _fixture_rows()[: n_train + n_val]
    ys = [[float(r[c]) for c in FIXTURE_COLS] for r in rows]
    if as_reverse:
        ys = [[ea, ea - d, lam] for ea, d, lam in ys]
    dps = [
        ReactionDatapoint.from_smi(r["smiles"], y=y, keep_h=True, add_h=False)
        for r, y in zip(rows, ys)
    ]
    train = ReactionDataset(dps[:n_train], featurizer=feat)
    val = ReactionDataset(dps[n_train:], featurizer=feat)
    # as in the README: fit the transform, normalize both datasets with its scaler
    target = ParityTargetTransform.fit(train.Y, **(spec or dict(odd=[1], even=[2])))
    train.normalize_targets(target.to_standard_scaler())
    val.normalize_targets(target.to_standard_scaler())
    return train, val, target


def _fixture_model(feat, target, task_weights=None):
    torch.manual_seed(0)
    mp = BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=32)
    predictor = RegressionFFN(
        n_tasks=3, input_dim=mp.output_dim, hidden_dim=32, task_weights=task_weights
    )
    return ParityEquivariantMPNN.from_featurizer(
        feat, mp, MeanAggregation(), predictor, target, batch_norm=True
    )


# Two third-party warnings `trainer.fit` cannot avoid here: Lightning's CombinedLoader calls a
# torch pytree API that torch has deprecated, and the num_workers=0 dataloaders (required for
# a deterministic, fast test) trigger Lightning's "not many workers" advice
lightning_warnings = pytest.mark.filterwarnings(
    "ignore:`isinstance\\(treespec, LeafSpec\\)` is deprecated:FutureWarning",
    "ignore:The '.*dataloader' does not have many workers",
)


def _trainer(max_epochs, **kwargs):
    return pl.Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=kwargs.pop("enable_checkpointing", False),
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=max_epochs,
        **kwargs,
    )


@torch.no_grad()
def _eval_loss(model, loader):
    """Criterion on eval-mode combined predictions, in the scaled target space."""
    was_training = model.training
    model.eval()
    # in eval() the target transform unscales, so map back to the space of the scaled targets
    tf = model.target_transform
    total, n = 0.0, 0
    for bmg, _, _, targets, weights, *_ in loader:
        preds = (model(bmg) - tf.mean) / tf.scale
        if tf.pairs:  # the loss also scores each pair's implied member
            preds = torch.cat((preds, tf.implied(preds)), dim=1)
            targets = torch.cat((targets, tf.implied(targets)), dim=1)
        mask = targets.isfinite()
        loss = model.criterion(preds, targets.nan_to_num(nan=0.0), mask, weights, None, None)
        total += float(loss) * len(bmg)
        n += len(bmg)
    model.train(was_training)
    return total / n


def _fit(max_epochs=100, **trainer_kwargs):
    pl.seed_everything(0)
    feat = ParityCGRFeaturizer(share_elements=False)
    train, val, transform = _datasets(feat)
    model = _fixture_model(feat, transform)
    train_loader = build_dataloader(train, batch_size=8, num_workers=0, seed=0)
    val_loader = build_dataloader(val, batch_size=8, num_workers=0, shuffle=False)
    eval_loader = build_dataloader(train, batch_size=N_TRAIN, num_workers=0, shuffle=False)
    before = _eval_loss(model, eval_loader)
    trainer = _trainer(max_epochs, **trainer_kwargs)
    trainer.fit(model, train_loader, val_loader)
    return feat, model, trainer, before, _eval_loss(model, eval_loader), val


@lightning_warnings
def test_training_reduces_loss_on_an_exactly_odd_target():
    # meaningful because y_odd genuinely has odd parity: a constrained model can fit a
    # function of the parity it exists for
    _, _, _, before, after, _ = _fit()
    assert after < 0.1 * before, (before, after)


@torch.no_grad()
@lightning_warnings
def test_checkpoint_round_trip_preserves_parity(tmp_path):
    feat, model, trainer, _, _, val = _fit(max_epochs=3)
    path = tmp_path / "parity.ckpt"
    trainer.save_checkpoint(path)

    loaded = ParityEquivariantMPNN.load_from_checkpoint(path)
    for hp in (loaded.hparams, model.hparams):
        assert hp["odd_atom_start"] == feat.odd_atom_start
        assert hp["odd_bond_start"] == feat.odd_bond_start
        assert hp["target_transform"].task_parity == FIXTURE_PARITY
    assert loaded.odd_atom_start == feat.odd_atom_start
    assert loaded.odd_bond_start == feat.odd_bond_start
    assert loaded.task_parity == FIXTURE_PARITY
    assert torch.equal(loaded.parity_mask, model.parity_mask)

    model.eval()
    loaded.eval()
    bmg = BatchMolGraph([val[i].mg for i in range(len(val))])
    f = loaded(bmg)
    torch.testing.assert_close(f, model(bmg), rtol=0, atol=1e-5)
    f_rev = loaded(_reverse(feat, bmg))
    torch.testing.assert_close(f_rev[:, 1], -f[:, 1], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 2], f[:, 2], rtol=0, atol=1e-5)

    # chemprop's own loader rebuilds the model from the same checkpoint
    from_file = ParityEquivariantMPNN.load_from_file(path).eval()
    assert from_file.task_parity == FIXTURE_PARITY
    assert torch.equal(from_file.parity_mask, model.parity_mask)
    torch.testing.assert_close(from_file(bmg), model(bmg), rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------------------------
# paired columns: the fixture's "ea" column (0) is the forward value of a pair whose difference
# is the odd column y_odd; column 1 holds either that difference or the implied reverse value

PAIR_D = dict(paired_difference={0: 1}, even=[2])
PAIR_R = dict(paired_reverse={0: 1}, even=[2])
SCALING_D = dict(mean=[7.0, 0.0, 2.0], scale=[3.0, 5.0, 1.5])
SCALING_R = dict(mean=[7.0, 7.0, 2.0], scale=[3.0, 3.0, 1.5], difference_scale=[5.0])


def _as_double(bmg):
    bmg.V, bmg.E = bmg.V.double(), bmg.E.double()
    return bmg


@torch.no_grad()
@pytest.mark.parametrize("spec, scaling", [(PAIR_D, SCALING_D), (PAIR_R, SCALING_R)])
def test_pairs_are_exactly_consistent(spec, scaling):
    smis = [*parity_smis, CHAIN_SMI]
    feat, bmg = _featurize(smis)
    _, bmg_rev = _featurize([reverse_rxn_smi(s) for s in smis], feat)
    bmg, bmg_rev = _as_double(bmg), _as_double(bmg_rev)
    model = _model(feat, target=ParityTargetTransform(3, **spec, **scaling)).double().eval()
    f, f_rev = model(bmg), model(bmg_rev)
    torch.testing.assert_close(f_rev, model(_reverse(feat, bmg)), rtol=0, atol=1e-12)
    if spec is PAIR_D:  # the forward value of x̄ is forward - difference of x; difference is odd
        torch.testing.assert_close(f_rev[:, 0], f[:, 0] - f[:, 1], rtol=0, atol=1e-12)
        torch.testing.assert_close(f_rev[:, 1], -f[:, 1], rtol=0, atol=1e-12)
    else:  # reversal swaps forward and reverse
        torch.testing.assert_close(f_rev[:, :2], f[:, [1, 0]], rtol=0, atol=1e-12)
    assert (f[:, 0] - f_rev[:, 0]).abs().min() > 1e-6  # not vacuous for any reaction


@torch.no_grad()
@pytest.mark.parametrize("cls", [MveFFN, EvidentialFFN, QuantileFFN])
def test_pair_uncertainties_follow_their_members_under_reversal(cls):
    smis = [*parity_smis, CHAIN_SMI]
    feat, bmg = _featurize(smis)
    bmg = _as_double(bmg)
    bmg_rev = _reverse(feat, bmg)
    out = []
    for spec, scaling in ((PAIR_D, SCALING_D), (PAIR_R, SCALING_R)):
        torch.manual_seed(1)  # the same weights under both declarations
        predictor = cls(n_tasks=3, input_dim=D_H, hidden_dim=D_H)
        target = ParityTargetTransform(3, **spec, **scaling)
        model = _model(feat, predictor, target).double().eval()
        out.append((model(bmg), model(bmg_rev)))
    (d, d_rev), (r, r_rev) = out

    # reversal swaps forward and reverse, each with its own uncertainty
    torch.testing.assert_close(r_rev[:, :2], r[:, [1, 0]], rtol=0, atol=1e-12)
    assert (r[:, 0, 1:] - r[:, 1, 1:]).abs().min() > 1e-6  # the two are not tied
    # the difference is odd and its uncertainty even
    torch.testing.assert_close(d_rev[:, 1, 0], -d[:, 1, 0], rtol=0, atol=1e-12)
    torch.testing.assert_close(d_rev[:, 1, 1:], d[:, 1, 1:], rtol=0, atol=1e-12)
    # the forward column is the same under both declarations, as are their values
    torch.testing.assert_close(d[:, 0], r[:, 0], rtol=0, atol=1e-12)
    torch.testing.assert_close(d[:, 1, 0], r[:, 0, 0] - r[:, 1, 0], rtol=0, atol=1e-12)


@torch.no_grad()
def test_paired_training_and_eval_paths_agree():
    # the loss sees scaled predictions; unscaling them gives the eval-mode output
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    target = ParityTargetTransform(3, **PAIR_D, **SCALING_D)
    model = _model(feat, target=target).eval()
    model.on_validation_model_eval()  # eval, with the target transform in train mode
    scaled = model._train_preds(bmg)[:, :3]  # the declared columns, before the implied member
    model.eval()
    torch.testing.assert_close(model(bmg), scaled * target.scale + target.mean, rtol=0, atol=1e-5)


@torch.no_grad()
def test_loss_scores_the_implied_member_where_both_labels_are_known():
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    model = _model(feat, target=ParityTargetTransform(3, **PAIR_D, **SCALING_D)).train()
    tf = model.target_transform
    Y = torch.tensor([[10.0, 4.0, 1.0]]).repeat(len(bmg), 1)
    Y[1, 1] = float("nan")  # forward only
    Y[2, 0] = float("nan")  # difference only
    batch = _batch(bmg, targets=(Y - tf.mean) / tf.scale)
    batch[6][3, 0] = True  # a bound, not a value, on the forward column
    preds, targets, mask, _, lt_mask, gt_mask = model._loss_inputs(batch)

    assert preds.shape == targets.shape == mask.shape == lt_mask.shape == (len(bmg), 4)
    torch.testing.assert_close(preds[:, 3:], tf.implied(preds[:, :3]))
    # the implied reverse value 10 - 4 = 6, scaled like the forward column
    torch.testing.assert_close(targets[0, 3], torch.tensor((6.0 - 7.0) / 3.0))
    assert mask[:, 3].tolist() == [True, False, False, False, *[True] * (len(bmg) - 4)]
    assert not lt_mask[:, 3].any() and not gt_mask[:, 3].any()


@pytest.mark.parametrize("cls", [RegressionFFN, MveFFN, EvidentialFFN, QuantileFFN])
def test_both_declarations_of_a_pair_train_alike(cls, monkeypatch):
    # the same data given as (forward, difference) or as (forward, reverse) gives the same loss,
    # since both score forward, reverse and difference, each with its own uncertainty
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    rng = np.random.default_rng(0)
    n = len(bmg)
    ea, de, lam = rng.normal(30.0, 8.0, n), rng.normal(5.0, 9.0, n), rng.random(n)
    de[0] = np.nan  # a row with the forward value only
    losses = []
    for Y, spec in ((np.c_[ea, de, lam], PAIR_D), (np.c_[ea, ea - de, lam], PAIR_R)):
        target = ParityTargetTransform.fit(Y, **spec)
        torch.manual_seed(1)  # the same weights under both declarations
        predictor = cls(n_tasks=3, input_dim=D_H, hidden_dim=D_H)
        model = _model(feat, predictor, target).train()
        _record_logs(model, monkeypatch)
        Z = torch.tensor((Y - target.mean.double().numpy()) / target.scale.double().numpy())
        losses.append(model.training_step(_batch(bmg, targets=Z.float()), 0))
    assert torch.isfinite(losses[0])
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=0)


@pytest.mark.parametrize(
    "pair, extra", [(dict(paired_difference={0: 1}), 2.0), (dict(paired_reverse={0: 1}), 1.0)]
)
def test_implied_members_get_loss_weights(pair, extra):
    # an implied reverse value takes its forward column's weight, an implied difference 1
    predictor = RegressionFFN(
        n_tasks=3, input_dim=D_H, hidden_dim=D_H, task_weights=[2.0, 3.0, 4.0]
    )
    model = _model(predictor=predictor, target=ParityTargetTransform(3, **pair))
    for metric in (model.criterion, model.metrics[-1]):
        assert metric.task_weights.tolist() == [[2.0, 3.0, 4.0, extra]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_implied_member_weights_follow_a_predictor_on_the_gpu():
    predictor = RegressionFFN(
        n_tasks=3, input_dim=D_H, hidden_dim=D_H, task_weights=[2.0, 3.0, 4.0]
    ).cuda()
    model = _model(predictor=predictor, target=ParityTargetTransform(3, paired_reverse={0: 1}))
    assert model.criterion.task_weights.device == predictor.ffn[0][0].weight.device
    assert model.criterion.task_weights.tolist() == [[2.0, 3.0, 4.0, 1.0]]


def test_paired_column_trains_without_difference_labels(monkeypatch):
    # rows with only the forward value labeled still train it, through both heads
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    model = _model(feat, target=ParityTargetTransform(3, **PAIR_D)).train()
    _record_logs(model, monkeypatch)
    targets = torch.randn(len(bmg), 3)
    targets[:, 1] = float("nan")
    loss = model.training_step(_batch(bmg, targets=targets), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    w = model.predictor.ffn[-1][-1].weight.grad  # output layer: one row per head
    assert w[0].abs().sum() > 0 and w[1].abs().sum() > 0


@torch.no_grad()
@lightning_warnings
def test_checkpoint_round_trip_preserves_pairing(tmp_path):
    pl.seed_everything(0)
    feat = ParityCGRFeaturizer(share_elements=False)
    train, val, target = _datasets(feat, spec=PAIR_R, as_reverse=True)
    model = _fixture_model(feat, target, task_weights=[1.0, 2.0, 1.0])
    trainer = _trainer(2)
    with torch.enable_grad():
        trainer.fit(
            model,
            build_dataloader(train, batch_size=8, num_workers=0, seed=0),
            build_dataloader(val, batch_size=8, num_workers=0, shuffle=False),
        )
    path = tmp_path / "paired.ckpt"
    trainer.save_checkpoint(path)

    model.eval()
    bmg = BatchMolGraph([val[i].mg for i in range(len(val))])
    for loaded in (
        ParityEquivariantMPNN.load_from_checkpoint(path),
        ParityEquivariantMPNN.load_from_file(path),
    ):
        loaded.eval()
        tf = loaded.target_transform
        assert tf.paired_reverse == {0: 1} and tf.paired_difference == {}
        for name in ("mean", "scale", "difference_scale"):
            torch.testing.assert_close(getattr(tf, name), getattr(target, name), rtol=0, atol=0)
        assert loaded.criterion.task_weights.tolist() == [[1.0, 2.0, 1.0, 1.0]]
        f = loaded(bmg)
        torch.testing.assert_close(f, model(bmg), rtol=0, atol=1e-5)
        f_rev = loaded(_reverse(feat, bmg))
        torch.testing.assert_close(f_rev[:, :2], f[:, [1, 0]], rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------------------------
# the transform's mode decides scaled vs unscaled output, so it must start in sync with the model


def test_models_built_from_one_transform_are_independent():
    target = _target(mean=[0.0, 5.0, 7.0], scale=[2.0, 3.0, 4.0]).eval()  # even if left in eval
    a, b = _model(target=target), _model(target=target)
    assert a.target_transform.training and b.target_transform.training
    a.eval()
    assert b.target_transform.training
    a.double()
    assert b.target_transform.scale.dtype == torch.float32


@torch.no_grad()
@lightning_warnings
def test_model_saved_in_eval_mode_reloads_ready_to_train(tmp_path):
    feat, model, trainer, _, _, val = _fit(max_epochs=1)
    model.eval()
    path = tmp_path / "eval.ckpt"
    trainer.save_checkpoint(path)
    bmg = BatchMolGraph([val[i].mg for i in range(len(val))])
    for loaded in (
        ParityEquivariantMPNN.load_from_checkpoint(path),
        ParityEquivariantMPNN.load_from_file(path),
    ):
        assert loaded.training and loaded.target_transform.training
        # train mode returns scaled predictions, as the loss expects
        tf = loaded.target_transform
        loaded.eval()
        expected = (loaded(bmg) - tf.mean) / tf.scale
        loaded.train()
        loaded.bn.eval()  # running statistics, so only the transform's mode differs
        torch.testing.assert_close(loaded._train_preds(bmg), expected, rtol=0, atol=1e-4)


@torch.no_grad()
@lightning_warnings
@pytest.mark.parametrize("as_reverse", [False, True], ids=["difference", "reverse"])
def test_pair_learns_its_difference(as_reverse):
    # the fixture's odd column is learnable, so after training the predicted forward(x) -
    # forward(x̄), which is exactly the predicted difference, must track the labeled difference
    pl.seed_everything(0)
    feat = ParityCGRFeaturizer(share_elements=False)
    train, val, target = _datasets(
        feat, spec=PAIR_R if as_reverse else PAIR_D, as_reverse=as_reverse
    )
    model = _fixture_model(feat, target)
    loader = build_dataloader(train, batch_size=N_TRAIN, num_workers=0, shuffle=False)
    before = _eval_loss(model, loader)
    with torch.enable_grad():
        _trainer(100).fit(
            model,
            build_dataloader(train, batch_size=8, num_workers=0, seed=0),
            build_dataloader(val, batch_size=8, num_workers=0, shuffle=False),
        )
    assert _eval_loss(model, loader) < 0.5 * before

    model.eval()
    bmg = BatchMolGraph([train[i].mg for i in range(len(train))])
    gap = model(bmg)[:, 0] - model(_reverse(feat, bmg))[:, 0]
    Y = train._Y
    labeled = torch.tensor(Y[:, 0] - Y[:, 1] if as_reverse else Y[:, 1], dtype=gap.dtype)
    assert torch.corrcoef(torch.stack([gap, labeled]))[0, 1] > 0.9
