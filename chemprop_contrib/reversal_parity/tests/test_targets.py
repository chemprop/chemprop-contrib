import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from chemprop_contrib.reversal_parity.parity import EVEN, FREE, ODD
from chemprop_contrib.reversal_parity.targets import ParityTargetTransform


def _Y():
    rng = np.random.default_rng(0)
    return rng.normal(loc=[3.0, -2.0, 5.0, 1.0], scale=[1.0, 2.0, 0.5, 3.0], size=(50, 4))


# ---------------------------------------------------------------------------------------------
# declaration


def test_unlisted_columns_are_free_and_pairs_get_an_even_and_an_odd_head():
    t = ParityTargetTransform(7, odd=[5], even=[3], paired_difference={0: 1}, paired_reverse={2: 4})
    assert t.odd == [5] and t.even == [3]
    assert t.paired_difference == {0: 1} and t.paired_reverse == {2: 4}
    # forward column: the pair's even mean; the other column: its odd difference
    assert t.task_parity == [EVEN, ODD, EVEN, EVEN, ODD, ODD, FREE]


def test_accepts_numpy_integer_indices():
    t = ParityTargetTransform(
        np.int64(3), odd=np.array([0]), paired_difference={np.int32(1): np.int64(2)}
    )
    assert t.odd == [0] and t.paired_difference == {1: 2}
    ints = [t.n_tasks, *t.odd, *t.paired_difference, *t.paired_difference.values()]
    assert all(type(i) is int for i in ints)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(odd=[0], even=[0]), "more than one role"),
        (dict(odd=[0, 0]), "more than once"),
        (dict(odd=[3]), r"\[0, 3\)"),
        (dict(odd=[-1]), r"\[0, 3\)"),
        (dict(odd=[0.0]), r"\[0, 3\)"),
        (dict(even=[True]), r"\[0, 3\)"),
        (dict(paired_difference={0: 0}), "pairs column 0 with itself"),
        # a forward column with both a reverse and a difference column
        (dict(paired_difference={0: 1}, paired_reverse={0: 2}), r"\[0\] are given more than"),
    ],
)
def test_rejects_invalid_declarations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ParityTargetTransform(3, **kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(odd=0), "odd must be a list"),
        (dict(even="1"), "even must be a list"),
        (dict(paired_difference=[(0, 1)]), "paired_difference must map forward columns"),
    ],
)
def test_rejects_wrongly_typed_declarations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ParityTargetTransform(3, **kwargs)


@pytest.mark.parametrize("bad", [0, 2.0, True])
def test_rejects_invalid_n_tasks(bad):
    with pytest.raises(ValueError, match="n_tasks"):
        ParityTargetTransform(bad)


def test_rejects_nonzero_mean_on_odd_and_difference_columns():
    pair = dict(paired_difference={0: 1})
    with pytest.raises(ValueError, match=r"column\(s\) \[1, 2\] have a nonzero mean"):
        ParityTargetTransform(3, odd=[2], **pair, mean=[5.0, 1.0, 1.0])
    ParityTargetTransform(3, odd=[2], **pair, mean=[5.0, 0.0, 0.0])


def test_forward_and_reverse_columns_share_their_scaling():
    pair = dict(paired_reverse={0: 1})
    for mean, scale in (([1.0, 2.0], [3.0, 3.0]), ([1.0, 1.0], [3.0, 4.0])):
        with pytest.raises(ValueError, match="share mean and scale"):
            ParityTargetTransform(2, **pair, mean=mean, scale=scale)
    ParityTargetTransform(2, **pair, mean=[1.0, 1.0], scale=[3.0, 3.0])


def test_difference_scale_defaults_and_checks():
    # pairs are ordered by their forward column, whatever their kind
    pairs = dict(paired_difference={2: 1}, paired_reverse={0: 3})
    scale = [3.0, 5.0, 2.0, 3.0]
    t = ParityTargetTransform(4, **pairs, scale=scale)
    # a difference column's own scale; the forward column's for a pair given with reverse=
    assert t.difference_scale.tolist() == [[3.0, 5.0]]
    t = ParityTargetTransform(4, **pairs, scale=scale, difference_scale=[7.0, 5.0])
    assert t.difference_scale.tolist() == [[7.0, 5.0]]
    with pytest.raises(ValueError, match=r"column's scale for \['paired_difference=\{2: 1\}'\]"):
        ParityTargetTransform(4, **pairs, scale=scale, difference_scale=[7.0, 4.0])
    with pytest.raises(ValueError, match="2 positive values"):
        ParityTargetTransform(4, **pairs, scale=scale, difference_scale=[0.0, 5.0])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(scale=[1.0, 0.0]), "positive"),
        (dict(mean=[0.0]), "2 finite values"),
        (dict(mean=[0.0, np.nan]), "2 finite values"),
    ],
)
def test_rejects_invalid_scaling(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ParityTargetTransform(2, **kwargs)


# ---------------------------------------------------------------------------------------------
# fit


def test_fit_scales_odd_columns_about_zero_by_their_rms():
    Y = _Y()
    t = ParityTargetTransform.fit(Y, odd=[1], even=[2])
    mean, scale = t.mean[0].double().numpy(), t.scale[0].double().numpy()
    assert mean[1] == 0
    np.testing.assert_allclose(scale[1], np.sqrt(np.mean(Y[:, 1] ** 2)), rtol=1e-6)
    for i in (0, 2, 3):
        np.testing.assert_allclose(mean[i], Y[:, i].mean(), rtol=1e-6)
        np.testing.assert_allclose(scale[i], Y[:, i].std(), rtol=1e-6)


def _barriers(n=60, seed=1):
    """Forward and reverse barriers and their difference."""
    rng = np.random.default_rng(seed)
    ea_f = rng.normal(40.0, 10.0, n)
    de = rng.normal(15.0, 20.0, n)
    return ea_f, ea_f - de, de


def test_fit_pools_both_directions_of_a_pair():
    ea_f, ea_r, de = _barriers()
    t = ParityTargetTransform.fit(np.c_[ea_f, de], paired_difference={0: 1})
    both = np.r_[ea_f, ea_r]
    mean, scale = t.mean[0].double().numpy(), t.scale[0].double().numpy()
    np.testing.assert_allclose([mean[0], scale[0]], [both.mean(), both.std()], rtol=1e-6)
    rms = np.sqrt(np.mean(de**2))
    assert mean[1] == 0
    np.testing.assert_allclose(scale[1], rms, rtol=1e-6)
    np.testing.assert_allclose(t.difference_scale.item(), rms, rtol=1e-6)


def test_fit_scales_a_pair_alike_whichever_columns_are_given():
    ea_f, _, de = _barriers()
    de[:5] = np.nan  # rows with the forward barrier only
    ea_r = ea_f - de
    fd = ParityTargetTransform.fit(np.c_[ea_f, de], paired_difference={0: 1})
    fr = ParityTargetTransform.fit(np.c_[ea_f, ea_r], paired_reverse={0: 1})
    rf = ParityTargetTransform.fit(np.c_[ea_r, ea_f], paired_reverse={1: 0})
    for t in (fr, rf):
        torch.testing.assert_close(t.difference_scale, fd.difference_scale)
    torch.testing.assert_close(fr.mean[:, 0], fd.mean[:, 0])
    torch.testing.assert_close(fr.scale[:, 0], fd.scale[:, 0])
    torch.testing.assert_close(fr.scale[:, 1], fr.scale[:, 0])
    torch.testing.assert_close(rf.mean[0], fr.mean[0].flip(0))


def test_fit_rejects_a_reverse_pair_with_no_row_holding_both_labels():
    Y = np.c_[[1.0, np.nan], [np.nan, 2.0]]
    with pytest.raises(ValueError, match="no row has both labels"):
        ParityTargetTransform.fit(Y, paired_reverse={0: 1})


def test_fit_ignores_missing_values():
    Y = _Y()
    Y[::3, 0] = np.nan
    t = ParityTargetTransform.fit(Y, odd=[1])
    np.testing.assert_allclose(t.mean[0, 0].item(), np.nanmean(Y[:, 0]), rtol=1e-6)
    assert torch.isfinite(t.scale).all()


def test_fit_gives_constant_columns_unit_scale():
    Y = np.ones((10, 2))
    Y[:, 1] = 0.0
    t = ParityTargetTransform.fit(Y, odd=[1])
    assert t.scale.tolist() == [[1.0, 1.0]]


def test_fit_rejects_a_column_without_labels():
    Y = _Y()
    Y[:, 3] = np.nan
    with pytest.raises(ValueError, match=r"\[3\] have no finite target values"):
        ParityTargetTransform.fit(Y, paired_difference={0: 3})


def test_fit_rejects_non_2d_targets():
    with pytest.raises(ValueError, match="2-D"):
        ParityTargetTransform.fit(np.zeros(5))


def test_fit_rejects_infinite_targets():
    Y = _Y()
    Y[4, 2] = np.inf
    with pytest.raises(ValueError, match=r"\[2\] hold infinite"):
        ParityTargetTransform.fit(Y, odd=[1])


def test_standard_scaler_normalizes_like_the_transform():
    Y = _Y()
    Y[0, 1] = np.nan
    t = ParityTargetTransform.fit(Y, odd=[1], paired_difference={0: 3})
    scaler = t.to_standard_scaler()
    assert isinstance(scaler, StandardScaler)
    Z = scaler.transform(Y)
    expected = (Y - t.mean.double().numpy()) / t.scale.double().numpy()
    np.testing.assert_allclose(Z, expected, rtol=1e-12)
    assert np.isnan(Z[0, 1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_standard_scaler_from_a_transform_on_the_gpu():
    t = ParityTargetTransform.fit(_Y(), odd=[1]).cuda()
    scaler = t.to_standard_scaler()
    np.testing.assert_allclose(scaler.scale_, t.scale[0].cpu().double().numpy())


# ---------------------------------------------------------------------------------------------
# forward


def test_unpaired_predictions_stay_scaled_in_train_mode_and_are_unscaled_in_eval_mode():
    t = ParityTargetTransform(3, odd=[0], mean=[0.0, 2.0, 3.0], scale=[2.0, 3.0, 4.0])
    Z = torch.randn(5, 3)
    torch.testing.assert_close(t.train()(Z), Z, rtol=0, atol=0)
    torch.testing.assert_close(t.eval()(Z), Z * t.scale + t.mean, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "powers", [(1, 2), (1, 0, 0, 2), (1, 1)], ids=["mve", "evidential", "quantile"]
)
def test_eval_mode_unscales_uncertainty_components(powers):
    t = ParityTargetTransform(2, odd=[0], mean=[0.0, 4.0], scale=[2.0, 3.0]).eval()
    Z = torch.rand(5, 2, len(powers)) + 0.5
    out = t(Z, powers)
    torch.testing.assert_close(out[..., 0], Z[..., 0] * t.scale + t.mean, rtol=0, atol=1e-6)
    for j, p in enumerate(powers[1:], start=1):
        torch.testing.assert_close(out[..., j], Z[..., j] * t.scale**p, rtol=0, atol=1e-6)


PAIRS = dict(paired_difference={0: 1}, paired_reverse={2: 3})
PAIR_SCALING = dict(
    mean=[10.0, 0.0, 30.0, 30.0], scale=[4.0, 6.0, 5.0, 5.0], difference_scale=[6.0, 8.0]
)


def _pair_transform():
    return ParityTargetTransform(4, **PAIRS, **PAIR_SCALING).double()


def test_pairs_are_their_mean_plus_or_minus_half_the_difference():
    t = _pair_transform().eval()
    H = torch.randn(7, 4, dtype=torch.float64)  # heads: means in 0 and 2, differences in 1, 3
    out = t(H)
    B0, D0 = H[:, 0] * 4.0 + 10.0, H[:, 1] * 6.0
    B1, D1 = H[:, 2] * 5.0 + 30.0, H[:, 3] * 8.0
    torch.testing.assert_close(out[:, 0], B0 + D0 / 2)
    torch.testing.assert_close(out[:, 1], D0)
    torch.testing.assert_close(out[:, 2], B1 + D1 / 2)
    torch.testing.assert_close(out[:, 3], B1 - D1 / 2)
    # the scaled (train) and unscaled (eval) paths agree
    torch.testing.assert_close(out, t.train()(H) * t.scale + t.mean)


def test_pairs_with_uncertainty_components_need_the_reversed_pass():
    t = _pair_transform()
    with pytest.raises(ValueError, match="reversed pass"):
        t(torch.rand(7, 4, 2, dtype=torch.float64), (1, 2))


def test_implied_member_of_targets():
    t = _pair_transform()
    Y = torch.tensor([[50.0, 12.0, 40.0, 25.0], [np.nan, 12.0, 40.0, np.nan]], dtype=torch.float64)
    implied = t.implied((Y - t.mean) / t.scale)
    # the first pair implies its reverse value, scaled like its forward column; the second pair
    # implies its difference, scaled by its difference_scale
    expected = torch.tensor([(38.0 - 10.0) / 4.0, 15.0 / 8.0], dtype=torch.float64)
    torch.testing.assert_close(implied[0], expected)
    assert implied[1].isnan().all()  # one label is not enough


def test_implied_member_of_predictions_is_consistent_with_the_pair():
    t = _pair_transform().train()
    H = torch.randn(7, 4, dtype=torch.float64)
    implied = t.implied(t(H))
    out = t.eval()(H)
    torch.testing.assert_close(implied[:, 0] * 4.0 + 10.0, out[:, 0] - out[:, 1])  # reverse
    torch.testing.assert_close(implied[:, 1] * 8.0, out[:, 2] - out[:, 3])  # difference


def test_forward_does_not_mutate_its_input():
    t = ParityTargetTransform(2, paired_difference={0: 1}).eval()
    Z = torch.randn(3, 2)
    Z0 = Z.clone()
    t(Z)
    torch.testing.assert_close(Z, Z0, rtol=0, atol=0)


def test_state_dict_holds_only_the_scaling():
    pair = dict(paired_difference={0: 1})
    t = ParityTargetTransform(2, **pair, mean=[1.0, 0.0], scale=[2.0, 3.0])
    assert set(t.state_dict()) == {"mean", "scale", "difference_scale"}
    assert "paired_difference={0: 1}, paired_reverse={}" in repr(t)
