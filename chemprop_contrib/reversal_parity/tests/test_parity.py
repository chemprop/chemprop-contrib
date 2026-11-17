import pytest
import torch

from chemprop.nn.predictors import (
    BinaryClassificationFFN,
    EvidentialFFN,
    MulticlassClassificationFFN,
    MveFFN,
    QuantileFFN,
    RegressionFFN,
    SpectralFFN,
)

from chemprop_contrib.reversal_parity.parity import EVEN, FREE, ODD, build_parity_mask


def test_regression_mask_is_1d():
    mask = build_parity_mask([ODD, EVEN, FREE], RegressionFFN(n_tasks=3))
    assert mask.shape == (3,)
    assert mask.is_floating_point()
    assert torch.equal(mask, torch.tensor([-1.0, 1.0, 0.0]))


def test_mve_mask_is_2d_with_even_uncertainty():
    mask = build_parity_mask([ODD, ODD], MveFFN(n_tasks=2))
    assert mask.shape == (2, 2)
    assert torch.equal(mask, torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]))


def test_evidential_mask_shape_and_values():
    mask = build_parity_mask([ODD], EvidentialFFN(n_tasks=1))
    assert mask.shape == (1, 4)
    assert torch.equal(mask, torch.tensor([[-1.0, 1.0, 1.0, 1.0]]))


def test_quantile_uses_the_generic_rule():
    mask = build_parity_mask([ODD], QuantileFFN(n_tasks=1))
    assert mask.shape == (1, 2)
    assert torch.equal(mask, torch.tensor([[-1.0, 1.0]]))


def test_free_task_has_free_uncertainty_components():
    # nothing relates a free task's values for x and its reversal, so nothing relates their
    # uncertainties either
    mask = build_parity_mask((FREE, EVEN), MveFFN(n_tasks=2))
    assert torch.equal(mask, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))


def test_listed_tasks_get_free_uncertainty_components():
    mask = build_parity_mask((EVEN, ODD), QuantileFFN(n_tasks=2), free_uncertainty=[0])
    assert torch.equal(mask, torch.tensor([[1.0, 0.0], [-1.0, 1.0]]))


@pytest.mark.parametrize(
    "predictor",
    [
        BinaryClassificationFFN(n_tasks=1),
        MulticlassClassificationFFN(n_classes=3, n_tasks=1),
        SpectralFFN(n_tasks=1),
    ],
    ids=lambda p: type(p).__name__,
)
def test_rejects_predictors_outside_allowlist(predictor):
    with pytest.raises(ValueError, match=type(predictor).__name__):
        build_parity_mask([ODD], predictor)


def test_rejects_regression_subclass_not_in_allowlist():
    class Custom(RegressionFFN):
        pass

    with pytest.raises(ValueError, match="Custom"):
        build_parity_mask([ODD], Custom(n_tasks=1))


def test_rejects_wrong_length_task_parity():
    with pytest.raises(ValueError, match="length"):
        build_parity_mask([ODD, EVEN], RegressionFFN(n_tasks=3))


@pytest.mark.parametrize("bad", [2, -1.0, True, "odd"])
def test_rejects_invalid_parity_value(bad):
    with pytest.raises(ValueError, match="Invalid parity values"):
        build_parity_mask([ODD, bad], RegressionFFN(n_tasks=2))
