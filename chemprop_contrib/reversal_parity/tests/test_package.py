def test_package_is_importable_from_namespace():
    from chemprop_contrib import reversal_parity

    assert {
        "ParityCGRFeaturizer",
        "reverse_bmg",
        "stack_reversed_bmg",
        "ParityEquivariantMPNN",
        "ParityTargetTransform",
    } <= set(reversal_parity.__all__)
    for name in reversal_parity.__all__:
        assert hasattr(reversal_parity, name)


def test_package_is_listed_in_contrib_namespace():
    import chemprop_contrib

    assert "reversal_parity" in chemprop_contrib.__all__
