# `reversal_parity`

This `chemprop-contrib` package predicts properties of a transformation `A>>B` that behave **exactly** as they must when the transformation is reversed to `B>>A`. The transformation can be a chemical reaction or an alchemical transformation between two molecules, such as phenol → toluene. Either way it is given as atom-mapped SMILES and encoded as a Condensed Graph of Reaction (CGR).

Many such properties have a fixed behavior under reversal. A reaction enthalpy or a relative solvation free energy changes sign,

$$\Delta H(A \to B) = -\Delta H(B \to A),$$

a reorganization energy stays the same, and a forward barrier becomes the reverse barrier. A standard Chemprop model satisfies these relations only approximately, to the extent it learns them from data (for example from a training set augmented with reversed reactions). This package enforces them exactly, by construction, for each target column. You declare each column's behavior with `ParityTargetTransform`, which also scales the targets. With $x$ a transformation, $\bar{x}$ its reversal and $f_i$ the prediction in column $i$:

| Declaration | Behavior under reversal | Example target |
|---|---|---|
| `odd=[i]` | $f_i(\bar{x}) = -f_i(x)$ | $\Delta H$, $\Delta G$, $\Delta E$, $\log K_{\text{eq}}$, $\Delta\Delta G_{\text{solv}}$ |
| `even=[i]` | $f_i(\bar{x}) = f_i(x)$ | reorganization energy |
| `paired_reverse={i: j}` | columns $i$ and $j$ swap: $f_i(\bar{x}) = f_j(x)$ | forward and reverse barriers; $\log k_{\text{f}}$ and $\log k_{\text{r}}$ |
| `paired_difference={i: j}` | $f_i(\bar{x}) = f_i(x) - f_j(x)$, and column $j$ is odd | forward barrier and $\Delta E$; $\log k_{\text{f}}$ and $\log K_{\text{eq}}$ |
| not listed | unconstrained (free) | any other target |

## How it works

1. `ParityCGRFeaturizer` emits CGR features in an $(m, d)$ basis, with $m = \tfrac12(x_r + x_p)$ the mean of the reactant-side and product-side features and $d = x_p - x_r$ their difference. In this basis, reversing a transformation is a sign flip on the contiguous $d$ block, so no second featurization is needed. By default the element is written once, in $m$ only, as Chemprop does for chemical reactions (see [Shared elements](#shared-elements)). With `ParityCGRFeaturizer(share_elements=False)` every atom feature gets both parts, including the element, so a mapped atom may change element (see [Alchemical transformations](#alchemical-transformations)).
2. `ParityEquivariantMPNN` runs the same network $g$ (shared weights) on the batch and on its reversal, then combines the outputs per column: $\tfrac12[g(x) - g(\bar{x})]$ for odd, $\tfrac12[g(x) + g(\bar{x})]$ for even, $g(x)$ for free. A pair is predicted from an even mean and an odd difference (see [Pairs](#pairs-barriers-and-other-two-direction-properties)). The guarantee holds for any network $g$.

Both directions go through the encoder in a single pass over a batch of twice as many graphs (`stack_reversed_bmg`). A training step therefore costs more than in stock Chemprop, which encodes each graph once.

## Minimal example

```python
import csv

import lightning.pytorch as pl
from chemprop.data import ReactionDatapoint, ReactionDataset, build_dataloader
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

from chemprop_contrib.reversal_parity import (
    ParityCGRFeaturizer, ParityEquivariantMPNN, ParityTargetTransform,
)

# columns: forward barrier Ea, reaction energy dE (Ea - dE is the reverse barrier),
# reorganization energy (even)
cols = ["Ea", "dE", "lambda"]
with open("reactions.csv") as f:  # atom-mapped reaction SMILES in "smiles"
    rows = list(csv.DictReader(f))

featurizer = ParityCGRFeaturizer()  # not chemprop's CondensedGraphOfReactionFeaturizer
dps = [
    ReactionDatapoint.from_smi(
        r["smiles"], y=[float(r[c]) for c in cols], keep_h=True, add_h=False
    )
    for r in rows
]
n_train = int(0.8 * len(dps))
train_dset = ReactionDataset(dps[:n_train], featurizer=featurizer)
val_dset = ReactionDataset(dps[n_train:], featurizer=featurizer)

# declare each column's parity, fit the target scaling, normalize both datasets with it
target = ParityTargetTransform.fit(train_dset.Y, paired_difference={0: 1}, even=[2])
train_dset.normalize_targets(target.to_standard_scaler())
val_dset.normalize_targets(target.to_standard_scaler())

mp = BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim)
predictor = RegressionFFN(n_tasks=len(cols), input_dim=mp.output_dim)  # no output_transform
model = ParityEquivariantMPNN.from_featurizer(
    featurizer, mp, MeanAggregation(), predictor, target, batch_norm=True
)

train_loader = build_dataloader(train_dset, batch_size=8, num_workers=0)
val_loader = build_dataloader(val_dset, batch_size=8, num_workers=0, shuffle=False)
trainer = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False)
trainer.fit(model, train_loader, val_loader)

preds = trainer.predict(model, val_loader)  # list of (batch, n_tasks) tensors, unscaled
```

Reactions must be atom-mapped and built with `keep_h=True`, and the datasets must use `ParityCGRFeaturizer`. `tests/test_models.py` contains a complete training setup on a bundled fixture.

## Pairs: barriers and other two-direction properties

Reversal swaps a forward barrier with its reverse barrier, $E_a^{\text{r}}(x) = E_a^{\text{f}}(\bar{x})$, and their difference is the reaction energy, $E_a^{\text{f}} - E_a^{\text{r}} = \Delta E$. Declare such a property by mapping its forward column to whichever of the reverse column or the difference column your data has:

```python
paired_reverse={0: 1}      # forward and reverse barriers in columns 0 and 1
paired_difference={0: 1}   # forward barrier and reaction energy in columns 0 and 1
```

The model predicts the pair from an even mean $M$ and an odd difference $D$, as

$$f_{\text{fwd}} = M + \tfrac12 D, \qquad f_{\text{rev}} = M - \tfrac12 D, \qquad f_{\text{diff}} = D,$$

so $f_{\text{fwd}} - f_{\text{rev}} = f_{\text{diff}}$ holds exactly, and predicting on the reversed reaction gives the reverse barrier as its forward value. Predictions come back as the columns of your dataset.

Training scores all three members of a pair. A row with both of its labels also scores the third member, computed from them, and a row with one label scores that one. Both forms therefore train the same model on the same data, and neither direction is favored. Rows with a forward barrier alone, or with a reaction energy alone with `paired_difference`, still train. `fit` scales the two directions together, from both pooled, and the difference about zero.

The same structure fits any property that reversal maps to its partner: rate constants ($\log k_{\text{f}}$ and $\log k_{\text{r}}$, whose difference is $\log K_{\text{eq}}$), free-energy barriers, or, for an alchemical pair `A>>B`, an endpoint property such as the solvation free energy of each end state. With $\Delta G_{\text{solv}}(B)$ as the forward column and $\Delta G_{\text{solv}}(A)$ as the reverse one, the difference is the relative $\Delta\Delta G_{\text{solv}}(A \to B)$ defined in [Alchemical transformations](#alchemical-transformations). The sign convention is $f_{\text{diff}} = f_{\text{fwd}} - f_{\text{rev}}$.

### Uncertainty for pairs

Pairs work with `MveFFN`, `EvidentialFFN` and `QuantileFFN`. Each member of a pair then has its own uncertainty outputs $u$ (a variance, the evidential parameters, or a quantile interval), and reversal treats them as it treats the members:

$$u_{\text{rev}}(x) = u_{\text{fwd}}(\bar{x}), \qquad u_{\text{diff}}(\bar{x}) = u_{\text{diff}}(x).$$

The forward and reverse uncertainties are one unconstrained output of the network, evaluated on the reaction and on its reverse. Reversal therefore swaps them exactly, and they are free to differ within a reaction, as they should: the two barriers of a reaction share a transition state but start from different minima. The uncertainty of the difference is combined as even, like that of any odd task. The predictions themselves are unchanged, so $f_{\text{fwd}} - f_{\text{rev}} = f_{\text{diff}}$ still holds exactly.

Training scores all three members, each with its own uncertainty, but the model outputs only the columns of your dataset. With `paired_difference`, the uncertainty of the reverse value is the forward column's uncertainty for the reversed reaction.

Two things are not guaranteed:

- **The three uncertainties are learned separately.** Nothing forces them to be mutually consistent. For example, any joint distribution of the two barriers has $|\sigma_{\text{fwd}} - \sigma_{\text{rev}}| \le \sigma_{\text{diff}} \le \sigma_{\text{fwd}} + \sigma_{\text{rev}}$, and the three variances of `MveFFN` need not satisfy it.
- **For `QuantileFFN`, the pair relation holds between interval midpoints.** Chemprop's `QuantileFFN` reports the midpoint of each interval as the prediction. The midpoint of the difference's interval equals the difference of the barriers' midpoints only if the predictive distributions are symmetric about their midpoints, which the model assumes.

## Alchemical transformations

Any atom-mapped pair of molecules `A>>B` works as input with `ParityCGRFeaturizer(share_elements=False)`, including alchemical transformations in which mapped atoms change element. An odd task then predicts a relative property, such as the relative solvation free energy

$$\Delta\Delta G_{\text{solv}}(A \to B) = \Delta G_{\text{solv}}(B) - \Delta G_{\text{solv}}(A),$$

which is exactly antisymmetric under swapping $A$ and $B$.

The atom map alone defines the topology of the transformation, in the sense of alchemical free-energy methods. For phenol → toluene (hydrogens as counts):

| Topology | `A>>B` | What the CGR contains |
|---|---|---|
| single | `[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[OH:7]>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH3:7]` | atoms paired across the two molecules wherever possible, even between elements: the O node turns into a C in place |
| hybrid | `[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[OH]>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH3]` | the common core (the ring) shared, the groups that differ kept apart: an O node vanishes and a C node appears |

In single topology, an atom without a partner on the other side vanishes or appears, as a dummy atom does: with explicit hydrogens, `[O:7][H:8]` mapped to `[C:7]([H:8])([H])[H]` turns O into C and makes two hydrogens appear.

A chemical reaction is the special case in which no mapped atom changes element. With `share_elements=False` nothing checks atom maps for chemical plausibility, so a map that pairs atoms of different elements by mistake is featurized as an element change; with the default, it raises `ValueError`.

With `share_elements=False`, on the side where an atom is absent it is a zero dummy atom: Chemprop's atom featurizer has no null bit, and an all-zero vector (`atom_featurizer(None)`) is unambiguous, since every real atom has exactly one element bit set. A vanishing atom therefore has $d = -x_A$ and an appearing atom $d = +x_B$, and the net element change summed over the graph is the same in every topology. With `ParityCGRFeaturizer(balanced=True)`, an atom absent from one side is instead copied across unchanged (Chemprop's meaning of an unrecorded spectator in an incomplete reaction).

## Shared elements

Chemprop's CGR featurizer writes each atom's element once, since a mapped atom keeps its element in a chemical reaction. `ParityCGRFeaturizer` does the same by default (`share_elements=True`): the element one-hot appears only in the even block $m$, and the odd block $d$ leaves it out. The node features are then as wide as Chemprop's `REAC_DIFF` ones, and they are the same information in another basis: Chemprop's $(x_r, d)$ becomes $(x_r + \tfrac12 d, d)$, with the element columns of $x_r$ unchanged. An atom absent from one side keeps its element there, as in Chemprop (`atom_featurizer.num_only`), unless `balanced=True` copies it across whole.

This suits chemical reactions. Featurizing a reaction in which a mapped atom changes element raises `ValueError`, since the odd block has no element columns to record the change; pass `share_elements=False` for alchemical transformations. Unmapped atoms that vanish or appear are allowed. For a fully mapped chemical reaction the two settings carry the same information: with `share_elements=False`, $d$ is zero in the element columns, and the default drops those columns. They differ only for unmapped atoms, whose absent side carries the element by default and is a zero dummy atom with `share_elements=False`.

## Things to know

**Use `ParityCGRFeaturizer`, not Chemprop's CGR featurizer.** Chemprop's `CondensedGraphOfReactionFeaturizer` is not in the $(m, d)$ basis, and a model built for it gives predictions on real reversed reactions that are only approximately odd or even. `ParityEquivariantMPNN.from_featurizer` raises `TypeError` for any other featurizer; the plain constructor only sees dimensions and offsets and cannot check this.

**Scale targets with `ParityTargetTransform`, not the predictor.** The model combines the forward and reversed predictions in the scaled space and then unscales, so the predictor must not have its own `output_transform`, and the datasets must be normalized with `target.to_standard_scaler()`. Odd columns and the differences of pairs are scaled about zero (mean 0, root-mean-square scale): a shift is even, so it would break their parity. The two directions of a pair share one mean and scale. `fit` handles all of this, ignores missing values, and leaves even and free columns with their usual mean and standard deviation. The code cannot check that the datasets were normalized with the matching scaler, so build both from the same transform, as in the example. The model keeps its own copy of the transform, so one transform can serve several models, such as the members of an ensemble.

**Exactness holds whenever dropout is inactive.** In `eval()` every prediction is exactly odd or even (to float precision). Batch norm does not change this in train mode: the forward and reversed copies share one batch, so they see the same batch statistics. Dropout does. While it is active, the two copies get independent masks, so a single draw is only approximately odd or even. The draws are still symmetric as a distribution: those for $\bar{x}$ are distributed as the negated (odd) or unchanged (even) draws for $x$. Uncertainty from repeated dropout passes (Chemprop's dropout estimator) therefore has the declared parity up to sampling noise, not exactly. Call `model.eval()` before checking parity yourself; `trainer.predict` does so for you.

**Return to train mode before fitting again.** `trainer.predict` puts the model in eval mode. Call `model.train()` before any further `trainer.fit`: Lightning trains a module in the mode it was left in, and in eval mode the target transform unscales, so the model would fit unscaled predictions to scaled targets and fail to learn.

**Extra descriptors must be direction-invariant.** `X_d` (and per-atom `V_d`, which `ReactionDataset` does not currently produce) is passed identically to the forward and reversed passes, so it is treated as invariant under reversal. Reaction conditions such as temperature or solvent are fine. Direction-dependent descriptors (for example reactant-side-only descriptors) break exactness unless you symmetrize them or fold them into the graph.

**Atom and bond feature scaling is fine.** A `graph_transform` on the message passing runs inside the encoder, after the reversal, so it is just part of the network $g$ and exactness is unaffected.

**`fingerprint` is direction-dependent.** `model.fingerprint(...)` returns the forward-direction fingerprint only, with the usual shape. The parity structure lives in the combination of predictor outputs, not in the fingerprint.

**Loading saved models.** Use `ParityEquivariantMPNN.load_from_file` or `ParityEquivariantMPNN.load_from_checkpoint`. Chemprop's generic `MPNN.load_from_file`, `chemprop.models.utils.load_model` and the `chemprop predict` and `chemprop fingerprint` CLIs cannot rebuild this class (they raise a `TypeError` about `odd_atom_start`). **Never use `MPNN.load_from_checkpoint`:** Lightning silently drops the parity hyperparameters and returns a plain `MPNN`, whose predictions differ and have no parity guarantee.

## Unsupported combinations

Construction raises `ValueError` for:

- predictors outside `RegressionFFN`, `MveFFN`, `EvidentialFFN` and `QuantileFFN`. The uncertainty outputs (variance, evidential parameters, quantile interval) of an odd or even task are combined as even, so they stay valid; those of a free task are free, like its prediction;
- a predictor with its own `output_transform`, or a different number of tasks than the target transform.

`ParityTargetTransform` raises `ValueError` for indices out of range (floats and booleans are rejected, not truncated), a column given two roles (including a forward column in both `paired_reverse` and `paired_difference`), and a scaling that breaks the declaration (a nonzero mean on an odd or difference column, or different scalings for the two directions of a pair).

Training and prediction raise `TypeError` for a batch class that carries fields or attributes beyond those of chemprop's `BatchMolGraph`, since the stacked batch would silently leave them out.

Reaction-plus-molecule inputs (for example a reaction with a solvent, via `MulticomponentMPNN`) are not supported: `ParityEquivariantMPNN` takes a single reaction graph.

`CuikmolmakerCGRFeaturizer` is not supported, since it cannot express the $(m, d)$ layout. There is no CLI integration; use the Python API.

## Future work

Extra descriptors that flip sign under reversal, such as a reaction enthalpy used as an input, are not supported yet.

## Tests

```
pytest chemprop_contrib/reversal_parity/
```

The bundled fixture `tests/test_data.csv` is derived from Chemprop's `rxn.csv`; see `tests/data_provenance.txt`.

## Compatibility

The test suite passes with Chemprop 2.2.3 and 2.3.1. The package overrides private methods of Chemprop's CGR featurizer and reads private attributes of `BatchMolGraph`, so other versions may need changes. With Chemprop 2.2.0 and a recent PyTorch (2.15 was tried), loading a saved model fails: PyTorch's weights-only checkpoint loading rejects the stored `ParityTargetTransform`.
