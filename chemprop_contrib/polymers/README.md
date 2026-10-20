# `polymers`

A chemprop v2 implementation of the weighted directed message passing neural network (wD-MPNN) for modelling polymers, originally proposed by Aldeghi and Coley [1] in [polymer-chemprop](https://github.com/coleygroup/polymer-chemprop) for chemprop v1. The original port to chemprop v2 was done by Marc Short in [chemprop/chemprop#1214](https://github.com/chemprop/chemprop/pull/1214). The conversion to a `chemprop-contrib` submodule was done by Nathan Morgan with help from Claude Code.

The hyperparameter optimization code was not convereted over due to current issues with Ray Tune.

## Method

A polymer is represented as a weighted directed graph of its repeating unit. Weighting is added to the underlying MPNN:

 - the initial edge-based message is calculated and multiplied by the bond weight (`E_w`), which is the probability of the bond occurring in the repeat unit;
 - before aggregation, each node's (atom's) features are multiplied by the corresponding atom weight (`V_w`), which comes from the monomer stoichiometry;
 - mean aggregation divides by the sum of the atom weights rather than the number of atoms;
 - the graph representation is multiplied by `1 + log10(Xn)`, where `Xn` is the degree of polymerisation.

The `PolymerMolGraphFeaturizer` generates the atom and bond weights, as well as the degree of polymerisation, from the polymer string representation described in [1].

## Installation

This package was designed to work with `chemprop==2.3.1`.

## Command line interface

Installing `chemprop-contrib` provides a `chemprop-polymer` command with Chemprop's `train`, `predict` and `fingerprint` subcommands. They accept all of the usual Chemprop arguments, plus:

 - `-p`/`--polymer-columns`: Column names in the input CSV containing polymer SMILES strings

which transforms the polymer strings to the corresponding weighted molecular graphs, and trains or loads a `wMPNN`. For example, from the root of this repository:

```bash
chemprop-polymer train -i chemprop_contrib/polymers/tests/data/regression/polymer/dataset-poly_chemprop_small.csv -p poly_chemprop_input --target-columns "EA vs SHE (eV)" -o polymer_model
chemprop-polymer predict -i chemprop_contrib/polymers/tests/data/regression/polymer/dataset-poly_chemprop_small.csv -p poly_chemprop_input --model-path polymer_model/model_0/best.pt
```

Notes:

 - All input columns are polymers: if `--polymer-columns` is not given, the first column is used (as Chemprop does for SMILES), and `-s`/`--smiles-columns` is treated the same as `--polymer-columns`.
 - Only one polymer column is supported. `--reaction-columns`, atom/bond targets (`--mol-target-columns`, `--atom-target-columns`, `--bond-target-columns`), `--use-cuikmolmaker-featurization` and `--from-foundation` are not supported for polymers and raise an error.
 - `hpopt` is not provided.

## Differences from the original PR

The code is copied from the PR with imports changed from `chemprop.*` to `chemprop_contrib.polymers.*`, except for the changes below.

Library:

 - `PolymerDatapoint` subclasses `MoleculeDatapoint`. The PR built it from mixins introduced by its refactor of Chemprop's `datapoints.py`.
 - `build_dataloader` is a copy of Chemprop's with the PR's `PolymerDataset` branch added, instead of a change to `chemprop.data.build_dataloader`.
 - The weighted mean is computed in `wMPNN.fingerprint` when the aggregation is a `MeanAggregation`, instead of in a modified `MeanAggregation.forward`. The other aggregations ignored the atom weights in the PR and still do.
 - Bug fix: the featurizer's branch for inputs without atoms created `V_w` with `atom_fdim` entries instead of only one weight, so it always raised an error.
 - Unchanged: the check that the incoming bond weights of each connection point sum to 1 never runs, as in the PR and polymer-chemprop (`np.isclose(...) is False` is never true). It is not enabled because it counts self-loop rules such as `1-1` twice, so it would reject most of the example data.

Tests and docs:

 - The datapoints example uses `edge_rules`, which the PR's tutorial called `edges`.

## References

[1] M. Aldeghi and C. W. Coley, "A graph representation of molecular ensembles for polymer property prediction", Chem. Sci., 2022, 13, 10486-10498, doi: [10.1039/D2SC02839E](https://doi.org/10.1039/D2SC02839E).

Please cite [1] when using this package. The example data is a subset of the vinyl copolymer dataset from [1] (see [`tests/data_providence.txt`](tests/data_providence.txt)).
