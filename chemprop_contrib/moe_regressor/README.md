# `moe_regressor`

This `chemprop-contrib` package implements the Adaptive Mixture of Local Experts [1] model for regression tasks.
See `test.py` for example usage, which is broadly the same as a typical Chemprop workflow.

The method works by passing the learned representation from message passing into one "gating network" and a configurable number of "experts".
The outputs of the individual experts are multiplied element-wise by the output of the gating network, enabling the overall architecture to 'specialize' experts in certain types of inputs dynamically during training.

This network is useful in and of itself (see, for example, this [highly performant aqueous solubility prediction code using it](https://github.com/JacksonBurns/chemeleon_aqueous_solubility)) but is also a good demonstration of how to make a contribution to Chemprop contrib.
Anyone seeking to make a contribution should model it off what is done here, and the associated [Pull Request](https://github.com/chemprop/chemprop-contrib/pull/2).

## References
[1] R. A. Jacobs, M. I. Jordan, S. J. Nowlan and G. E. Hinton, "Adaptive Mixtures of Local Experts" Neural Computation, vol. 3, no. 1, pp. 79-87, March 1991, doi: [10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79).
