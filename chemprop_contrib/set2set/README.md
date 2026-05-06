# `Set2Set Aggregation`

This `chemprop-contrib` package implements the Set2Set aggregation method [1].

See `test_set2set.py` for example usage, which is broadly the same as typical Chemprop aggregation methods except it requires the input dimension to be explicitly defined and returns twice the input dimension as output.

The Set2Set aggregation operator performs the following operations:

```math
\begin{matrix}
\mathbf{q}_t = \mathrm{LSTM}(\mathbf{q}^{*}_{t-1}) \\  
\alpha_{i,t} = \mathrm{softmax}(\mathbf{h}_v \cdot \mathbf{q}_t) \\  
\mathbf{r}_t = \sum_{i=1}^N \alpha_{i,t} \mathbf{h}_v \\  
\mathbf{q}^{*}_t = \mathbf{q}_t \, \Vert \, \mathbf{r}_t
\end{matrix}
```

where $\mathbf{q}^{*}_T$ defines the output of the layer with twice
the dimensionality as the input.

```
Parameters
----------
in_channels : int
    The size of each input sample.
processing_steps : Optional[int], default=6
    The number of processing steps.
n_layers : Optional[int], default=3
    The number of recurrent LSTM layers.
```

## References
[1] O. Vinyals, S. Benigio, M. Kudlur, "Order Matters: Sequence to sequence for sets", February 2016, doi: [10.48550/arXiv.1511.06391](https://doi.org/10.48550/arXiv.1511.06391).
