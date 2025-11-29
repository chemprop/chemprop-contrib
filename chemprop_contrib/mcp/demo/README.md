# `demo`

This brief demo code is provided to accompany the short manuscript describing `chemprop-mcp`.

The `baseline` run was done using the LLM interface, as shown in the accompanying [video](./chemprop_mcp_demo.mp4).

The `llm_opt` was also done using the LLM interface, as described in the [paper](https://doi.org/10.26434/chemrxiv-2025-tsx5s).
The model is prompted with the below input until the shown table is completely filled out:

```
Here is the current status of hyperparameter optimization
| Trial Number | message_hidden_dim | ffn_hidden_dim | depth | Result (test MSE) |
| --- | --- | --- | --- | --- |
| 1 (baseline) | 300 | 300 | 3 | 0.78 |
| 2 | 400 | 400 | 4 | 0.7713 |
| 3 | 500 | 500 | 5 | 0.7771 |
| 4 | 400 | 400 | 5 | 0.7869 |
| 5 | 400 | 400 | 3 | 0.7847 |
| 6 |  |  |  |  |
| 7 |  |  |  |  |
| 8 |  |  |  |  |

Suggest and run a new trial for trial based on the results of the current trials to try and improve the test MSE, then add its MSE to the table. Ensure that you set random seeds to make the results reproducible, and save the output to a new subfolder in the previous one which is named to reflect the trial number
```

The `optuna` run was done using the typical Chemprop CLI, the commands for which are in `run.sh`.

Finally, `prepare_data.py` contains instructions and code to ready the data for training and testing (which should be done before running the LLM) and `parity.py` generates parity plots for the predictions (run `python parity.py` for usage instructions).
