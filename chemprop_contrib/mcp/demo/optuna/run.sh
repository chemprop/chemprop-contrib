chemprop hpopt \
    --data-path ../train.csv \
    --raytune-use-gpu \
    --raytune-search-algorithm optuna \
    --raytune-num-samples 8 \
    --search-parameter-keywords message_hidden_dim ffn_hidden_dim depth \
    --data-seed 42 \
    --pytorch-seed 42 \
    --task-type regression \
    --smiles-columns SmilesCurated \
    --target-columns ExperimentalLogS

chemprop predict \
    --test-path ../test.csv \
    --preds-path test_predictions.csv \
    --model-paths chemprop_hpopt/train/best_checkpoint.ckpt 
