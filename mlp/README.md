# How to run train script

### 0. Prepare Data
Check all dataset prepared
- 1. validation data in "../parsed_data"
- 2. all data in "../data"
- 3. dataset must be processed since indices of items need to be kept with other models;

### 1. Run Jupyter Notebook named

`1-submit-mlp-build-data.ipynb` for submission
`1-val-mlp-build-data.ipynb` for augmentation

### 2. Model Learning & prediction
We used two MLP models (identical, but datlaoder is shuffled or not shuffled when training)

```
2-submit-mlp-model-training-no-shuffle.ipynb  # train submission model with no data shuffle
2-submit-mlp-model-training-with-shuffle.ipynb  # #train submission model with data shuffle

2-val-mlp-model-training-no-shuffle.ipynb  # train internal validation model with no data shuffle
2-val-mlp-model-training-with-shuffle.ipynb  # train internal validation model with data shuffle
```

### 3. Generating Logits (validation only)
See `3-val-mlp-pred.ipynb`. For submission data, we generate logits in `2-submit-mlp-model-training-no-shuffle.ipynb`, and `2-submit-mlp-model-training-with-shuffle.ipynb`.
Training results are saved in
`../logits/{val, test, leader, final}/mlp/{mlp-no-shuffle, mlpshuffle}_{timestamp}.logit`
Stored logits are used in next ensemble step.
