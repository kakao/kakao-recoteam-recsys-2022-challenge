# Recsys Challenge 2022

https://www.recsyschallenge.com/2022/

### Experimental Environments:
- We tested on Ubuntu 18.04
- We tested on Python==3.9.1, torch==1.11.0 with cuda 11.3
- GCC>=7.2.1 is required since BERT4REC dataset builder uses C++-17
### Before Running
- Making Dataset (Download [link](https://www.dressipi-recsys2022.com/profile/download_dataset)):
    put unzipped data(`candidate_items.csv`, `train_purchase.csv`, ...,) in the directory `./data`
- Install Dependencies:
```python
    pip3 install -r requirements.txt
```

- Install Torch-Geometric
We used Python-1.11.0 and Cuda 11.3. See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for details
```bash
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html 
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html 
pip3 install torch-geometric 
```

We have several models: MLP, GRU, GNN, GRU + GNN (grun), and PCos.
### Preprocessing
* For submission (leaderboard, final)
```
python preprocessing.py --submit=True
```


We used sessions/purchases in last month in train data as validation/test for internal validation.
If we set `--submit=False`, we build preprocessed for internal validation.

```
python preprocessing.py --submit=False
```

### Model Training

Train each model and save model weights. (`$bash ./run.sh`)
> Note that the save_fname rule is as follows.
> `{model_name}-{extra_kwargs_1}-{extra_kwargs_2}-{extra_kwargs_k}.pt`
```python
#!/bin/bash
mkdir save
for i in {1..20}
do
    python module/models/gru.py --submit=True --num_workers=4 --save_fname=save/gru.pt --seed=$i
    python module/models/gnn.py --submit=True --num_workers=4 --save_fname=save/gnn.pt --seed=$i
    python module/models/grun.py --submit=True --num_workers=4 --save_fname=save/grun.pt --seed=$i
    python module/models/grun.py --submit=True --num_workers=4 --all=True --save_fname=save/grun-all.pt --seed=$i
    python module/models/mlp.py --submit=True --num_workers=4 --augmentation=True --save_fname=save/mlp-augmentation.pt --seed=$i
done
```

BERT4REC has dependencies on our internal workflow so we removed the dependencies and included it in a separate directory(`./BERT4REC`). See [BERT4REC/README.md](/BERT4Rec/README.md).
> The logit file `logits/*/bert4rec_{timestamp}.logit` is generated from the BERT4REC model.
> Model ensemble below can be run without BERT4REC yet we achieved best score with BERT4REC.

### Ensemble Models
Saved models are as follows.

```python
In [1]: from glob import glob
In [2]: saves = glob('save/*.pt*')
In [3]: saves
Out[3]:
['save/gru.pt.1656302353',
 'save/gnn.pt.1656302415',
 'save/grun.pt.1656302622',
 'save/grun-all.pt.1656304624'
 'save/mlp-augmenation.pt.1656303446', ...]
```

Generate logits and ensemble all of them to generate recommended lists for leaderboard sessions.
> Please note that predictions are made from [parsed logit files](/module/models/ensemble.py#L195-L206), so training process should be done beforehand.
> Note that BERT4REC itself generates logits.

```
python module/models/ensemble.py --fork_fallback=True --submit=True --kind=leader  # recommendations for leaderboard sessions
python module/models/ensemble.py --fork_fallback=True --submit=True --kind=final  # recommendations for final sessions
```

`submit` option determines the data we use to make ensemble.
- If `--submit=False`  is set to true, it generates logits for internal validation. Otherwise, it generates logits for submission.

`kind` option determines the target for recommendation results: `leader` and `final`. (`val` and `te` for internal validation)
- If `--kind=leader`, it generates recommendation results on leaderboard sessions. 
- If `--kind=final`, it generates recommendation results on final sessions.
- If `kind` is not set, it will default to `val`.


### Contributors
* iggy.ll@kakaocorp.com
* tony.yoo@kakaocorp.com
* joel.j@kakaocorp.com
* andrew.y@kakaocorp.com
