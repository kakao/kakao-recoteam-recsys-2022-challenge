# How to run train script

0. Prepare Data

```
python preprocessing.py --submit=True
python preprocessing.py --submit=False
```

DATA MUST BE PROCESSED IN parental directory using `preprocessing.py` for both 'train' and 'submit' data.

1. Install our BERT4REC implementation
```
python setup.py install
```

2. Run dataset builder

```
python basic_dataset_builder.py
python bert4rec_dataset_builder.py
```

3. Run training scripts


```
python train/recsys_train_manager.py run_all_once --conf_fname=train/conf/recsys/pretrain_submit-len20.json --finetune_conf_fname=train/conf/recsys/finetuning_submit-len20.json  # Submit
python train/recsys_train_manager.py run_all_once --conf_fname=train/conf/recsys/pretrain-len20.json --finetune_conf_fname=train/conf/recsys/finetuning-len20.json  # Internal Validation
```

Training results are saved in `../logits/*/bert4rec_{timestamp}.logit`
Stored logits are used in next ensemble step.
