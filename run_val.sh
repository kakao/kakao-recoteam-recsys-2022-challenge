#!/bin/bash
mkdir save
for i in {1..10}
do
    python module/models/gru.py --num_workers=4 --save_fname=save/gru.pt --seed=$i
    python module/models/gnn.py --num_workers=4 --save_fname=save/gnn.pt --seed=$i
    python module/models/grun.py --num_workers=4 --save_fname=save/grun.pt --seed=$i
    python module/models/grun.py --num_workers=4 --all=True --save_fname=save/grun-all.pt --seed=$i
done
