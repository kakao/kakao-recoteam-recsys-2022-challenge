#!/bin/bash
mkdir save
for i in {1..5}
do
    python module/models/gru.py --submit=True --num_workers=4 --save_fname=save/gru.pt --seed=$i
    python module/models/gnn.py --submit=True --num_workers=4 --save_fname=save/gnn.pt --seed=$i
    python module/models/grun.py --submit=True --num_workers=4 --save_fname=save/grun.pt --seed=$i
    python module/models/grun.py --submit=True --num_workers=4 --all=True --save_fname=save/grun-all.pt --seed=$i
done
