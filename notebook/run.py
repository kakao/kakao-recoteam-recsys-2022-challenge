import sys
import joblib
import fire
sys.path.append('..')
from os.path import exists
from module.models.pcos import PCos
from module.models.ensemble import EnsembleLogit
from glob import glob

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval


def tuning(model, dl, num_iters=10):
    best_weights = None
    best_mrr = -1e+5

    def objective(params):
        nonlocal best_mrr, best_weights
        model.set_weights(params)
        hit, mrr = model.validate(dl, fork_fallback=True, tqdm_off=True)
        if best_mrr < mrr:
            best_weights = params
            best_mrr = mrr
        return -mrr

    # define a search space
    # space = hp.choice('weights',[('c', hp.randint('c', 10)) for k in model.models.keys()])
    space = {k: hp.randint(k, 5.) for k in model.models.keys()}

    best = fmin(
        fn=objective, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals=num_iters # Number of optimization attempts
    )
    print(best)
    return best_weights, best_mrr


def main(num_iters=10, folder='../processed'):
    folder = '../processed'
    kind = 'val' 
    test_logit_fnames = [
        ('bert', '../logits/te/bert4rec_1656675057269.logits'),
        ('mlp', '../logits/te/mlp-noshuffle-1657090556'),
        ('gru', '../logits/te/gru_1657082688.logits'),
        ('gnn', '../logits/te/gnn_1657082648.logits'),
        ('grun', '../logits/te/grun_1657082542.logits'),
        ('grun_all', '../logits/te/grun_all_1657082589.logits'),
    ]
    logit_fnames = [
        ('bert', '../logits/val/bert4rec_1656675057269.logits'),
        ('mlp', '../logits/val/mlp-noshuffle-1657085862'),
        ('gru', '../logits/val/gru_1657082264.logits'),
        ('gnn', '../logits/val/gnn_1657082302.logits'),
        ('grun', '../logits/val/grun_1657082495.logits'),
        ('grun_all', '../logits/val/grun_all_1657082442.logits'),
    ]
    for x in ['val', 'te']:
        for save_fname in [f'logits/{x}/pcos.logits', f'logits/{x}/pcos.similarity']:
            if exists(save_fname): continue
            pcos = PCos(dir_name=folder, save_fname=save_fname)
            pcos.fit('similarity' not in save_fname, x)
    logit_fnames.append(('pcos', f'logits/val/pcos.logits', 5.0))
    model = EnsembleLogit(logit_fnames, folder=folder, device='cuda')
    model.set_fallback(folder=folder, kind=kind)
    model.set_pcos(joblib.load(f'logits/{kind}/pcos.similarity'))
    dl = model.get_dataloader(512, folder, kind=kind)
    ret = tuning(model, dl, num_iters)
    print(f'Best params: {ret[0]}')
    print(f'Best MRR: {ret[1]}')
    
    kind = 'te'
    test_logit_fnames.append(('pcos', f'logits/{kind}/pcos.logits', 5.0))
    model = EnsembleLogit(test_logit_fnames, folder=folder, device='cuda')
    model.set_fallback(folder=folder, kind=kind)
    model.set_pcos(joblib.load(f'logits/te/pcos.similarity'))
    model.set_weights(ret[0])
    dl = model.get_dataloader(512, folder, kind=kind)
    hit, mrr = model.validate(dl, fork_fallback=True)
    print(f'HIT: {hit}, MRR: {mrr}')


if __name__ == '__main__':
    fire.Fire(main)