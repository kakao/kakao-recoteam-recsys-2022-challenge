import torch


def get_mrr(pred, pos, topk, mean=False):
    top_hit = (pred[:, :topk] == pos.unsqueeze(1))
    div = torch.arange(1, 1 + topk).unsqueeze(0).to(pred.device)
    mrr = (top_hit / div).sum(-1)
    mrr = mrr.detach().cpu().numpy()
    if mean:
        return mrr.mean()
    return mrr

def get_hit(pred, pos, topk, mean=False):
    hit = (pred[:, :topk] == pos.unsqueeze(1)).sum(-1)
    hit = hit.detach().cpu().numpy()
    if hit:
        return hit.mean()
    return hit

def get_hit_and_mrr(pred, pos, topk, mean=False):
    top_hit = (pred[:, :topk] == pos.unsqueeze(1)).type(torch.float)
    hit = top_hit.sum(-1).cpu().numpy()
    div = torch.arange(1, 1 + topk).unsqueeze(0).to(pred.device)
    mrr = (top_hit / div).sum(-1)
    mrr = mrr.detach().cpu().numpy()
    if mean:
        return hit.mean(), mrr.mean()
    return hit, mrr