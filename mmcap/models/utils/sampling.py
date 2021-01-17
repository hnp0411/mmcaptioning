import random
from functools import partial

import torch


def topktopp(pred, topk, topp):
    # topk
    topkfunc = partial(torch.topk,
                       k=topk,
                       dim=-1,
                       largest=True)

    topk_values = topkfunc(pred).values.squeeze(dim=0)
    topk_indices = topkfunc(pred).indices.squeeze(dim=0)

    # topp
    sum_ = 0
    candidate_indice_list = list()
    for val, ind in zip(topk_values, topk_indices):
        candidate_indice_list.append(ind.item())
        sum_ += val.item()
        if sum_ > topp:
            break
    return random.sample(candidate_indice_list, 1)
