import torch
from torch import Tensor

def reward_func(queries, prompts):
    # queries is prompts + responses
    nega_query_lens = [-len(q)/100.0 for q in queries]
    return Tensor(nega_query_lens)
