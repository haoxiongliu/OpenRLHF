import torch
from torch import Tensor

def reward_func(queries, prompts, labels):
    # queries is prompts + responses
    nega_query_lens = [-len(q)/100.0 for q in queries]
    return Tensor(nega_query_lens)

if __name__ == "__main__":
    queries = ["I am a prompt\n\n" + "I am a response"]
    prompts = ["I am a prompt"]
    labels = ["I am an answer"]

    print(reward_func(queries, prompts, labels))