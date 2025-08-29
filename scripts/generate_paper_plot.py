from matplotlib import pyplot as plt
import json
import os
import fire


label2name = {
    "ppo": "0820-q2515bi-pset10k-sft-pset10k-n8-rloo-3090-bs64-record_pa_reward-mix6-0808-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo+cons": "0821-q2515bip10k-n8-rloo-bs64-mix6-max2depth-cons-0821-3072-kl0.0-cl0.2-0.28-trT0.6",
    "ppo+aggr": None,
    "ppo+direct": None, # preliminary
    "plmo_plain": None,
    "plmo_avg": None,
    "plmo_sum": None,
    "plmo_single": None
}



def main(log_fp="results/summary.log", output_root="results/paper_plot"):
    """
    Generate the ProofAug+ plot from the paper
    and save the pngs into output_root
    """
    
    os.makedirs(output_root, exist_ok=True)

if __name__ == "__main__":
    fire.Fire(main)