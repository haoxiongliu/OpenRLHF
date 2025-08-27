"""
Draw histograms of the tactic usage statistics.
Output the top tactic usage rate (and the sum of top 10).
To figure out if 'tactic-overfitting' really happen for ProofAug+RL.
results/pset_test/0812-q2515bi-pset10k-sft-pset140k-n8-rloo-3090-bs64-mix6-remove-0809-3072-kl0.0-ch0.3/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl
"""

import re
from prover.utils import PROOF_PATTERN, load_jsonl_objects
import fire
from collections import defaultdict

def extract_line_tactic(line: str) -> list[str|None]:
    """
    Extract the tactic from the line.
    """
    line.lstrip()
    words = line.split(" ")
    components = []
    for i, word in enumerate(words):
        if word == "<;>" and i == 0:
            continue
        elif word == "try" or ";" in word:
            continue
        else:
            components.append(word)
            break
    
    return " ".join(components)
    
def main(record_fp=None):
    if record_fps is None:
        record_fps = ["results/pset_test/0812-q2515bi-pset10k-sft-pset140k-n8-rloo-3090-bs64-mix6-remove-0809-3072-kl0.0-ch0.3/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl", "results/pset_test/0826-1-mix6-max2depth-cons-0821-plmo-single-n8-rloo-3072-kl0.0-cl0.2-0.28-trT0.6/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl"]
    
    
    
    # Process each file
    for record_fp in record_fps:
        tactic2freq = defaultdict(int)
        data = load_jsonl_objects(record_fp)
        for item in data:
            full_code = item["full_code"][0]
            full_code = "" if not full_code else full_code
            if not re.match(PROOF_PATTERN, full_code):
                continue
            m = re.match(PROOF_PATTERN, full_code)
            if not m:
                continue
            body = m.group('suffix')
            lines = body.strip().split("\n")
            for line in lines:
                tactic = extract_line_tactic(line)
                if tactic:
                    tactic2freq[tactic] += 1
    
        tactic2freq = sorted(tactic2freq.items(), key=lambda x: x[1], reverse=True)
        print(tactic2freq)

if __name__ == "__main__":
    fire.Fire(main)