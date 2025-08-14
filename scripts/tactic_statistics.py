"""
Draw histograms of the tactic usage statistics.
Output the top tactic usage rate (and the sum of top 10).
To figure out if 'tactic-overfitting' really happen for ProofAug+RL.
results/pset_test/0812-q2515bi-pset10k-sft-pset140k-n8-rloo-3090-bs64-mix6-remove-0809-3072-kl0.0-ch0.3/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl
"""

import re
from prover.utils import PROOF_PATTERN, load_jsonl_objects
from prover.lean.psa import ProposalStructure
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
    
def main(record_fp="results/pset_test/0812-q2515bi-pset10k-sft-pset140k-n8-rloo-3090-bs64-mix6-remove-0809-3072-kl0.0-ch0.3/global_step10_hf-n1-4096-T0.6-s7-orig/full_records.jsonl"):
    data = load_jsonl_objects(record_fp)
    tactic2freq = defaultdict(int)
    for item in data:
        full_code = item["full_code"]
        if not PROOF_PATTERN.match(full_code):
            continue
        m = re.match(PROOF_PATTERN, full_code)
        if not m:
            continue
        body = m.group('body')
        prop_struct = ProposalStructure(body)
        lines = body.strip().split("\n")
        for line in lines:
            tactic = extract_line_tactic(line)
            if tactic:
                tactic_list.append(tactic)

if __name__ == "__main__":
    fire.Fire(main)