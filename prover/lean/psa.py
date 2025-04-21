"""
Implementation of the proposal structure analysis. 
We do not follow what we do for Isabelle in the ProofAug paper,
since lean tactics can be time-consuming
Rather, we first infer from the proposal the whole structure
(It is like AST tree but for proposal there is no guarantee we can compile)

"""

from __future__ import annotations
from typing import Optional
from prover.utils import remove_lean_comments, is_statement
import re

class Conjecture(object):
    def __init__(self, parent: Optional[Conjecture], conjecture: str):
        self.conjecture = conjecture
        self.parent = parent
        self.level = parent.level + 1 if parent else 0
        self.proof = "" # type: str
        self.children = []
        Conjecture(self.proof)

    def __repr__(self):
        return f"Conjecture(level={self.level}, conjecture={self.conjecture})"



class ProposalStructure(object):
    def __init__(self, proposal: str):
        self.proposal = proposal

        self._analyze(proposal)

    def _analyze(self, proposal: str):
        proposal = remove_lean_comments(proposal)
        lines = proposal.split("\n")
        # maybe no need of lines. we have pos
        indent2parent = {0: None}
        current_blocks = []
        for i, line in enumerate(lines):
            n_indent = len(line) - len(line.lstrip())
            if is_statement(line):
                assert re.search(r'(=|\b)by\b', line)
                after_by = line.split("by")[-1].strip()
                if after_by != "":
                    tactic = after_by
                Conjecture(indent2parent[n_indent], line)
            # search [=\b]by\b
                if re.search(r'[=\b]by\b', line):
                    # it's a tactic line
                    pass
        # idea: find all have .. := by blocks (with dotall)
        #  
        # not OK. since have only creates goals. not required to immediately proved
# TODO: verify semi-proof validness in LeanServer