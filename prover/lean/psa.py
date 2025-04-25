"""
Implementation of the proposal structure analysis. 
We do not follow what we do for Isabelle in the ProofAug paper,
since lean tactics can be time-consuming
Rather, we first infer from the proposal the whole structure
(It is like AST tree but for proposal there is no guarantee we can compile)

the body code is like this:

theorem mathd_algebra_114 (a : ℝ) (h₀ : a = 8) :
    (16 * (a ^ 2) ^ ((1 : ℝ) / 3)) ^ ((1 : ℝ) / 3) = 4 := by
  have ha : a ^ 2 = 64 := by
    rw [h₀]
    norm_num
  have h1 : (a ^ 2) ^ ((1 : ℝ) / 3) = 4 := by
    rw [ha]
    have h4 : (64 : ℝ) ^ ((1 : ℝ) / 3) = 4 := by
      rw [show (64 : ℝ) = 4 ^ (3 : ℝ) by norm_num]
      rw [←Real.rpow_mul]
      norm_num
      all_goals linarith
    exact h4
  have h2 : 16 * (a ^ 2) ^ ((1 : ℝ) / 3) = 64 := by
    rw [h1]
    norm_num
  rw [h2]
  have h3 : (64 : ℝ) ^ ((1 : ℝ) / 3) = 4 := by
    rw [show (64 : ℝ) = 4 ^ (3 : ℝ) by norm_num]
    rw [←Real.rpow_mul]
    norm_num
    all_goals linarith
  exact h3
"""

from __future__ import annotations
from typing import Optional
from prover.utils import remove_lean_comments, is_statement, split_header_body
import re


class Snippet(object):
    """A snippet of a lean 4 proof. Always add a newline before adding a new snippet."""
    def __init__(self, content: Optional[str] = None):
        self.content = content
    
    def receive_snippet(self, snippet: Snippet | str):
        new_content = snippet.content if isinstance(snippet, Snippet) else snippet
        if self.content:
            self.content += "\n" + new_content
        else:
            self.content = new_content

    def __repr__(self):
        return f"-- Snippet(content=\n{self.content})"

class Block(object):
    """each block is a list of snippets and subblocks"""
    def __init__(self, parent: Optional[Block]):
        self.parts = [] # type: list[Block|Snippet]
        self.subblock_indices = [] # type: list[int]
        self.parent = parent
        self.level = parent.level + 1 if parent else -1
        self.content_snapshot = None # type: str
      
    @property
    def content(self):
        return "\n".join([part.content for part in self.parts])
    
    @property
    def statement(self):
        return self.parts[0].content.split(':=')[0]

    def receive_block(self, block: Block):
        self.parts.append(block)
        self.subblock_indices.append(len(self.parts) - 1)

    def receive_snippet(self, snippet: Snippet | str):
        if not self.parts or isinstance(self.parts[-1], Block):
            self.parts.append(Snippet())
        self.parts[-1].receive_snippet(snippet)

    def __repr__(self):
        return f"-- Block(level={self.level}, content=\n{self.content})"




class ProposalStructure(object):
    def __init__(self, proposal: str):
        self.proposal = proposal
        self.root = None
        self._analyze(proposal)

    def _analyze(self, proposal: str):
        lines = proposal.split("\n")
        # determine the blocks by finding 'have' and ':=' and by the indentation
        indent2level = {}
        block_stack = [Block(parent=None)] # type: list[Block]
        in_statement = False
        for i, line in enumerate(lines):
            
            # we require that the proof never opens a new goal by 'have' when the current same level block is not yet closed
            
            if in_statement:
                if ':=' in line:
                    in_statement = False
                snippet = Snippet(line)
                block_stack[-1].receive_snippet(snippet)
                continue

            indent = len(line) - len(line.lstrip())
            if indent not in indent2level:
                indent2level[indent] = len(indent2level)
            level = indent2level[indent]
            if is_statement(line):
                for i in range(len(block_stack) - 1, -1, -1):
                    if block_stack[i].level >= level:
                        block_stack.pop()
                    else:
                        break
                last_block = block_stack[-1]
                block = Block(parent=last_block)
                last_block.receive_block(block)
                block_stack.append(block)
                in_statement = True
            else:
                for i in range(len(block_stack) - 1, -1, -1):
                    if block_stack[i].level >= level:
                        block_stack.pop()
                    else:
                        break    
                last_block = block_stack[-1]
            block_stack[-1].receive_snippet(Snippet(line))
            if ':=' in line:
                in_statement = False
        self.root = block_stack[0]
    
    def _traverse_blocks(self, block: Block):
        # we know that things happen in this block.
        # when error happens, 
        for part in block.parts:
            if isinstance(part, Block):
                self._traverse_blocks(part)
            else:
                print(part.content)

    def traverse(self):
        self._traverse_blocks(self.root)
