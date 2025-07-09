"""
Implementation of the proposal structure analysis. 

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

the root.parts[0].parts is like

[-- Snippet(content=
theorem mathd_algebra_114 (a : ℝ) (h₀ : a = 8) :
    (16 * (a ^ 2) ^ ((1 : ℝ) / 3)) ^ ((1 : ℝ) / 3) = 4 := by), 
-- Block(level=1, content=
  have ha : a ^ 2 = 64 := by
    rw [h₀]
    norm_num), -- Block(level=1, content=
  have h1 : (a ^ 2) ^ ((1 : ℝ) / 3) = 4 := by
    rw [ha]
    have h4 : (64 : ℝ) ^ ((1 : ℝ) / 3) = 4 := by
      rw [show (64 : ℝ) = 4 ^ (3 : ℝ) by norm_num]
      rw [←Real.rpow_mul]
      norm_num
      all_goals linarith
    exact h4), 
-- Block(level=1, content=
  have h2 : 16 * (a ^ 2) ^ ((1 : ℝ) / 3) = 64 := by
    rw [h1]
    norm_num), 
-- Snippet(content=
  rw [h2]), 
-- Block(level=1, content=
  have h3 : (64 : ℝ) ^ ((1 : ℝ) / 3) = 4 := by
    rw [show (64 : ℝ) = 4 ^ (3 : ℝ) by norm_num]
    rw [←Real.rpow_mul]
    norm_num
    all_goals linarith), 
-- Snippet(content=
  exact h3)]
  
"""

from __future__ import annotations
from typing import Optional
from prover.utils import remove_lean_comments, statement_starts, analyzable, n_indent
import re
from enum import StrEnum


class BlockState(StrEnum):
    UNVERIFIED = 'unverified' # initial state
    WAIT_SORRY = 'wait_sorry'
    SORRY_FAILED = 'sorry_failed'
    PASSED = 'compilation_passed' # almost legacy
    STTM_FAILED = 'sttm_failed'
    COMPLETED = 'completed' # indicate whole theorem complete or block complete
    NO_RECONSTRUCT = 'no_reconstruct' # only when require_reconstruct is True

class Snippet(object):
    """A snippet corresponding to a full tactic or ends with := by. 
    Always add a newline before adding a new snippet."""
    def __init__(self, content: str = ''):
        self.content = content
        self._proofaug_content = None
    
    @property
    def category(self):
        if statement_starts(self.content):
            return 'statement'
        else:
            return 'normal'
    
    @property
    def proofaug_content(self):
        if self._proofaug_content is None:
            return self.content
        else:
            return self._proofaug_content

    def _receive_snippet(self, snippet: Snippet | str):
        new_content = snippet.content if isinstance(snippet, Snippet) else snippet
        if self.content:
            self.content += "\n" + new_content
        else:
            self.content = new_content

    def __repr__(self):
        return f"-- Snippet(content=\n{self.content})"

class Block(object):
    """each block's parts is a list of snippets and subblocks"""
    def __init__(self, parent: Optional[Block], start_line: Optional[int] = None):
        self.parts = [] # type: list[Block|Snippet]
        self.parent = parent
        self.index = parent.index + f".{len(parent.parts)}" if parent else "0"
        self.level = parent.level + 1 if parent else -1
        self.content_snapshot = None # type: str
        self.state = BlockState.UNVERIFIED # wait_sorry, sorry_failed, verified, sttm_failed
        self._proofaug_parts = None
        self.start_line = start_line
        self.end_line = None
        
    @property
    def content(self):
        return "\n".join([part.content for part in self.parts])
    
    @property
    def proofaug_parts(self):
        if self._proofaug_parts is None:
            return self.parts
        else:
            return self._proofaug_parts

    @property
    def proofaug_content(self):
        return "\n".join([part.proofaug_content for part in self.proofaug_parts])

    @property
    def category(self):
        return 'block'

    @property
    def statement(self):
        return self.parts[0].content.split(':=')[0]

    def _receive_block(self, block: Block):
        self.parts.append(block)

    def _receive_snippet(self, snippet: Snippet | str, append: bool = False):
        if append and self.parts and isinstance(self.parts[-1], Snippet):
            self.parts[-1]._receive_snippet(snippet)
        else:
            self.parts.append(snippet)

    def __repr__(self):
        return f"-- Block(level={self.level}, content=\n{self.content})"




class ProposalStructure(object):
    def __init__(self, proposal: str):
        self.proposal = proposal
        self.root = None
        self._analyze(remove_lean_comments(proposal, normalize=False))

    def _analyze(self, proposal: str):
        lines = proposal.split("\n")
        # determine the blocks by finding 'have' and ':=' and by the indentation
        indent2level = {}   # init block level is -1
        block_stack = [Block(parent=None)] # type: list[Block]
        i = 0   # pointer
        while i < len(lines):
            line = lines[i]
            if line.strip() == '':
                block_stack[-1]._receive_snippet(Snippet(), append=True) # in fact add new line
                i += 1
                continue
            # we assume that the proof never opens a new goal by 'have' when the current same level block is not yet closed
            # determine the level and the current block
            indent = n_indent(line) # indicate the block level.
            if indent not in indent2level:
                indent2level[indent] = len(indent2level)
            level = indent2level[indent]
            while block_stack[-1].level >= level:
                top_block = block_stack.pop()
                top_block.end_line = i # [start: end) is the block content
            parent_block = block_stack[-1]
            if statement_starts(line):  # add a new block
                # we know that block_stack[0] is level -1 and level>=0
                block = Block(parent=parent_block, start_line=i)
                sttm_content = line
                while True: # exit condition is complex. using while True is easier.
                    i += 1
                    # when we find := by, we stop
                    if analyzable(sttm_content) or i >= len(lines):
                        break
                    # sometimes there is no := by, we stop when <= indent occurs.
                    if n_indent(lines[i]) < indent:
                        break
                    elif n_indent(lines[i]) == indent:
                        if not lines[i].strip().startswith('|'):
                            break
                    sttm_content += "\n" + lines[i]
                block._receive_snippet(Snippet(sttm_content))
                parent_block._receive_block(block)
                block_stack.append(block)
            else:
                tactic_content = line
                i += 1
                while i < len(lines):   # when indent > current_indent, we append to this tactic
                    if n_indent(lines[i]) < indent:
                        break
                    elif n_indent(lines[i]) == indent:
                        special_start_indicators = ['|', '<;>', ')']
                        special_end_indicators = ['|', '<;>', '(']
                        if not ( any(lines[i].strip().startswith(indicator) for indicator in special_start_indicators) \
                            or any(lines[i-1].strip().endswith(indicator) for indicator in special_end_indicators)):
                            break
                    tactic_content += "\n" + lines[i]
                    i += 1
                parent_block._receive_snippet(Snippet(tactic_content))
        self.root = block_stack[0]
        while block_stack:
            top_block = block_stack.pop()
            top_block.end_line = len(lines) # start:end, so not minus 1
        
    
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
