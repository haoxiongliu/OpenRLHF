"""Utils that used by agent_func_proofaug.py
Set independent since that module will be dynamically loaded in training.
"""


def remove_indent(content: str) -> str:
    l_indent = len(content.split('\n')[0])
    return "\n".join([line[l_indent:] for line in content.split("\n")])

def add_indent(content: str, indent: int) -> str:
    return "\n".join([" "*indent + line for line in content.split("\n")])