from collections import defaultdict
from dataclasses import dataclass, field
import random
from typing import Generator
import numpy as np


@dataclass(frozen=True)
class TreeNode:
    label: str
    children: tuple["TreeNode", ...]
    depth: int = 0

    def pretty_str(self):
        """Return a representation of the tree ready for pretty printing"""
        setattr(self, "ha", 5)


def preorder_traversal(tree: TreeNode) -> Generator[TreeNode, None, None]:
    yield tree
    for child in tree.children:
        yield from preorder_traversal(child)


def tree_from_dict(root: dict[str, dict]) -> TreeNode:
    r"""Construct a tree from a nested dict of strings.
    
    Requires:
     - Exactly one key at top to represent the root.
     - Empty dicts represent leaf nodes.
    
    For example, the tree,
      A
     / \
    B   C
       / \
      D   E
      
    can be constructed from the dict:
    {
        "A": {
            "B": {},
            "C": {
                "D": {},
                "E": {},
            }
        }
    }
    """
    if len(root) == 0:
        raise TypeError("Empty trees not supported")
    if len(root) > 1:
        raise TypeError("Dict, root, must have exactly one key to represent the root node.")

    key, children = next(iter(root.items()))
    return _node_from_sub_dict(key, children)


def random_tree(
    *,
    min_height=1,
    max_depth: int,
    fanouts: tuple[int, ...],
    labels: tuple[str, ...] | None = None,
) -> TreeNode:
    for i in range(1000):
        root = _random_tree(max_depth=max_depth, fanouts=fanouts, labels=labels)
        height = max(node.depth for node in preorder_traversal(root)) + 1
        if height >= min_height:
            return root

    raise RuntimeError("Unable to generate a tree with the min_height; check paramaters.")


def _random_tree(
    *,
    depth: int = 0,
    max_depth: int,
    fanouts: tuple[int, ...],
    labels: tuple[str, ...] | None = None,
) -> TreeNode:
    if labels:
        label = random.choice(labels)
    else:
        label = chr(ord("a") + random.randint(0, 25))

    if depth == max_depth:
        return TreeNode(label, (), depth)

    num_children = random.choice(fanouts)
    children = tuple(
        _random_tree(depth=depth + 1, max_depth=max_depth, fanouts=fanouts, labels=labels)
        for _ in range(num_children)
    )

    return TreeNode(label, children, depth)


def _node_from_sub_dict(name: str, children_dict: dict | None, depth: int = 0) -> TreeNode:
    if children_dict is None:
        return TreeNode(name, (), depth)

    children_nodes = []
    for node_name, node_children in children_dict.items():
        children_nodes.append(_node_from_sub_dict(node_name, node_children, depth + 1))
    return TreeNode(name, tuple(children_nodes), depth)
