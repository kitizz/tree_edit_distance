from dataclasses import dataclass


@dataclass
class TreeNode:
    label: str
    children: tuple["TreeNode", ...]


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


def _node_from_sub_dict(name: str, children_dict: dict | None) -> TreeNode:
    if children_dict is None:
        return TreeNode(name, ())

    children_nodes = []
    for node_name, node_children in children_dict.items():
        children_nodes.append(_node_from_sub_dict(node_name, node_children))
    return TreeNode(name, tuple(children_nodes))
