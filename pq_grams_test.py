from tree import TreeNode, tree_from_dict
from pq_grams import pq_grams
from zhang_shasha import zhang_shasha, CostFunctions


def fwted(a_tree: TreeNode, b_tree: TreeNode, *, q: int) -> float:
    """Fanout weight tree edit distance.

    Ensures:
    - Alignment (edit) costs (Definition 5.4):
        Insert: f_v + c
        Delete: f_v' + c
        Relabel: (f_v + f_v')/2 + c
      where f_v and f_v' are the fanouts of a_tree's and b_tree's nodes, respectively;
      c is the constant cost offset used (see below).
    - The minimum constant cost offset is chosen (Theorem 7.4, Section 7.3):
        c = max(2*q - 1, 2)
      such that the pq-grams distance is a lower bound on this distance for p=1
    """
    cost_offset = max(2 * q - 1, 2)

    def insert_or_delete(node: TreeNode):
        return len(node.children) + cost_offset

    def relabel(a: TreeNode, b: TreeNode):
        return (len(a.children) + len(b.children)) / 2 + cost_offset

    cost_funcs = CostFunctions(insert=insert_or_delete, delete=insert_or_delete, relabel=relabel)

    return zhang_shasha(a_tree, b_tree, cost_funcs=cost_funcs)


def test_single_equal():
    tree = TreeNode("root", ())
    assert pq_grams(tree, tree) == 0


# Many of the following tests simply ensure that the PQ-Grams distance is in fact
# a lower bound on the fanout weighted tree edit distance (TED).


def test_single_not_equal():
    a_tree = TreeNode("a", ())
    b_tree = TreeNode("b", ())
    # The fanout weighted TED is 1 here.
    for q in range(2, 5):
        # Symmetric
        assert pq_grams(a_tree, b_tree, p=1, q=5) == pq_grams(b_tree, a_tree, p=1, q=5)
        # Lower bound
        assert pq_grams(a_tree, b_tree, p=1, q=5) <= fwted(a_tree, b_tree, q=5)


def test_simple_rename():
    a_tree = tree_from_dict({"a": {"b": {}, "c": {}}})
    z_tree = tree_from_dict({"z": {"b": {}, "c": {}}})
    for q in range(2, 5):
        # Symmetric
        assert pq_grams(a_tree, z_tree, p=1, q=q) == pq_grams(z_tree, a_tree, p=1, q=q)
        # Lower bound
        assert pq_grams(a_tree, z_tree, p=1, q=q) <= fwted(a_tree, z_tree, q=q)


def test_medium_equal():
    tree = tree_from_dict({"root": {"a": {"f": {}}, "b": {"c": {"d": {}, "e": {}}}}})
    assert pq_grams(tree, tree) == 0
