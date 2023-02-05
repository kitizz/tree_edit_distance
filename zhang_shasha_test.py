from .tree import TreeNode, tree_from_dict
from .zhang_shasha import CostFunctions, zhang_shasha


def test_single_equal():
    tree = TreeNode("root", ())
    assert zhang_shasha(tree, tree) == 0


def test_single_not_equal():
    a_tree = TreeNode("a", ())
    b_tree = TreeNode("b", ())
    # Default relabel distance is 1
    assert zhang_shasha(a_tree, b_tree) == 1
    assert zhang_shasha(b_tree, a_tree) == 1


def test_medium_equal():
    tree = tree_from_dict({"root": {"a": {"f": {}}, "b": {"c": {"d": {}, "e": {}}}}})
    assert zhang_shasha(tree, tree) == 0


def test_medium_relabels_only():
    #       a                  z
    #    b     c     =>     b     c
    #  d e f     g        d y f     x
    a_tree = tree_from_dict({"a": {"b": {"d": {}, "e": {}, "f": {}}, "c": {"g": {}}}})
    z_tree = tree_from_dict({"z": {"b": {"d": {}, "y": {}, "f": {}}, "c": {"x": {}}}})

    assert zhang_shasha(a_tree, z_tree) == 3
    assert zhang_shasha(z_tree, a_tree) == 3


def test_medium_delete_all():
    #       a                  a
    #    b     c     =>
    #  d e f     g
    a_tree = tree_from_dict({"a": {"b": {"d": {}, "e": {}, "f": {}}, "c": {"g": {}}}})
    z_tree = TreeNode("a", ())

    # Default deletes and inserts both cost 1.
    assert zhang_shasha(a_tree, z_tree) == 6
    assert zhang_shasha(z_tree, a_tree) == 6


def test_medium_mixed():
    #       a                  z
    #    b     c     =>     b     g    x
    #  d e f     g        y e
    a_tree = tree_from_dict({"a": {"b": {"d": {}, "e": {}, "f": {}}, "c": {"g": {}}}})
    z_tree = tree_from_dict({"z": {"b": {"y": {}, "e": {}}, "g": {}, "x": {}}})

    # Default deletes and inserts both cost 1.
    assert zhang_shasha(a_tree, z_tree) == 5
    assert zhang_shasha(z_tree, a_tree) == 5


def test_medium_assymetric_costs():
    #       a                  a
    #    b     c     =>
    #  d e f     g
    a_tree = tree_from_dict({"a": {"b": {"d": {}, "e": {}, "f": {}}, "c": {"g": {}}}})
    single_tree = TreeNode("a", ())

    def insert(node):
        return 2

    def delete(node):
        return 1

    costs_funcs = CostFunctions(insert=insert, delete=delete)

    # Default deletes and inserts both cost 1.
    assert zhang_shasha(a_tree, single_tree, cost_funcs=costs_funcs) == 6
    assert zhang_shasha(single_tree, a_tree, cost_funcs=costs_funcs) == 12
