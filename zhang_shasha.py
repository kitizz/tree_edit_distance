"""This implementation was greatly aided by the blog post at
https://www.baeldung.com/cs/tree-edit-distance
And so many thanks to the author, Milos Simic.

The main problem I faced with that article was that it missed or glossed over
the critical point that the cost to relabel needs to be depependent on whether
the relabeled node on each side is already equal. When running through the
final proposed algorithm in my head for two equal trees, the cost wasn't zero
which confused me greatly.

The original paper by Zhang and Shasha: https://tinyurl.com/zhang-shasha-ted
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import dataclasses
from functools import cache

from tree import TreeNode


@dataclass
class SubForest:
    """A subforest is a set of subtrees of a source tree.

    Here, the SubForest is a particular set of subtrees obtained by selecting
    any slice (range) of nodes ordered by there post-order traversal index.
    This is best understood by visualization:

           a                             7
        b     c      post-order       3     6
      d e f   g  h       =>         0 1 2   4 5

    Take the slice, [0:6] (using python's notation, stop index is not inclusive):
         .
      3     .
    0 1 2   4 5

    Notice this leaves us with 3 subtrees, with (3) at the root of the first,
    and the other two being single nodes each, (4) and (5).

    Also notice that for any slice, the last included index will always by the
    root of a sub-tree. Further, it will always by the right-most root.

    To calculate the Tree Edit Distance (TED) with Zhang + Shasha's method, we
    wish to be able to identify the range of indexes that reference that whole
    right subtree (not just the node). This is achieved by generating
    subtree_start_index during pre-processing. For any given node index, it
    gives us the index of its left-most descendant. For post-order indexes, the
    nodes in the slice [subtree_start_index[i], i] give us the entire subtree
    for the node at index i.
    """

    #: The root node of the source tree for this SubForest.
    source_tree: TreeNode

    #: All nodes from the source tree in post-order traversal order.
    post_ordered_nodes: list[TreeNode] = field(default_factory=list)

    #: For post-order indexes, the nodes in the slice [subtree_start_index[i], i]
    #: give us the entire subtree for the node at index i.
    subtree_start_index: list[int] = field(default_factory=list)

    #: Reference the post-ordered index range of the source tree's nodes with
    #: start and stop indexes.
    start_index: int = 0
    #: Non-inclusive.
    stop_index: int = 0

    @classmethod
    def from_tree(cls, root: TreeNode):
        """Preprocess a Tree to initialize a top-level SubForest."""
        subforest = cls(root)
        subforest._process_tree(root)
        subforest.stop_index = len(subforest.post_ordered_nodes)
        return subforest

    def is_empty(self):
        """Return if this subforest contains no trees/nodes. This is the ∅ case in the paper."""
        return self.start_index == self.stop_index

    def last_node(self) -> TreeNode:
        """Retrieve the last node to check its cost of insertion, deletion, or relabeling."""
        assert not self.is_empty()
        return self.post_ordered_nodes[self.stop_index - 1]

    def last_node_dropped(self) -> "SubForest":
        """Return a copy of this SubForest with the last node dropped.

        Usually to check the rest of the SubForest if this node were inserted or deleted.
        """
        assert not self.is_empty()
        return dataclasses.replace(self, stop_index=self.stop_index - 1)

    def last_tree_dropped(self) -> "SubForest":
        """Return a copy of this SubForest with the last (right-most) tree dropped.

        Visual:
              .                                         .
           3     6       last_tree_dropped() ->      3     .
         0 1 2   4 5                               0 1 2   . .

        This is used when the last node is being considered for relabeling
        where the compared SubForests are split into last_tree_dropped() and
        last_subforest(). Not that the last node is dropped from this split.
        """
        assert not self.is_empty()
        return dataclasses.replace(
            self,
            start_index=self.start_index,
            stop_index=self.subtree_start_index[self.stop_index - 1],
        )

    def last_subforest(self) -> "SubForest":
        """Return a copy of the last tree in this SubForest, but with the last node dropped.

        Visual:
              .                                         .
           3     6       last_subforest() ->         .     .
         0 1 2   4 5                               . . .   4 5

        This is used when the last node is being considered for relabeling
        where the compared SubForests are split into last_tree_dropped() and
        last_subforest(). Not that the last node is dropped from this split.
        """
        assert not self.is_empty()
        return dataclasses.replace(
            self,
            start_index=self.subtree_start_index[self.stop_index - 1],
            stop_index=self.stop_index - 1,
        )

    def __hash__(self):
        """Used for the cache/memoization"""
        return hash((id(self.source_tree), self.start_index, self.stop_index))

    def __eq__(self, other):
        """Used for the cache/memoization"""
        if not isinstance(other, SubForest):
            return NotImplemented
        return (
            id(self.source_tree) == id(other.source_tree)
            and self.start_index == other.start_index
            and self.stop_index == other.stop_index
        )

    def _process_tree(self, node: TreeNode):
        # In a post-order iteration of the tree, the next registered node will
        # be the left-most leaf of this node's subtree. And so the current
        # length of the post_ordered_nodes will be the index of that node.
        left_most_subtree_index = len(self.post_ordered_nodes)
        for child in node.children:
            self._process_tree(child)

        self.post_ordered_nodes.append(node)
        self.subtree_start_index.append(left_most_subtree_index)


@dataclass
class CostFunctions:
    """Override these to customize the cost functions."""

    delete: Callable[[TreeNode], float] = lambda node: 1
    insert: Callable[[TreeNode], float] = lambda node: 1
    relabel: Callable[[TreeNode, TreeNode], float] = lambda a, b: int(a.label != b.label)


def zhang_shasha(
    a_tree_root: TreeNode,
    b_tree_root: TreeNode,
    cost_funcs: CostFunctions = CostFunctions(),
) -> float:
    a_forest = SubForest.from_tree(a_tree_root)
    b_forest = SubForest.from_tree(b_tree_root)

    @cache
    def forestdist(a_forest: SubForest, b_forest: SubForest) -> float:
        """This implements the initial recursive definitions laid out in the paper,
        using the functools.cache to achieve the same optimizations as the formalized
        dynamic program ultimately proposed by Zhang + Shasha.

        Where appropriate, the code below makes references to the original paper, so
        one can see/learn where the logic was derived from. Even though the code here
        is (hopefully) written in style more conducive to a healthy, maintainable
        software library.

        Notice that each time the algorithm enters this function, its focus is on the
        last post-ordered node in each SubForest. Some helpful observations:
         - At the start, these nodes are the root of each tree.
         - Dropping of this last node is achieved by truncating the forest by a
           single index.
         - A delete operation looks like dropping the node from a_forest.
         - An insert operation looks like dropping the node from b_forest.
         - A relabel operation looks like splitting the children of the last node
           from the non-children in both forests.
        """
        if a_forest.is_empty() and b_forest.is_empty():
            # The most trivial of cases. Two empty forests are identical, so distance is zero.
            # Corresponds to Lemma 3(i): forestdist(∅, ∅) = 0
            return 0.0

        if b_forest.is_empty():
            # With a non-empty a_forest, all there is to do is recursively delete all
            # nodes in a_forest. If you knew that cost_delete is constant, then you
            # could optimize this to:
            #   return cost_delete * size(a_forest)

            # Corresponds to Lemma 3(ii),
            # forestdist(T1[l(i1)..i], ∅) = forestdist(T1[l(i1)..i-1]) + γ(T1[i] -> Λ)
            cost_delete = cost_funcs.delete(a_forest.last_node())
            return cost_delete + forestdist(a_forest.last_node_dropped(), b_forest)

        if a_forest.is_empty():
            # With a non-empty b_forest, all there is to do is recersively insert all
            # nodes to b_forest. If you knew that cost_insert is constant, then you
            # could optimize this to:
            #   return cost_insert * size(b_forest)

            # Corresponds to Lemma 3(iii),
            # forestdist(∅, T2[l(j1)..j]) = forestdist(∅, T2[l(j1)..j-1]) + γ(Λ -> T2[j])
            cost_insert = cost_funcs.insert(b_forest.last_node())
            return cost_insert + forestdist(a_forest, b_forest.last_node_dropped())

        # At this point, we have two forests flush with tree(s). We let the algorithm
        # explore the cost of

        # Delete a's last root.
        # Paper (p1251): forestdist(T1[l(i1)..i-1], T2[l(j1)..j]) + γ(T1[i] -> Λ)
        cost_delete = cost_funcs.delete(a_forest.last_node())
        dist_delete = cost_delete + forestdist(a_forest.last_node_dropped(), b_forest)

        # Insert b's last root:
        # Paper (p1251): forestdist(T1[l(i1)..i], T2[l(j1)..j-1]) + γ(Λ -> T2[j])
        cost_insert = cost_funcs.insert(b_forest.last_node())
        dist_insert = cost_insert + forestdist(a_forest, b_forest.last_node_dropped())

        # Relabel a's last root to match b's last root.
        # The forests are split on either side
        # Paper (p1251): forestdist(T1[l(i1)..l(i)-1], T2[l(j1)..l(j)-1])
        #              + forestdist(T1[l(i)..i-1], T2[l(j)..j-1])
        #              + γ(T1[i] -> T2[j])
        cost_relabel = cost_funcs.relabel(a_forest.last_node(), b_forest.last_node())
        dist_relabel = (
            cost_relabel
            + forestdist(a_forest.last_tree_dropped(), b_forest.last_tree_dropped())
            + forestdist(a_forest.last_subforest(), b_forest.last_subforest())
        )

        return min(dist_delete, dist_insert, dist_relabel)

    return forestdist(a_forest, b_forest)
