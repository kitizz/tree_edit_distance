"""Original paper: The pq-Gram Distance between Ordered Labeled Trees
https://tinyurl.com/pq-grams-paper

"""

from collections import deque
from collections.abc import Sequence
from typing import TypeAlias
from tree import TreeNode

PQGram: TypeAlias = tuple[str, ...]

# Label value used for "dummy nodes". This implementation means that input
# trees cannot have empty label strings.
DUMMY = ""


def shift_inplace(deq: deque[str], value: str) -> str:
    deq.append(value)
    return deq.popleft()


def count_bag_intersection(a: Sequence[PQGram], b: Sequence[PQGram]) -> int:
    """
    Requires:
    - a and b have sortable elements
    - a and b are both sorted
    """
    num_intersections = 0
    index_a = 0
    index_b = 0
    while index_a < len(a) and index_b < len(b):
        a_val = a[index_a]
        b_val = b[index_b]

        if a_val == b_val:
            num_intersections += 1
            index_a += 1
            index_b += 1
            continue

        if a_val < b_val:
            index_a += 1
            continue

        if b_val < a_val:
            index_b += 1
            continue

    return num_intersections


class PQGramIndex:
    #: The depth/height of the generated PQ-Grams
    p: int

    #: The width of the generated PQ-grams
    q: int

    # TODO: Doc
    pq_grams: list[PQGram]

    def __init__(self, root: TreeNode, p: int, q: int):
        self.p = p
        self.q = q
        self.pq_grams = []
        stem: deque[str] = deque([DUMMY] * p)

        self._build_index(root, stem)
        self.pq_grams.sort()

    def _build_index(self, node: TreeNode, stem: deque):
        """Recursively build the actual PQ-Grams index of a tree.

        The index-building part of the PQ-Grams technique is truly the core of the
        published technique. The rest is simply comparing sets of tuples, and proofs
        for various properties claimed about the technique.

        A note on that, be wary of the claim that it is a lower bound of the fanout
        weighted tree edit distance, since this is only shown for p=1. See the
        associated pq_grams Jupyter notebook for experimental studies for p>1.
        """
        # Corresponds to Algorithm 8.2
        #  - "node" is "a" in the paper
        #  - self.pg_grams is an in-place edited equivalent to "I"
        #
        # Drag a p-deep x q-wide window through the tree. The resulting "shape" is
        # an up-side-down T where the stem of the ⊥ captures "p" ancestor nodes,
        # and the base of the ⊥ captures the "q" children of the deepest node of
        # the stem.

        # Algorithm 8.2: line 5
        base: deque[str] = deque([DUMMY] * self.q)

        if node.label == DUMMY:
            raise TypeError("This implementation of PQ-Grams cannot handle empty node labels.")

        # Algorithm 8.2: line 6
        stem = stem.copy()
        shift_inplace(stem, node.label)

        stem_gram = tuple(stem)  # Save re-tupling this multiple times.

        if len(node.children) == 0:
            # node is a leaf
            self.pq_grams.append(stem_gram + tuple(base))
            return

        for child in node.children:
            shift_inplace(base, child.label)
            self.pq_grams.append(stem_gram + tuple(base))
            self._build_index(child, stem)

        for k in range(self.q - 1):
            shift_inplace(base, DUMMY)
            self.pq_grams.append(stem_gram + tuple(base))


def pq_grams(
    a_tree_root: TreeNode,
    b_tree_root: TreeNode,
    *,
    p: int = 2,
    q: int = 3,
    normalized=False,
    halved=True,
) -> float:
    """

    Ensures:
    - If halved is true, then the *non-normalized* distance returned will be halved [1]

    [1] Why? See Section 7.3 which claims that half the pq-gram distance is a lower bound
        on fanout weighted tree edit distance.
    """
    a_index = PQGramIndex(a_tree_root, p=p, q=q)
    b_index = PQGramIndex(b_tree_root, p=p, q=q)

    # Bag union size: |I1 ⊎ I2|
    union_size = len(a_index.pq_grams) + len(b_index.pq_grams)
    intersection_size = count_bag_intersection(a_index.pq_grams, b_index.pq_grams)
    pq_dist = union_size - 2 * intersection_size

    if normalized:
        pq_dist = pq_dist / (union_size + intersection_size)
    elif halved:
        pq_dist = pq_dist // 2

    return pq_dist
