from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generator, Self, TypeAlias, TypeVar

__all__ = ["pretty_format"]

Node = TypeVar("Node")

GetLabel: TypeAlias = Callable[[Node], str]
GetChildren: TypeAlias = Callable[[Node], Sequence[Node]]


def _default_get_label(node) -> str:
    return getattr(node, "label")


def _default_get_children(node: Node) -> Sequence[Node]:
    return getattr(node, "children")


def pretty_format(
    tree: Node,
    get_label: GetLabel[Node] = _default_get_label,
    get_children: GetChildren[Node] = _default_get_children,
) -> str:
    """Format the tree into a pretty string

    An example:
         ┌d──t─┬s
         │  ┌v └q
      ┌b─┤  ├z
      │  ├e─┼y
    a─┤  ├f ├x
      │  └l └w
      └c─┬g
         └h
    """
    pretty_tree = _wrap_tree(tree, get_label, get_children)

    y_to_nodes: defaultdict[int, list[PrintNode]] = defaultdict(list)
    column_to_max_width: defaultdict[int, int] = defaultdict(int)
    for node in _preorder_traversal(pretty_tree):
        y_to_nodes[node.y].append(node)
        column_to_max_width[node.depth] = max(column_to_max_width[node.depth], len(node.label))

    lines = []
    active_parent_y_by_column: dict[int, int | None] = {}
    for y in range(min(y_to_nodes), max(y_to_nodes) + 1):
        nodes = y_to_nodes[y]
        x_to_node: dict[int, PrintNode] = {}
        for node in nodes:
            assert x_to_node.setdefault(node.depth, node) is node, "Duplicate x values"

        line_labels = []
        for column in range(max(x_to_node) + 1):
            column_width = column_to_max_width[column]

            if column in x_to_node:
                node = x_to_node[column]
                pre, active_parent_y = _label_prefix(node, y)
                active_parent_y_by_column[column] = active_parent_y

                label = node.label

                post = BoxChar()
                if node.children:
                    post.add_east().add_west()

            else:
                pre = BoxChar()
                active_parent_y = active_parent_y_by_column.get(column)
                if active_parent_y is not None:
                    pre.add_north().add_south()
                    if active_parent_y == y:
                        pre.add_west()
                label = ""
                post = " "

            line_labels.append(f"{pre}{label:{post}<{column_width + 1}}")
        lines.append("".join(line_labels))

    return "\n".join(lines)


def _label_prefix(node: "PrintNode", current_y: int) -> tuple[str, int | None]:
    """Return the prefix for the label of the node and the active parent's y position

    The active parent's y is used to check if vertical branches need to be printed
    between siblings of node's column.

    Ensures:
    - active_parent_y is None if node is the last child of its parent or if node is root.
    """
    if not node.parent:
        # Root node
        return " ", None

    pre_box_char = BoxChar()
    pre_box_char.add_east()

    if current_y == node.parent.y:
        # Connect this node with its parent.
        pre_box_char.add_west()

    siblings = node.parent.children
    if node is not siblings[0]:
        # Connect with siblings above.
        pre_box_char.add_north()

    if node is not siblings[-1]:
        # Connect with siblings below.
        pre_box_char.add_south()

    pre = str(pre_box_char)
    active_parent_y = node.parent.y if node is not siblings[-1] else None
    return pre, active_parent_y


@dataclass
class PrintNode:
    # node: TreeNode
    label: str
    children: tuple["PrintNode", ...]

    #: Columner index of the node.
    depth: int

    #: Absolute y position of the node.
    y: int = 0

    #: Relative y displacement from parent.
    delta_y: int = 0

    parent: "PrintNode | None" = None

    #: The bounds of the ancestors' relative y displacements.
    descendent_y_range: list[tuple[int, int]] = field(default_factory=list)


def _preorder_traversal(node: PrintNode) -> Generator[PrintNode, None, None]:
    yield node
    for child in node.children:
        yield from _preorder_traversal(child)


def _wrap_tree(
    node: Node, get_label: GetLabel[Node], get_children: GetChildren[Node], depth: int = 0
) -> PrintNode:
    """Wrap the tree nodes with PrintNode objects and calculate their positions"""
    children = get_children(node)
    children = tuple(_wrap_tree(child, get_label, get_children, depth + 1) for child in children)
    pnode = PrintNode(get_label(node), children, depth=depth)

    for child in children:
        child.parent = pnode

    _adjust_children_delta_y(pnode)

    if depth == 0:
        # At the root, fill in the y positions by propagating the delta_y values.
        _fill_y_positions(pnode, y=0)

    return pnode


def _target_delta_y(a: PrintNode, b: PrintNode, gap_size: int) -> int:
    """Calculate the target delta_y for b to avoid collisions between a and b's descendents"""
    min_gap = 0
    for a_y_range, b_y_range in zip(a.descendent_y_range, b.descendent_y_range):
        min_gap = min(min_gap, b_y_range[0] - a_y_range[1])

    # 1 offset because the gap is zero when the max-min are one line apart.
    return -min_gap + 1 + gap_size


def _adjust_children_delta_y(node: PrintNode, gap_size: int = 0):
    """Adjust the delta_y of node's children to avoid collisions between their descendents

    Assumes the descendents of each child have already been adjusted.
    """
    if not node.children:
        return

    # Close the gap between the descendents
    for curr_child, next_child in zip(node.children, node.children[1:]):
        target_dy = _target_delta_y(curr_child, next_child, gap_size)
        next_child.delta_y = curr_child.delta_y + target_dy

    # Center the descendents around node (ie. the middle delta_y bounds of children should be zero)
    min_delta_y = min(child.delta_y for child in node.children)
    max_delta_y = max(child.delta_y for child in node.children)
    offset = (min_delta_y + max_delta_y) // 2
    for child in node.children:
        child.delta_y -= offset

    node.descendent_y_range = [(min_delta_y - offset, max_delta_y - offset)]
    for child in node.children:
        for i, (min_y, max_y) in enumerate(child.descendent_y_range):
            proposed_min_y = child.delta_y + min_y
            proposed_max_y = child.delta_y + max_y

            if i + 1 >= len(node.descendent_y_range):
                node.descendent_y_range.append((proposed_min_y, proposed_max_y))
                continue

            curr_min_y, curr_max_y = node.descendent_y_range[i + 1]
            node.descendent_y_range[i + 1] = (
                min(curr_min_y, proposed_min_y),
                max(curr_max_y, proposed_max_y),
            )


def _fill_y_positions(node: PrintNode, y: int = 0):
    """Recursively fill in the y positions of each node from the relative delta_y values"""
    node.y = y
    for child in node.children:
        _fill_y_positions(child, y=node.y + child.delta_y)


class BoxChar:
    """Build a box char programmatically based on which sides it connects to.

    Example:
        str(BoxChar().add_north().add_east().add_south().add_west()) == "┼"
    """

    # Box-drawing characters labeled by which sides they connect to, ordered alphabetically.
    BOX_CHARS = {
        "ENSW": "┼",
        "ENS": "├",
        "ENW": "┴",
        "NSW": "┤",
        "ESW": "┬",
        "EN": "└",
        "NS": "│",
        "NW": "┘",
        "ES": "┌",
        "EW": "─",
        "SW": "┐",
        "": " ",
    }

    def __init__(self):
        self.sides = set()

    def add_north(self) -> Self:
        self.sides.add("N")
        return self

    def add_east(self) -> Self:
        self.sides.add("E")
        return self

    def add_south(self) -> Self:
        self.sides.add("S")
        return self

    def add_west(self) -> Self:
        self.sides.add("W")
        return self

    def __str__(self) -> str:
        key = "".join(sorted(self.sides))
        return self.BOX_CHARS[key]
