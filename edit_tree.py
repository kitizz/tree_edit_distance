import random
from tree import TreeNode, preorder_traversal


def with_node_deleted(root: TreeNode, delete: TreeNode) -> TreeNode:
    """Return a copy of the tree, root, with the given node deleted."""
    children = []
    for child in root.children:
        if child is not delete:
            children.append(with_node_deleted(child, delete))
            continue

        # Shift the grandkids up as children of the new root.
        for grandchild in child.children:
            children.append(grandchild)

    return TreeNode(root.label, tuple(children), root.depth)


def with_node_inserted(root: TreeNode, insert_label: str, parent: TreeNode, index: int) -> TreeNode:
    """Return a copy of the tree, root, with the given node inserted as a child of parent."""
    if root is parent:
        # Create a new tuple of children, with the new node inserted at the given index.
        new_node = TreeNode(insert_label, (), root.depth + 1)
        children = root.children[:index] + (new_node,) + root.children[index:]
        return TreeNode(root.label, children, root.depth)

    children = tuple(
        with_node_inserted(child, insert_label, parent, index) for child in root.children
    )
    return TreeNode(root.label, tuple(children), root.depth)


def with_node_relabeled(root: TreeNode, relabel: TreeNode, label: str) -> TreeNode:
    """Return a copy of the tree, root, with the given node relabeled."""
    if root is relabel:
        return TreeNode(label, root.children, root.depth)

    children = tuple(with_node_relabeled(child, relabel, label) for child in root.children)
    return TreeNode(root.label, tuple(children), root.depth)


def with_random_edit(root: TreeNode) -> tuple[TreeNode, str]:
    """Return a copy of the tree, root, with a random edit applied."""
    if not root.children:
        raise RuntimeError("Tree must have more than just the root node.")

    # Choose a random edit to apply.
    edit_kind = random.choice(("delete", "insert", "relabel"))

    if edit_kind == "delete":
        # Choose a random node to delete (exclude the root node).
        nodes_no_root = list(preorder_traversal(root))[1:]
        node = random.choice(nodes_no_root)
        return with_node_deleted(root, node), f"Deleted {node.label}"

    if edit_kind == "insert":
        # Choose a random parent node to insert the node into.
        parent = random.choice(list(preorder_traversal(root)))

        # Choose a random index to insert the node at.
        index = random.randint(0, len(parent.children))

        new_label = chr(ord("a") + random.randint(0, 25))

        msg = f"Inserted {new_label} as child of {parent.label} at index {index}"
        return with_node_inserted(root, new_label, parent, index), msg

    if edit_kind == "relabel":
        # Choose a random node to edit.
        node = random.choice(list(preorder_traversal(root)))

        # Choose a random label to relabel the node with.
        label = chr(ord("a") + random.randint(0, 25)) * 2

        msg = f"Relabeled {node.label} as {label}"
        return with_node_relabeled(root, node, label), msg

    raise RuntimeError("Unexpected edit function")
