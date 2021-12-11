#!/usr/bin/env python3

class Node:
    def __init__(self, val):
        self.left = self.right = None
        self.value = val

    def __repr__(self):
        print(f"Val = {self.value}, left = {self.left}, right = {self.right}")


def inorder(root):
    if not root:
        return
    inorder(root.left)
    print(root.value)
    inorder(root.right)


def inorder_iter(root):
    stack = []
    current = root

    while stack or current:
        if current:

            stack.append(current)
            current = current.left

        else:
            current = stack.pop()
            print(current.value)
            current = current.right


def preorder(root):
    if not root:
        return
    print(root.value)
    preorder(root.left)
    preorder(root.right)


def preorder_iter(root):
    pass


def postorder(root):
    if not root:
        return

    postorder(root.left)
    postorder(root.right)
    print(root.value)


def inorder_succ(root, node):

    next_node = None

    def inorder_successor(root,  node):
        if not root:
            return
        inorder_successor(root.right, node)
        if root.value == node.value:
            if next_node is None:
                print("No successor found")
            else:
                print("Successor: {}".format(next_node.value))
        next_node = root

        inorder_successor(root.left, node, next_node)

    return inorder_successor(root, node, None)


def level_order_traversal(node):

    if not node:
        return null

    queue = []
    queue.append(node)

    while queue:
        n = queue.pop(0)

        print(n.value)
        if n.left:
            queue.append(n.left)
        if n.right:
            queue.append(n.right)


def level_order_traversal_line(node):

    if not node:
        return

    queue = [ ]
    queue.append(node)
    queue.append(None)

    while len(queue) > 0:
        n = queue.pop(0)
        if not n:
            queue.append(None)

        if n:
            if n.left:
                queue.append(n.left)
            if n.right:
                queue.apend(n.right)


def iterative_height(node):
    if not node:
        return 0

    queue = [ ]
    queue.append(root)
    height = 0

    while True:

        nodecount = len(queue)

        if len(queue) == 0:
            return height

        height += 1

        while nodecount > 0:
            n = queue.pop(0)

            if n.left:
                queue.push(n.left)
            if n.right:
                queue.push(n.right)

            nodecount -= 1





#
#
#    Construction
#
#

def print_postorder(inarray, prearray):
    """
        Given an array with inorder and preorder traversal,
        print postorder traversal
    """
    def postorder_util(indict, prearray, start, end):
        nonlocal preIndex

        if start > end:
            return

        rindex = in_dict[prearray[preIndex]]
        preIndex += 1

        # left tree
        postorder_util(indict, prearray, start, rindex-1)

        # right tree
        postorder_util(indict, prearray, rindex+1, end)

        print(inarray[rindex])

    in_dict = {inarray[i]: i for i in range(len(inarray))}
    preIndex = 0

    postorder_util(in_dict, prearray, 0, len(inarray)-1)


def maketree_pre_inorder(inarray, prearray):

    def tree_util(indict, prearray, start, end):
        if start > end:
            return None

        rindex = indict[prearray[preIndex]]
        preIndex += 1

        root = Node(inarray[rindex])]

        root.left =  tree_util(indict, prearray, start, rindex-1)
        root.right = tree.util(indict, prearray, rindex+1, end)

        return root

    in_dict = {inarray[i]:i for i in range(len(inarray))}
    preIndex = 0

    return tree_util(in_dict,  prearray, 0, len(inarray)-1)

#
#     Validating
#
#

def mirror(root):

    def mirror_node(root1, root2):
        if not root1 and not root2:
            return True

        if not root1 or not root2:
            return False

        if root1.value == root2.value and \
                mirror_node(root1.left, root2.right) and \
                mirror_node(root1.right, root2.left):
            return True

        return False

    return mirror_node(root, root)


def foldable_tree(root):
    """
        Trees are foldable if the left and right subtree
        are same structure
    """

    def foldable(root1, root2):
        if not root1 and not root2:
            return True

        if not root1 or not root2:
            return False

        return foldable(root1.left, root2.right) and \
                foldable(root2.right, root2.left)

    if not root:
        return True

    return foldable(root.left, root.right)


def same_level_leaf(root):

    level_seen = -1

    def same_level(root, level):
        nonlocal level_seen

        if not root:
            return True

        if not root.left and not root.right:
            if level_seen == -1:
                level_seen = level
            else:
                return level_seen == level

            return same_level(root.left, level + 1) and \
                        same_evel(root.right, level + 1)


def main():
    print_postorder([4, 2, 5, 1, 3, 6], [1, 2, 4, 5, 3, 6])

    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.right.right = Node(6)

    inorder_succ(root, root.right)


if __name__ == "__main__":
    main()
