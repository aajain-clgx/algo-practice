#!/usr/bin/env python3

class Node:
    def __init__(self, val):
        this. val = val
        this.left = None
        this.right = None


def search(node, val):
    if node is None:
        return False
    if node.val == val:
        return True

    elif val < node.val:
        return search(node.left, val)

    return search(node.right, val)


def search_iter(node, val):

    current = node
    found = False

    while current:
        if current.val == val:
            found = True
            break
        elif val < current.val:
            current = current.left
        else:
            current = current.right

    return found


def insert(root, value):
    if not root:
        return Node(value)

    if value < root.val:
        root.left = insert(root.left, value)
    else:
        root.right = insert(root.right, value)

    return root


def insert_iter(root, value):
    """

    """
    current = root
    while current:
        if current.val == value:
            return
        if value < current.left:
            current = current.left
        elif value > current.right:
            current = current.right

   current = Node(val)




def main():
    pass


if __name__ == "__main__":
    main()
