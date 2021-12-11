#!/usr/bin/env python3

class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.value = val


def inorder(node):
    if not node:
        return
    inorder(node.left)
    print(node.value)
    inorder(node.right)


def inorder_iter(node):
    stack = []
    current = node

    while current or len(stack) > 0:

        if current:
            stack.append(current)
            current = current.left
        else:
            if len(stack) > 0:
                current = stack.pop()
                print(current.value)
            current = current.right


def preorder_iter(node):
    stack = []
    current = node

    stack.append(current)
    while len(stack) > 0:
        current = stack.pop()
        print(current.value)

        if current.right:
            stack.append(current.right)
        if current.left:
            stack.append(current.left)


def level_order(node):
    queue = []
    current = node
    queue.append(current)

    while queue:
        current = queue.pop(0)
        print(current.value)

        if current.left:
            queue.append(current.left)
        if current.right:
            queue.append(current.right)


def level_order_print(node):
    queue = []
    queue.append(node)
    queue.append(None)

    while len(queue) > 1:
        temp = queue.pop(0)
        if temp:

            print(temp.value, end=" ")
            if temp.left:
                queue.append(temp.left)
            if temp.right:
                queue.append(temp.right)

        else:

            queue.append(None)
            print()


def level_order_print2(node):
    queue = []
    queue.append(node)

    while queue:
        temp = []

        while queue:
            current = queue.pop(0)
            print(current.value, end=" ")

            if current.left:
                temp.append(current.left)
            if current.right:
                temp.append(current.right)

        queue = temp
        print()


def preorder(node):
    if not node:
        return
    print(node.value)
    preorder(node.left)
    preorder(node.right)


def postorder(node):
    if not node:
        return

    postorder(node.left)
    postorder(node.right)
    print(node.value)


def node_at_height(node, h):
    if not node:
        return

    node_at_height(node.left, h-1)
    if h == 1:
        print(node.value)
    node_at_height(node.right, h-1)


def height(node):
    if not node:
        return 0

    return 1 + max(height(node.left) , height(node.right))


def tree_diameter(node, diameter):
    if not node:
        return 0, diameter
    left_height, diameter = tree_diameter(node.left, diameter)
    right_height, diameter = tree_diameter(node.right, diameter)

    current_diameter = 1 + left_height + right_height
    diameter = max(diameter, current_diameter)

    return 1 + max(left_height, right_height), diameter


def max_val(node):
    if not node:
        return -1
    return max(node.value, max_val(node.left), max_val(node.right))

def main():

    root = Node(20)
    root.left = Node(9)
    root.right = Node(49)
    root.right.left = Node(23)
    root.right.right = Node(52)
    root.right.right.left = Node(50)
    root.left.left = Node(5)
    root.left.right = Node(12)
    root.left.right.right = Node(12)

    print("===InOrder===")
    inorder(root)
    print("===LevelOrder_n===")
    level_order(root)
    print("====LeveOrderPrint===")
    level_order_print(root)
    print("====LevelOrderPrint2===")
    level_order_print2(root)
    print("===InOrder Iterative===")
    inorder_iter(root)
    print("===PreOrder===")
    preorder(root)
    print("==PreOrder Iterative===")
    preorder_iter(root)
    print("===PostOrder===")
    postorder(root)
    print("===Height====")
    print(height(root))
    print("====Diameter====")
    print(tree_diameter(root, 0)[1])
    print("====NodeAtHeigt===")
    print(node_at_height(root, 2))
    print("====MaxVal=====")
    print(max_val(root))

if __name__ == "__main__":
    main()
