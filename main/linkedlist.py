#!/usr/bin/env python3


def reverse_list(head):

    current = head
    prev = None

    while current:
        next = current.next
        current.next = prev
        prev = current
        current = next

    return prev


def reverse_list_recur(head):

    if head is None or head.next is None:
        return head

    rest = reverse_list_recur(head.next)

    rest.next = head
    head.next = None

    return rest


def nth_from_last(node, n):

    if node is None:
        return 1

    index_from_last = nth_from_last(node.next, n)

    if index_from_last == n:
        print(node)

    return index_from_last + 1


def middle_node(head):
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow


def has_loop(head):

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True

    return False


def is_palindrome(head):

    if not head or not head.next:
        return True

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    slow.next = reverse_list(slow.next)
    slow = slow.next

    while slow:
        if slow.value != head.value:
            return False
        slow = slow.next
        head = head.next

    return True
