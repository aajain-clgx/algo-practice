#!/usr/bin/env python3


class QueueUsingStack:

    def __init__(self):
        self.s1 = []
        self.s2 = []

    def enqueue(self, value):
        self.s1.append(value)

    def dequeue(self):

        if len(self.s2) == 0 and len(self.s1) == 0:
            return None

        if len(self.s2) == 0 and len(self.s1) > 0:
            while self.s1:
                self.s2.append(self.s1.pop())
            return self.s2.pop()

        if len(self.s2) > 0:
            return self.s2.pop()

    def is_empty(self):
        return len(self.s1) == 0 and len(self.s2) == 0


class QueueUsingArray:

    def __init__(self, size):
        self.front = -1
        self.rear = -1
        self.len = size
        self.arr = [] * size

    def is_full(self):
        if self.front == self.rear:
            return True
        else:
            return False

    def is_empty(self):
        return self.front == -1 and self.rear == -1

    def enque(self, val):

        if self.is_full():
            return False

        if self.is_empty():
            self.front = self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.len

        self.arr[self.rear] = val

        return True

    def deque(self):

        if self.is_empty():
            return False

        if self.is_full():
            self.rear = self.front = -1

        value = self.arr[self.front]
        self.front = (self.front + 1) % self.len

        return value


class LRUNode:
    def __init__(self, key, val):
        self.key = key
        self.value = val
        self.next = None
        self.prev = None


class LRUDeque:

    def __init__(self):
        self.head = LRUNode('head', 'head')
        self.tail = LRUNode('tail', 'tail')
        self.head.next = self.tail
        self.tail.prev = self.head

    def remove_node(self, node):
        prev_node = node.prev
        next_node = node.next

        prev_node.next = next_node
        next_node.prev = prev_node

    def move_to_front(self, node):

        current_front = self.head.next
        node.next = current_front
        current_front.prev = node
        node.prev = self.head
        self.head.next = node

    def get_tail_node(self):
        return self.tail.prev


class LRUCache:

    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = LRUDeque()
        self.cache = {}

    def get(self, key):

        if key not in self.cache:
            return None

        node = self.cache[key]
        val = node.value

        self.queue.remove_node(node)
        self.queue.move_to_front(node)

        return val

    def put(self, key, val):

        if key in self.cache:
            self.get(key)
        else:
            node = LRUNode(key, val)
            if len(self.cache) < self.capacity:
                self.queue.move_to_front(node)
            else:
                remove_node = self.queue.get_tail_node()
                key_to_remove = remove_node.key
                remove_node = None
                del self.cache[key_to_remove]

                self.queue.remove(remove_node)
                self.queue.move_to_front(node)


def main():
    pass


if __name__ == "__main__":
    main()
