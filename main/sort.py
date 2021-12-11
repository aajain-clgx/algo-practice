#!/usr/bin/env python

def bubble_sort(arr):

    n = len(arr)
    swapped = False

    for i in range(n):
        for j in range(n-1-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True

            if not swapped:
                break


def insertion_sort(arr):

    n = len(arr)
    for i in range(n):
        j = i-1
        key = arr[i]

        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key


def selection_sort(arr):
    n = len(arr)

    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if arr[min_index] > arr[j]:
                min_index = j

        arr[min_index], arr[i] = arr[i], arr[min_index]


def merge_sort(arr):

    if len(arr) < 2:
        return arr

    middle = len(arr) // 2
    result = []

    left = merge_sort(arr[:middle])
    right = merge_sort(arr[middle:])

    while left and right:
        if left[0] < right[0]:
            result.append(left.popleft(0))
        else:
            result.append(right.popleft(0))

    if left:
        result.extend(left)
    if right:
        result.extend(right)

    return result


class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def merge_sort_ll(node):

    def middle(node):

        if not node or not node.left:
            return node, None

        slow = node
        fast = node

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        second = slow.next
        slow.next = None

        return node, second

    if not node or not node.next:
        return node

    left, right = middle(node)
    result = None

    left_ll = merge_sort_ll(left)
    right_ll = merge_sort_ll(right)

    current = result

    while left_ll and right_ll:
        if left_ll.value > right_ll.value:
            current.Next = left_ll
        else:
            current.next = right_ll
        current = current.next

    if left_ll:
        current.next = left_ll
    if right_ll:
        current.next = right_ll

    return result


def quick_sort(arr, low, high):

    def partition(arr, low, high):
        pivot_index = low + (high - low) // 2
        pivot = arr[pivot_index]

        while low < high:
            while arr[low] <= pivot:
                low += 1
            while arr[high] > pivot:
                high -= 1

            if low >= high:
                return high

            arr[low], arr[high] = arr[high], arr[low]
            low += 1
            high -= 1

    if low < high:
        pivot_index = partition(arr, low, high)
        quick_sort(arr, low, pivot_index)
        quick_sort(arr, pivot_index+1, high)


def quickselect(arr, low, high, k):

    def partition(arr, low, high):
        pivot_index = low + (high - low) // 2
        pivot = arr[pivot_index]

        while low < high:
            while arr[low] < pivot:
                low += 1

            while arr[high] > pivot:
                high -= 1

            if low >= high:
                return high

            arr[low], arr[high] = arr[high], arr[low]
            low += 1
            high -= 1

    if k > 0 and k <= (high - low + 1):

        pivot_index = partition(arr, low, high)

        if pivot_index - low == k-1:
            return arr[pivot_index]

        if pivot_index-low < k-1:
            quickselect(arr, pivot_index+1, high, k)
        else:
            quickselect(arr, low, pivot_index, k)
    else:
        return None


def dutch_flag(arr):

    p0 = 0
    current = 0
    p2 = len(arr)-1

    while current <= p2:

        if arr[current] == 2:
            arr[p2], arr[current] = arr[current], arr[p2]
            p2 -= 1

        elif arr[current] == 1:
            current += 1

        elif arr[current] == 0:
            arr[p0], arr[current] = arr[current], arr[p0]
            p0 += 1


def counting_sort(arr, k):

    output = [0] * len(arr)
    counts = [0] * k

    for i in range(len(arr)):
        counts[arr[i]] += 1

    for i in range(1, len(arr)):
        counts[i] += count[i-1]

    for i in range(len(arr)-1, -1 , -1)):
        key = arr[i]
        count[key] = -1
        output[count[key]] = key

    for i in range(len(arr)):
        arr[i] = output[i]


    return arr


def radix_sort(arr):

    def count_sort(arr, base, exp):

        n = len(arr)
        output = [0] * n
        counts = [0] * base

        for i in range(n):
            count[arr[i] // exp] += 1

        for i in range(1, n):
            count[i] += count[i-1]

        for i in range(n-1, -1, -1):
            index = arr[i] // exp
            count[i] -= 1
            output[index] = arr[i]

        for i in range(n):
            arr[i] = output[i]


    max_len = max(arr)
    exp = 1
    while max_len < exp * 10:
        count_sort(arr, 10, exp)
        exp *= 10









def quicksort(arr):

    def partition(arr, low, high):

        pivot = arr[low + (high-low) //2]
        i, j = low-1, high+1

        while True:
            while True:
                i += 1
                if a[i] >= pivot:
                    break

            while True:
                j  -= 1
                if a[j] < pivot:
                    break

            if i >= j:
                return j

            a[i], a[j] = a[j], a[i]


    def quick_sort_util(arr, low, high):
        if low < high:
            pivot = partition(arr, low, high)
            quicksort(arr, low, pivot-1)
            quicksort(arr, pivot+1, high)



    def quickselect(arr, low, high, k):

        if k > 0 and k <= (high-low+1):
            part


def main():
    pass


if __name__ == "__main__":
    main()
