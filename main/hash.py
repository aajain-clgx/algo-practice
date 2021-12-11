#!/usr/bin/env python3

from collections import defaultdict


def longest_0_1(arr):

    sum_dict = {}
    longest_sub = 0
    c_sum = 0
    for i in range(len(arr)):
        val = -1 if arr[i] == 0 else arr[i]
        c_sum += val
        if c_sum in sum_dict:
            longest_sub = max(longest_sub, i - sum_dict[c_sum])
        else:
            sum_dict[sum] = i

    return longest_sub


def count_all_equal_0_1(arr):
    sum_dict = {}
    count = 0
    c_sum = 0

    for i in range(len(arr)):
        val = -1 if arr[i] == 0 else arr[i]
        c_sum += val
        if val in sum_dict:
            count += 1
        else:
            sum_dict[c_sum] = 1

    return count


def twosum(arr, k):
"""
    Time: O(n)
    Space: O(n)
"""
    sum_dict = {}
    solutions = []

    for i, n in enumerate(arr):
        diff = k - arr[i]
        if diff in sum_dict:
            solutions.append((sum_dict[diff], i))
        else:
            sum_dict[i] = n

    return solutions


def twosum_2(arr, k):
"""
    Time: O(nlogn) if not sorted,
          O(n) if pre-sorted
    Space: O(1)
"""

    left, right = 0, len(arr)-1

    sorted(arr)

    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum > k:
            right -= 1
        elif current_sum < k:
            left += 1
        else:
            solutions.add((left, right))

    return solutions


def threesum(arr, k):
    result = []
    arr.sort()
    n = len(arr)

    for i in range(n-2):
        if i > 0 and arr[i] == arr[i-1]:
            continue

        left, right = i+1, len(arr) - 1

        while left < right:

            current_sum = arr[i] + arr[left] + arr[right]
            if current_sum > k:
                right -= 1
            elif current_sum < k:
                left += 1
            else:
                result.append((i, left, right))
                while arr[left] == arr[left+1] and left < right:
                    left += 1
                while arr[right] == arr[right-1] and left < right:
                    right -= 1

                left += 1
                right -= 1

    return result


def foursum(arr, k):

    result = []
    arr.sort()
    n = len(arr)

    for i in range(n-3):
        if i > 0 and arr[i] == arr[i-1]:
            continue

        for j in range(i+1, n-2):
            if j != i+1 and arr[j] == arr[j-1]:
                continue

            left = j+1
            right = len(arr)-1
            while left < right:

                current_sum = arr[i] + arr[j] + arr[left] + arr[right]
                if current_sum > k:
                    right -= 1
                elif current_sum < k:
                    left += 1

                else:
                    result.append((i, j, left, right))
                    while left < right and arr[left] == arr[left+1]:
                        left += 1
                    while left < right and arr[right] == arr[right-1]:
                        right -= 1

                    left += 1
                    right -= 1

    return result


def count_triple_product(arr, p):

    count = 0
    n = len(arr)
    d = {arr[i]: i for i in range(n)}

    for i in range(n-1):
        if arr[i] != 0 and p % arr[i] == 0:

            for j in range(i+1, n):
                if arr[j] != 0 and (p % (arr[i]*arr[j])) == 0:
                    q, r = divmod(p, arr[i] * arr[j])
                    if r in d:
                        if d[r] != i and d[r] != j and (d[r] > i and d[r] > j):
                            count += 1
    return count


def anagrams(arr):
    d = defaultdict(list)

    for w in arr:

        key = "".join(sorted(w))
        d[key].appnd(w)

    for i in d:
        print(d[i])


def longest_substr_nonrepeating(s):

    n = len(s)
    left = 0
    right = 0
    max_len = 0

    d = set()

    while right < n:
        if s[right] not in d:
            d.add(s[right])
            right += 1
            max_len = max(max_len, right-left+1)
        else:
            while left != right:
                d.remove(s[left])
                left += 1

    return max_len


