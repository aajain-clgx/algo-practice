#!/usr/bin/env python3

def bsearch(arr, left, right, n):
    if left <= right:
        mid = left + (right - left)//2

        if arr[mid] == n:
            return mid
        elif arr[mid] < n:
            return bsearch(arr, left+1, right, n)
        else:
            return bsearch(arr, left, right - 1, n)

    else:
        return -1


def bsearch_iter(arr, n):
    left = 0
    right = len(arr) - 1
,,
    while lef t <= right:
        mid = left + (right - left) // 2

        if arr[mid] == n:
            return mid
        elif arr[mid] < n:
            left = left + 1
        else:
            right = right - 1

    return -1


def floor(arr, left, right, n):

    if left <= right:
        if n >= arr[right]:
            return right

        mid = left + (right - left) // 2

        if arr[mid] == n:
            return mid

        if mid  > 1 and arr[mid-1] <= n < arr[mid]:
            return mid - 1
        elif n < arr[mid]:
            return floor(arr, left, right-1, n)
        else:
            return floor(arr, left+1, right, n)

    else:
        return -1


def median(arr1, arr2):

    if not arr1  and not arr2:
        return -1

    if len(arr1) == 1:
        return (arr1[0] + arr2[0]) // 2
    elif len(arr1) == 2:
        return max(arr1[0], arr2[0]) + min(arr1[1], arr2[1])  // 2

    else:
        m1 = median(arr1)
        m2 = median(arr2)

        if m1 < m2:
            return median(arr1[m1:]




def main():
    l = [1, 2, 3, 5, 6, 7, 9, 10]

    print(bsearch(l, 0, len(l)-1, 7))
    print(bsearch_iter(l, 7))


if __name__ == "__main__":
    main() 
