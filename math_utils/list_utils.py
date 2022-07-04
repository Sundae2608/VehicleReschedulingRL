"""
Contain some list utilities
"""


def rotate_list(array, n):
    """
    Rotate all elements in the list to the right by n indidces
    """
    n = n % (len(array))
    return array[n:] + array[:n]