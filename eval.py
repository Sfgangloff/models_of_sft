import numpy as np

def merge_patterns_subbox(subbox_pattern, outside_subbox_pattern, box_size):
    """
    Merge two 2D arrays by taking a centered subbox of given size from A,
    and the rest from B.

    Parameters:
        A, B (np.ndarray): arrays of shape (H, W)
        box_size (int): size of the square subbox (box_size x box_size)

    Returns:
        np.ndarray: merged array
    """
    assert subbox_pattern.shape == outside_subbox_pattern.shape, "Arrays must have the same shape"
    H, W = subbox_pattern.shape
    s = box_size
    assert s <= H and s <= W, "Subbox must fit inside the array"

    i0 = (H - s) // 2
    i1 = i0 + s
    j0 = (W - s) // 2
    j1 = j0 + s

    C = outside_subbox_pattern.copy()
    C[i0:i1, j0:j1] = subbox_pattern[i0:i1, j0:j1]
    return C