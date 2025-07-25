import numpy as np

def merge_patterns_stack(subbox_stack, outside_subbox_stack, box_size):
    """
    Apply merge_patterns_subbox to each pair of 2D patterns in the stack.

    Parameters:
        subbox_stack (np.ndarray): shape (N, H, W), source for subboxes
        outside_subbox_stack (np.ndarray): shape (N, H, W), background
        box_size (int): size of the square subbox (box_size x box_size)

    Returns:
        np.ndarray: merged stack of shape (N, H, W)
    """
    assert subbox_stack.ndim == 3, f"Expected 3D stack, got shape {subbox_stack.shape}"
    assert subbox_stack.shape == outside_subbox_stack.shape, \
    f"Shape mismatch: subbox {subbox_stack.shape}, outside {outside_subbox_stack.shape}"

    N, H, W = subbox_stack.shape
    s = box_size
    assert s <= H and s <= W, \
    f"Subbox of size {s} does not fit in array of shape ({H},{W})"

    i0 = (H - s) // 2
    i1 = i0 + s
    j0 = (W - s) // 2
    j1 = j0 + s

    merged_stack = outside_subbox_stack.copy()
    merged_stack[:, i0:i1, j0:j1] = subbox_stack[:, i0:i1, j0:j1]
    return merged_stack

def subbox_all_negative_flags(stack, box_size):
    """
    For each 2D array in a stack, check whether all values in the centered subbox are negative.

    Parameters:
        stack (np.ndarray): array of shape (N, H, W)
        box_size (int): size of the square subbox (box_size x box_size)

    Returns:
        np.ndarray: array of shape (N,), with 1 if all values in subbox are negative, 0 otherwise
    """
    N, H, W = stack.shape
    s = box_size
    assert stack.ndim == 3, f"Expected 3D stack, got shape {stack.shape}"
    assert s <= H and s <= W, \
    f"Subbox of size {s} does not fit in array of shape ({H},{W})"

    i0 = (H - s) // 2
    i1 = i0 + s
    j0 = (W - s) // 2
    j1 = j0 + s

    subboxes = stack[:, i0:i1, j0:j1]  # shape (N, s, s)
    flags = np.all(subboxes < 0, axis=(1, 2))  # shape (N,)
    return flags.astype(np.int32)


def unzip_forbidden_patterns(forbidden_patterns):
     # Group forbidden patterns by direction
    forbidden_horizontal = set()
    forbidden_vertical = set()
    for pair, direction in forbidden_patterns:
        pair_tuple = tuple(pair)
        if direction == "horizontal":
            forbidden_horizontal.add(pair_tuple)
        elif direction == "vertical":
            forbidden_vertical.add(pair_tuple)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    return forbidden_horizontal,forbidden_vertical

def check_forbidden_horizontal(arr,forbidden_horizontal,valid):
    N, H, W = arr.shape
    assert W > 1, "Array width too small for horizontal pair checking"
    assert H > 1, "Array height too small for vertical pair checking"
    left = arr[:, :, :-1]    # (N, H, W-1)
    right = arr[:, :, 1:]    # (N, H, W-1)
    horz_pairs = np.stack([left, right], axis=-1)  # (N, H, W-1, 2)
    horz_pairs_flat = horz_pairs.reshape(-1, 2)     # (N*(H)*(W-1), 2)

    # Convert to tuple rows for matching
    horz_tuples = np.array([tuple(x) for x in horz_pairs_flat])
    matches = np.isin(horz_tuples.view([('', horz_tuples.dtype)]*2), 
                            np.array(list(forbidden_horizontal), dtype=horz_tuples.dtype).view([('', horz_tuples.dtype)]*2))
    matches = matches.reshape(N, H, W-1)
    valid &= ~matches.any(axis=(1, 2))
    return valid

def check_forbidden_vertical(arr,forbidden_vertical,valid):
    N, H, W = arr.shape
    assert W > 1, "Array width too small for horizontal pair checking"
    assert H > 1, "Array height too small for vertical pair checking"
    top = arr[:, :-1, :]     # (N, H-1, W)
    bottom = arr[:, 1:, :]   # (N, H-1, W)
    vert_pairs = np.stack([top, bottom], axis=-1)  # (N, H-1, W, 2)
    vert_pairs_flat = vert_pairs.reshape(-1, 2)     # (N*(H-1)*W, 2)

    # Convert to tuple rows for matching
    vert_tuples = np.array([tuple(x) for x in vert_pairs_flat])
    matches = np.isin(vert_tuples.view([('', vert_tuples.dtype)]*2), 
                          np.array(list(forbidden_vertical), dtype=vert_tuples.dtype).view([('', vert_tuples.dtype)]*2))
    matches = matches.reshape(N, H-1, W)
    valid &= ~matches.any(axis=(1, 2))
    return valid

def check_forbidden_patterns(arr, forbidden_patterns):
    """
    Check in parallel which 2D patterns in a 3D array contain forbidden horizontal or vertical patterns.

    Parameters:
        arr (np.ndarray): shape (N, H, W), numeric
        forbidden_patterns (list): each element is ([str, str], "horizontal"/"vertical")

    Returns:
        np.ndarray: shape (N,), with 0 if pattern contains a forbidden pair, 1 otherwise
    """
    arr_str = arr.astype(str)
    N, H, W = arr.shape

    forbidden_horizontal,forbidden_vertical = unzip_forbidden_patterns(forbidden_patterns)

    # Start with all valid
    valid = np.ones(N, dtype=bool)

    # --- Horizontal check ---
    if forbidden_horizontal:
        valid = check_forbidden_horizontal(arr=arr_str,forbidden_horizontal= forbidden_horizontal,valid=valid)

    # --- Vertical check ---
    if forbidden_vertical:
        valid = check_forbidden_vertical(arr=arr_str,forbidden_vertical=forbidden_vertical,valid=valid)

    return valid.astype(int)

def eval_subbox_to_outside(subbox_stack, outside_subbox_stack, box_size, forbidden_patterns):
    """
    Computes the fraction of 2D patterns that fail either the negative-subbox check or the forbidden pattern check.

    Parameters:
        subbox_stack (np.ndarray): shape (N, H, W)
        outside_subbox_stack (np.ndarray): shape (N, H, W)
        box_size (int): size of the square subbox
        forbidden_patterns (list): list of forbidden ([str, str], direction) pairs

    Returns:
        float: fraction of patterns that fail at least one check
    """
    assert subbox_stack.shape == outside_subbox_stack.shape, "Stacks must have the same shape"

    # # Step 1: check for negative-only subboxes in the outside pattern
    # checks_negative = subbox_all_negative_flags(outside_subbox_stack, box_size)  # shape (N,)

    # Step 2: merge the subbox and the outside patterns
    merged_stack = merge_patterns_stack(subbox_stack, outside_subbox_stack, box_size)  # shape (N, H, W)
    merged_stack = merged_stack.astype(str)

    # Step 3: check forbidden patterns in the merged result
    checks_forbidden_patterns = check_forbidden_patterns(merged_stack, forbidden_patterns)  # shape (N,)

    # # Step 4: OR between the two checks → violation if either is true
    # success = (checks_negative & checks_forbidden_patterns)  # shape (N,)
    
    # Step 4: OR between the two checks → violation if either is true
    success = (checks_forbidden_patterns)  # shape (N,)

    return int(success.sum()), int(success.size)

if __name__ == "__main__":
    arr = np.array([
        [[0, 1], [2, 3]],   # forbidden "0", "1" vertical → invalid
        [[3, 3], [3, 3]],   # all 3s → valid
        [[0, 2], [3, 3]],   # valid
        [[1, 0], [3, 3]]    # forbidden "1", "0" horizontal → invalid
    ])

    forbidden_patterns = [
        [["0", "1"], "vertical"],
        [["0", "2"], "vertical"],
        [["1", "0"], "horizontal"]
    ]

    print(check_forbidden_patterns(arr, forbidden_patterns))