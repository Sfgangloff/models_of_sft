import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eval import merge_patterns_subbox

def test_merge_patterns_subbox():
    # Case 1: Basic 5x5, subbox size 3
    pattern_1 = np.full((5, 5), 1)
    pattern_2 = np.full((5, 5), 0)
    merged_pattern = merge_patterns_subbox(pattern_1, pattern_2, 3)

    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    assert np.array_equal(merged_pattern, expected), "Subbox merge failed on 5x5 with size 3"

    # Case 2: Full replacement (box_size = 5)
    merged_pattern = merge_patterns_subbox(pattern_1, pattern_2, 5)
    assert np.array_equal(merged_pattern, pattern_1), "Subbox merge failed on full box replacement"

    # Case 3: box_size = 1
    pattern_1 = np.full((3, 3), 1)
    pattern_2 = np.full((3, 3), 0)
    merged_pattern = merge_patterns_subbox(pattern_1, pattern_2, 1)
    expected = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    assert np.array_equal(merged_pattern, expected), "Subbox merge failed on box_size=1"