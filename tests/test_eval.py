import numpy as np

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eval import merge_patterns_stack, subbox_all_negative_flags, check_forbidden_patterns,unzip_forbidden_patterns,check_forbidden_horizontal,check_forbidden_vertical

def test_merge_patterns_stack():
    # Case 1: Basic 5x5, subbox size 3
    pattern_1 = np.full((5, 5), 1)
    pattern_2 = np.full((5, 5), 0)

    subbox_stack = np.stack([pattern_1])
    outside_stack = np.stack([pattern_2])

    merged_stack = merge_patterns_stack(subbox_stack, outside_stack, 3)

    expected = np.array([[
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]])
    assert np.array_equal(merged_stack, expected), "Stacked subbox merge failed on 5x5 with size 3"

    # Case 2: Full replacement (box_size = 5)
    subbox_stack = np.stack([pattern_1])
    outside_stack = np.stack([pattern_2])
    merged_stack = merge_patterns_stack(subbox_stack, outside_stack, 5)
    assert np.array_equal(merged_stack, np.stack([pattern_1])), "Full box replacement failed"

    # Case 3: box_size = 1
    pattern_1 = np.full((3, 3), 1)
    pattern_2 = np.full((3, 3), 0)

    subbox_stack = np.stack([pattern_1])
    outside_stack = np.stack([pattern_2])
    merged_stack = merge_patterns_stack(subbox_stack, outside_stack, 1)

    expected = np.array([[
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]])
    assert np.array_equal(merged_stack, expected), "Subbox merge failed on box_size=1"

    # Case 4: Multiple examples in batch
    stack_1 = np.stack([np.full((3, 3), i) for i in [1, 2]])
    stack_2 = np.stack([np.full((3, 3), 0) for _ in range(2)])
    merged_stack = merge_patterns_stack(stack_1, stack_2, 1)

    expected = np.array([
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0]
        ]
    ])
    assert np.array_equal(merged_stack, expected), "Batch processing failed on multiple examples"


def test_subbox_all_negative_flags():
    # Case 1: All values in subbox are negative
    stack = np.array([
        [[-1, -2, -3],
         [-4, -5, -6],
         [-7, -8, -9]]
    ])
    result = subbox_all_negative_flags(stack, 3)
    expected = np.array([1])
    assert np.array_equal(result, expected), "Failed case with all negative values in subbox"

    # Case 2: Some values in subbox are non-negative
    stack = np.array([
        [[-1, -2, -3],
         [-4,  0, -6],
         [-7, -8, -9]]
    ])
    result = subbox_all_negative_flags(stack, 3)
    expected = np.array([0])
    assert np.array_equal(result, expected), "Failed case with 0 in subbox"

    # Case 3: Multiple 2D arrays
    stack = np.array([
        [[-1, -1, -1],
         [-1, -1, -1],
         [-1, -1, -1]],     # all negative → 1
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],        # all positive → 0
        [[0, -1, -1],
         [-1, -1, -1],
         [-1, -1, -1]],     # 0 in subbox → 0
    ])
    result = subbox_all_negative_flags(stack, 3)
    expected = np.array([1, 0, 0])
    assert np.array_equal(result, expected), "Failed batch processing with mixed cases"

    # Case 4: box_size = 1, center value negative
    stack = np.array([
        [[1, 1, 1],
         [1, -1, 1],
         [1, 1, 1]],
    ])
    result = subbox_all_negative_flags(stack, 1)
    expected = np.array([1])
    assert np.array_equal(result, expected), "Failed for box_size=1 with negative center"

    # Case 5: box_size = 1, center value zero
    stack = np.array([
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
    ])
    result = subbox_all_negative_flags(stack, 1)
    expected = np.array([0])
    assert np.array_equal(result, expected), "Failed for box_size=1 with non-negative center"

    # Case 6: box_size = full size, with one non-negative
    stack = np.array([
        [[-1, -2],
         [-3,  0]],
    ])
    result = subbox_all_negative_flags(stack, 2)
    expected = np.array([0])
    assert np.array_equal(result, expected), "Failed for full box with one non-negative value"

    print("All subbox_all_negative_flags tests passed.")


def test_check_forbidden_patterns():
    # Case 1: Basic 2x2, vertical match
    arr = np.array([
        [[0, 1],
         [1, 1]],  # forbidden ["0", "1"] vertically in col 0
        [[1, 2],
         [3, 4]]   # no forbidden pattern
    ])
    forbidden_patterns = [
        [["0", "1"], "vertical"]
    ]
    expected = np.array([0, 1])
    result = check_forbidden_patterns(arr, forbidden_patterns)
    assert np.array_equal(result, expected), "Failed on simple vertical match"

    # Case 2: Basic 2x2, horizontal match
    arr = np.array([
        [[1, 0],
         [3, 3]],  # forbidden ["1", "0"] horizontally in row 0
        [[1, 1],
         [2, 2]]   # no forbidden pattern
    ])
    forbidden_patterns = [
        [["1", "0"], "horizontal"]
    ]
    expected = np.array([0, 1])
    result = check_forbidden_patterns(arr, forbidden_patterns)
    assert np.array_equal(result, expected), "Failed on simple horizontal match"

    # Case 3: Both directions
    arr = np.array([
        [[0, 2],
         [3, 4]],  # ["0", "2"] vertically in col 0
        [[1, 0],
         [2, 2]],  # ["1", "0"] horizontally
        [[3, 3],
         [3, 3]]   # no match
    ])
    forbidden_patterns = [
        [["0", "2"], "vertical"],
        [["1", "0"], "horizontal"]
    ]
    expected = np.array([1, 0, 1])
    result = check_forbidden_patterns(arr, forbidden_patterns)
    assert np.array_equal(result, expected), "Failed on combined match"

    # Case 4: No forbidden patterns
    arr = np.ones((3, 3, 3))
    forbidden_patterns = []
    expected = np.array([1, 1, 1])
    result = check_forbidden_patterns(arr, forbidden_patterns)
    assert np.array_equal(result, expected), "Failed on empty forbidden list"

    # Case 5: Pattern at boundary
    arr = np.array([
        [[1, 2, 0],
         [1, 2, 0],
         [1, 2, 0]]  # ["2", "0"] vertically in col 1 and 2
    ])
    forbidden_patterns = [
        [["2", "0"], "vertical"]
    ]
    expected = np.array([1])
    result = check_forbidden_patterns(arr, forbidden_patterns)
    assert np.array_equal(result, expected), "Failed on vertical pattern at edge"

    # Case 6: Multiple matches in one image
    arr = np.array([
        [[0, 1, 0],
         [1, 2, 0]]  # ["0", "1"] horiz at row 0; ["1", "2"] horiz at row 1
    ])
    forbidden_patterns = [
        [["0", "1"], "horizontal"],
        [["1", "2"], "horizontal"]
    ]
    expected = np.array([0])
    result = check_forbidden_patterns(arr, forbidden_patterns)
    assert np.array_equal(result, expected), "Failed on multiple matches in one pattern"

    print("All tests passed.")

class TestUnzipForbiddenPatterns(unittest.TestCase):

    def test_empty_input(self):
        fh, fv = unzip_forbidden_patterns([])
        self.assertEqual(fh, set())
        self.assertEqual(fv, set())

    def test_single_horizontal(self):
        input_patterns = [(["0", "1"], "horizontal")]
        fh, fv = unzip_forbidden_patterns(input_patterns)
        self.assertEqual(fh, {("0", "1")})
        self.assertEqual(fv, set())

    def test_single_vertical(self):
        input_patterns = [(["1", "2"], "vertical")]
        fh, fv = unzip_forbidden_patterns(input_patterns)
        self.assertEqual(fh, set())
        self.assertEqual(fv, {("1", "2")})

    def test_mixed_patterns(self):
        input_patterns = [
            (["0", "0"], "horizontal"),
            (["1", "2"], "vertical"),
            (["2", "2"], "horizontal")
        ]
        fh, fv = unzip_forbidden_patterns(input_patterns)
        self.assertEqual(fh, {("0", "0"), ("2", "2")})
        self.assertEqual(fv, {("1", "2")})

    def test_raises_on_invalid_direction(self):
        input_patterns = [(["0", "1"], "diagonal")]
        with self.assertRaises(ValueError) as context:
            unzip_forbidden_patterns(input_patterns)
        self.assertIn("Unknown direction", str(context.exception))

class TestForbiddenChecks(unittest.TestCase):

    def test_check_forbidden_horizontal_positive(self):
        arr = np.array([
            [[0, 1],
             [2, 0]],
            [[1, 1],
             [0, 2]]
        ])  # shape (2, 2, 2)
        valid = np.ones(2, dtype=bool)
        forbidden_horizontal = {("0", "1")}  # Appears in sample 0, row 0
        arr_str = arr.astype(str)
        valid_out = check_forbidden_horizontal(arr_str, forbidden_horizontal, valid.copy())
        self.assertFalse(valid_out[0])
        self.assertTrue(valid_out[1])

    def test_check_forbidden_vertical_negative(self):
        arr = np.array([
            [[1, 0],
             [0, 2]],
            [[1, 1],
             [0, 0]]
        ])  # shape (2, 2, 2)
        valid = np.ones(2, dtype=bool)
        forbidden_vertical = {("1", "0")}  # Appears in sample 0 and 1, col 0
        arr_str = arr.astype(str)
        valid_out = check_forbidden_vertical(arr_str, forbidden_vertical, valid.copy())
        self.assertFalse(valid_out[0])
        self.assertFalse(valid_out[1])

    def test_check_forbidden_horizontal_none(self):
        arr = np.array([
            [[0, 0],
             [2, 2]],
            [[1, 0],
             [0, 1]]
        ])
        valid = np.ones(2, dtype=bool)
        forbidden_horizontal = {("1", "2")}  # Not present
        arr_str = arr.astype(str)
        valid_out = check_forbidden_horizontal(arr_str, forbidden_horizontal, valid.copy())
        np.testing.assert_array_equal(valid_out, np.ones(2, dtype=bool))

    def test_check_forbidden_vertical_all(self):
        arr = np.array([
            [[0, 1],
             [2, 2]],
            [[1, 0],
             [1, 0]]
        ])
        valid = np.ones(2, dtype=bool)
        forbidden_vertical = {("0", "2"), ("1", "1")}  # Match in both samples
        arr_str = arr.astype(str)
        valid_out = check_forbidden_vertical(arr_str, forbidden_vertical, valid.copy())
        self.assertFalse(valid_out[0])
        self.assertFalse(valid_out[1])

if __name__ == '__main__':
    unittest.main()

if __name__ == "__main__":
    test_merge_patterns_stack()
    test_subbox_all_negative_flags()
    test_check_forbidden_patterns()
    print("All tests passed.")