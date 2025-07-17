"""
subpattern_extractor.py

This module provides utility functions for modifying 2D symbolic patterns (stored as .txt files)
by masking selected regions with the symbol '*'. These functions operate on folders of pattern files
and create new versions with the specified regions masked or preserved.

The following types of masking are supported:
- Masking a centered sub-box
- Masking a "crown" region (an outer box minus an inner box)
- Keeping only a centered sub-box and masking everything else
"""

import os
import numpy as np

def keep_box_in_patterns(input_dir, output_dir, box_size):
    """
    For each pattern file in input_dir, keeps only the values inside a centered square subbox,
    masking everything else with '*'. Saves the result in output_dir.

    Parameters:
        input_dir (str): Path to the folder with original .txt pattern files.
        output_dir (str): Path to the folder where masked patterns will be saved.
        box_size (int): Size of the square region to keep (box_size x box_size), centered in the grid.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            with open(input_path, "r") as f:
                lines = [line.strip().split() for line in f.readlines()]

            n_rows = len(lines)
            n_cols = len(lines[0]) if n_rows > 0 else 0

            box_top = (n_rows - box_size) // 2
            box_left = (n_cols - box_size) // 2

            masked_lines = []
            for i in range(n_rows):
                row = []
                for j in range(n_cols):
                    if box_top <= i < box_top + box_size and box_left <= j < box_left + box_size:
                        row.append(lines[i][j])
                    else:
                        row.append("*")
                masked_lines.append(row)

            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w") as f:
                for row in masked_lines:
                    f.write(" ".join(row) + "\n")

    print(f"Masked (keep only box) patterns saved in: {output_dir}")

def keep_box_in_numpy_stack(input_path, output_path, box_size):
    """
    Loads a 3D numpy array from `input_path`, where each 2D slice along axis 0 is a pattern.
    Keeps only a centered `box_size x box_size` region in each slice, masking everything else with `*`.
    Saves the masked 3D array to `output_path`.

    Parameters:
        input_path (str): Path to the input `.npy` file (3D array: num_patterns × rows × cols)
        output_path (str): Path to save the masked `.npy` file
        box_size (int): Size of the square region to keep
        mask_symbol (str): Value to use outside the box (default is -1)
    """
    patterns = np.load(input_path, allow_pickle=True)

    if patterns.ndim != 3:
        raise ValueError("Input array must be 3D (stack of 2D patterns)")

    num_patterns, n_rows, n_cols = patterns.shape
    box_top = (n_rows - box_size) // 2
    box_left = (n_cols - box_size) // 2

    masked_patterns = np.full_like(patterns, fill_value="*")

    for idx in range(num_patterns):
        masked_patterns[idx,
                        box_top:box_top + box_size,
                        box_left:box_left + box_size] = patterns[idx,
                                                                 box_top:box_top + box_size,
                                                                 box_left:box_left + box_size]

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, masked_patterns)
    print(f"Masked array saved to: {output_path}")


def mask_crown_in_patterns(input_dir, output_dir, outer_box_size, inner_box_size):
    """
    Masks a crown region in each pattern file: the area inside the outer box but outside the inner box
    is replaced with '*'. The inner box remains untouched. The masking is centered on the pattern.

    Parameters:
        input_dir (str): Folder containing input .txt pattern files.
        output_dir (str): Folder to save crown-masked patterns.
        outer_box_size (int): Size of the outer square region to be masked.
        inner_box_size (int): Size of the inner square to remain visible (must be strictly smaller).
    """
    assert outer_box_size > inner_box_size, "Outer box must be strictly larger than inner box."
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            with open(input_path, "r") as f:
                lines = [line.strip().split() for line in f.readlines()]

            n_rows = len(lines)
            n_cols = len(lines[0]) if n_rows > 0 else 0

            outer_top = (n_rows - outer_box_size) // 2
            outer_left = (n_cols - outer_box_size) // 2
            inner_top = (n_rows - inner_box_size) // 2
            inner_left = (n_cols - inner_box_size) // 2

            masked_lines = []
            for i in range(n_rows):
                row = []
                for j in range(n_cols):
                    if outer_top <= i < outer_top + outer_box_size and outer_left <= j < outer_left + outer_box_size:
                        if inner_top <= i < inner_top + inner_box_size and inner_left <= j < inner_left + inner_box_size:
                            row.append(lines[i][j])  # keep inner box
                        else:
                            row.append("*")  # mask crown region
                    else:
                        row.append(lines[i][j])  # outside outer box
                masked_lines.append(row)

            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w") as f:
                for row in masked_lines:
                    f.write(" ".join(row) + "\n")

    print(f"Crown-masked patterns saved in: {output_dir}")


def mask_subbox_in_patterns(input_dir, output_dir, box_size):
    """
    For each pattern file in input_dir, replaces the values inside a centered square subbox with '*'
    and saves the modified pattern to output_dir.

    Parameters:
        input_dir (str): Path to the folder with original .txt pattern files.
        output_dir (str): Path to the folder where masked patterns will be saved.
        box_size (int): Size of the subbox to mask (box_size x box_size), centered in the grid.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            with open(input_path, "r") as f:
                lines = [line.strip().split() for line in f.readlines()]
            
            n_rows = len(lines)
            n_cols = len(lines[0]) if n_rows > 0 else 0

            box_top = (n_rows - box_size) // 2
            box_left = (n_cols - box_size) // 2

            masked_lines = []
            for i in range(n_rows):
                row = []
                for j in range(n_cols):
                    if box_top <= i < box_top + box_size and box_left <= j < box_left + box_size:
                        row.append("*")
                    else:
                        row.append(lines[i][j])
                masked_lines.append(row)

            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w") as f:
                for row in masked_lines:
                    f.write(" ".join(row) + "\n")

    print(f"Masked patterns saved in: {output_dir}")

def mask_subbox_in_numpy_stack(input_path, output_path, box_size):
    """
    Loads a 3D numpy array from `input_path`, where each 2D slice along axis 0 is a pattern.
    Replaces the values inside a centered `box_size x box_size` square with `mask_symbol`.
    Saves the modified 3D array to `output_path`.

    Parameters:
        input_path (str): Path to the input `.npy` file (shape: num_patterns × rows × cols)
        output_path (str): Path to save the masked `.npy` file
        box_size (int): Size of the square subbox to mask
        mask_symbol (str): Value to place inside the subbox (default is -1)
    """
    patterns = np.load(input_path, allow_pickle=True)

    if patterns.ndim != 3:
        raise ValueError("Input array must be 3D (stack of 2D patterns)")

    num_patterns, n_rows, n_cols = patterns.shape
    box_top = (n_rows - box_size) // 2
    box_left = (n_cols - box_size) // 2

    # Copy original patterns to avoid in-place mutation
    masked_patterns = np.copy(patterns)

    # Apply masking
    for idx in range(num_patterns):
        masked_patterns[idx,
                        box_top:box_top + box_size,
                        box_left:box_left + box_size] = "*"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, masked_patterns)
    print(f"Masked subboxes saved to: {output_path}")


if __name__ == "__main__":

    # mask_subbox_in_patterns(
    #     input_dir="patterns/subshift_0",
    #     output_dir="subbox_masked_patterns/subshift_0",
    #     box_size=7
    # )

    # mask_crown_in_patterns(
    #     input_dir="patterns/subshift_0",
    #     output_dir="crown_masked_patterns/subshift_0",
    #     outer_box_size=11,
    #     inner_box_size=5
    # )

    # keep_box_in_patterns(
    #     input_dir="patterns/subshift_0",
    #     output_dir="outside_subbox_masked_patterns/subshift_0",
    #     box_size=5
    # )

    keep_box_in_numpy_stack("patterns/full_shift/all_patterns.npy", "subbox_masked_patterns/full_shift/all_patterns.npy", 5)