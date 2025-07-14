import os

def keep_box_in_patterns(input_dir, output_dir, box_size):
    """
    For each pattern file in input_dir, keeps only the values inside a centered square subbox,
    masking everything else with '*'. Saves the result in output_dir.

    Parameters:
        input_dir (str): Path to the folder with original .txt pattern files.
        output_dir (str): Path to the folder where masked patterns will be saved.
        box_size (int): Size of the square region to keep (box_size x box_size).
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


def mask_crown_in_patterns(input_dir, output_dir, outer_box_size, inner_box_size):
    """
    Masks a crown region in each pattern file: the area inside the outer box but outside the inner box
    is replaced with '*'. The inner box remains untouched. The masking is centered on the pattern.

    Parameters:
        input_dir (str): Folder containing input .txt pattern files.
        output_dir (str): Folder to save crown-masked patterns.
        outer_box_size (int): Size of the full square region to be masked (outer boundary).
        inner_box_size (int): Size of the inner square to remain visible (must be < outer_box_size).
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
    For each pattern file in input_dir, replaces the values inside a square subbox with '*'
    and saves the modified pattern to output_dir.

    Parameters:
        input_dir (str): Path to the folder with original .txt pattern files.
        output_dir (str): Path to the folder where masked patterns will be saved.
        box_size (int): Size of the box to mask (box_size x box_size).
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            with open(input_path, "r") as f:
                lines = [line.strip().split() for line in f.readlines()]
            
            n_rows = len(lines)
            n_cols = len(lines[0]) if n_rows > 0 else 0

            # Compute top-left corner of the box (centered)
            box_top = (n_rows - box_size) // 2
            box_left = (n_cols - box_size) // 2

            # Mask the subbox
            masked_lines = []
            for i in range(n_rows):
                row = []
                for j in range(n_cols):
                    if box_top <= i < box_top + box_size and box_left <= j < box_left + box_size:
                        row.append("*")
                    else:
                        row.append(lines[i][j])
                masked_lines.append(row)

            # Save to output directory
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w") as f:
                for row in masked_lines:
                    f.write(" ".join(row) + "\n")

    print(f"Masked patterns saved in: {output_dir}")

if __name__ == "__main__":

    # mask_subbox_in_patterns(
    #     input_dir="patterns/subshift_0",      # folder with pattern_000.txt, etc.
    #     output_dir="subbox_masked_patterns/subshift_0",  # where to save masked patterns
    #     box_size=7                            # size of the subbox to mask
    # )

    # mask_crown_in_patterns(
    #     input_dir="patterns/subshift_0",               # Folder with original patterns
    #     output_dir="crown_masked_patterns/subshift_0",        # Folder to save crown-masked patterns
    #     outer_box_size=11,                             # Outer square size (centered)
    #     inner_box_size=5                               # Inner square to keep visible
    # )

    keep_box_in_patterns(
        input_dir="patterns/subshift_0",               # Folder with original patterns
        output_dir="outside_subbox_masked_patterns/subshift_0",      # Folder to save masked patterns
        box_size=5                                     # Size of the inner box to keep
    )
