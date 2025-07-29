"""
pattern_generator.py

This module uses a SAT solver to generate example patterns consistent with a nearest-neighbor
subshift of finite type (SFT). The SFT is defined by a finite alphabet and a list of forbidden
dominoes (i.e., 2-letter patterns in the horizontal or vertical direction).

Each pattern corresponds to an n×n configuration over the alphabet that avoids all forbidden
local patterns. These are encoded into Boolean constraints, and the SAT solver is used to search
for satisfying assignments, i.e., valid patterns.

Patterns are decoded from the solver model and saved as text files in a subfolder of "patterns/".

Dependencies:
- PySAT (https://pysathq.github.io/)
"""

import numpy as np

from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from itertools import product

# -----------------------------------------------------------------------------
# Variable Encoding Helpers
# -----------------------------------------------------------------------------

def var_id(i:int, j:int, a_idx:int, alphabet:list[str], n:int):
    """
    Computes a unique variable ID for the SAT solver based on position (i,j) and alphabet symbol index.

    Parameters:
        i (int): Row index (0 ≤ i < n)
        j (int): Column index (0 ≤ j < n)
        a_idx (int): Index of the letter in the alphabet (0 ≤ a_idx < len(alphabet))
        alphabet (List[str]): Finite alphabet of the shift
        n (int): Size of the pattern (producing an n x n grid)

    Returns:
        int: Unique positive integer representing the variable
    """
    assert 0 <= i < n, f"Row index i={i} out of bounds for grid size {n}"
    assert 0 <= j < n, f"Column index j={j} out of bounds for grid size {n}"
    assert 0 <= a_idx < len(alphabet), f"a_idx={a_idx} out of range for alphabet of size {len(alphabet)}"
    return i * n * len(alphabet) + j * len(alphabet) + a_idx + 1

# -----------------------------------------------------------------------------
# Decoding of SAT model
# -----------------------------------------------------------------------------

def decode_model(model:list[int], alphabet:list[str], n:int):
    """
    Converts a SAT model (a list of integers) into a dictionary representation
    of an n×n pattern with symbols from the alphabet.

    Parameters:
        model (List[int]): A satisfying assignment from the SAT solver
        alphabet (List[str]): Finite alphabet
        n (int): Grid size

    Returns:
        Dict[(int, int), str]: A mapping from grid coordinates to symbols
    """
    pattern = {}
    for i, j in product(range(n), repeat=2):
        for k in range(len(alphabet)):
            if var_id(i, j, k, alphabet, n) in model:
                pattern[(i, j)] = alphabet[k]
    return pattern

# -----------------------------------------------------------------------------
# Encoding of SFT constraints
# -----------------------------------------------------------------------------

def encode_forbidden_clauses(n,alphabet,forbidden_pairs):
    clauses = []
    for ((a1, a2), direction) in forbidden_pairs:
        idx1 = alphabet.index(a1)
        idx2 = alphabet.index(a2)
        if direction == 'horizontal':
            for i in range(n):
                for j in range(n - 1):
                    clauses.append([
                        -var_id(i, j, idx1, alphabet, n),
                        -var_id(i, j + 1, idx2, alphabet, n)
                    ])
        elif direction == 'vertical':
            for i in range(n - 1):
                for j in range(n):
                    clauses.append([
                        -var_id(i, j, idx1, alphabet, n),
                        -var_id(i + 1, j, idx2, alphabet, n)
                    ])
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'.")
    return clauses

def encode_sft(n:int, alphabet:list[str], forbidden_pairs):
    """
    Encodes the SFT constraints (nearest-neighbor exclusions and one-letter-per-cell constraint)
    into a SAT problem.

    Parameters:
        n (int): Grid size (n×n pattern)
        alphabet (List[str]): Finite alphabet
        forbidden_pairs (List[((str, str), str)]): List of forbidden dominoes, each represented
            as ((a, b), direction), where direction ∈ {"horizontal", "vertical"}

    Returns:
        Solver: A PySAT Solver instance containing the constraints
    """
    solver = Solver()

    # Enforce that each cell has exactly one letter
    for i, j in product(range(n), repeat=2):
        vars_ij = [var_id(i, j, k, alphabet, n) for k in range(len(alphabet))]
        solver.append_formula(CardEnc.atleast(lits=vars_ij, bound=1, encoding=EncType.pairwise))
        solver.append_formula(CardEnc.atmost(lits=vars_ij, bound=1, encoding=EncType.pairwise))

    forbidden_clauses = encode_forbidden_clauses(n,alphabet,forbidden_pairs)
    for clause in forbidden_clauses:
        solver.add_clause(clause)
    return solver

# -----------------------------------------------------------------------------
# Main Pattern Generation Function
# -----------------------------------------------------------------------------

def generate_patterns(n, alphabet, max_patterns, name, forbidden_pairs,save_as_txt):
    """
    Generates up to `max_patterns` valid patterns of size n×n avoiding forbidden pairs,
    and saves them to the disk under the folder "patterns/<name>/".

    Parameters:
        n (int): Grid size (n×n)
        alphabet (List[str]): Finite alphabet
        max_patterns (int): Maximum number of patterns to generate
        name (str): Name of the subshift, used as subfolder name
        forbidden_pairs (List[((str, str), str)]): Forbidden dominoes in the SFT
        save_as_txt (bool): if True, save the patterns as separate .txt files. Otherwise, save as a numpy array.
    """
    all_patterns = []
    solver = encode_sft(n, alphabet, forbidden_pairs)

    # Repeatedly extract satisfying models and block them to generate distinct patterns
    while len(all_patterns) < max_patterns and solver.solve():
        model = solver.get_model()
        pattern = decode_model(model, alphabet, n)
        all_patterns.append(pattern)

        # Add blocking clause to prevent generating the same pattern again
        block_clause = [-l for l in model if abs(l) <= n * n * len(alphabet)]
        solver.add_clause(block_clause)

    print(f"{len(all_patterns)} patterns generated.")

    # Create output directory
    import os

    if not os.path.exists("patterns"):
        os.makedirs("patterns")
    output_dir = os.path.join("patterns", name)
    os.makedirs(output_dir, exist_ok=True)

    if save_as_txt:
        # Save each pattern as a .txt file
        for idx, pattern in enumerate(all_patterns):
            filename = os.path.join(output_dir, f"pattern_{idx:03}.txt")
            with open(filename, "w") as f:
                for i in range(n):
                    row = [pattern[(i, j)] for j in range(n)]
                    f.write(' '.join(row) + '\n')
    else:
        # Save all the patterns in an array.
        pattern_arrays = []
        for pattern in all_patterns:
            array_2d = np.array([[pattern[(i, j)] for j in range(n)] for i in range(n)])
            pattern_arrays.append(array_2d)
        if pattern_arrays:
            # Stack into a 3D array of shape (num_patterns, n, n)
            data = np.stack(pattern_arrays)  # dtype='U1' if values are strings like '0' and '1'

            # Save to a .npy file
            output_path = os.path.join(output_dir, "all_patterns.npy")
            np.save(output_path, data)

    print(f"{len(all_patterns)} patterns saved in folder '{output_dir}'")

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Parameters
    N = 19                       # Box size: B_n = {0,...,n-1}^2
    ALPHABET = ['0', '1']        # Alphabet of the shift
    NAME = 'full_shift'          # Name of the shift (output subfolder)
    FORBIDDEN_PAIRS = [          # Forbidden nearest-neighbor dominos
        # Example: (('1', '1'), 'horizontal'), (('1', '1'), 'vertical')
    ]
    MAX_PATTERNS = 100         # Maximum number of distinct patterns to generate

    generate_patterns(N, ALPHABET, MAX_PATTERNS, NAME, FORBIDDEN_PAIRS,False)