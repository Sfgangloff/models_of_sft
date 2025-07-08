from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from itertools import product

# Parameters
N = 19  # box size: B_n = {0,...,n-1}^2
ALPHABET = ['0', '1'] # alphabet of the shift
NAME = 'quadrant' # name of the shift
FORBIDDEN_PAIRS = [(('0', '1'), 'horizontal'),
                   (('0', '1'), 'vertical'),
] # forbidden dominos - the shift is assume to be nearest-neighbor.
MAX_PATTERNS = 15 # limit on the number of generated patterns.

# ID of variables, depending on the position (i,j) and the index of the letter in the alphabet (a_idx).
def var_id(i, j, a_idx):
    return i * N * len(ALPHABET) + j * len(ALPHABET) + a_idx + 1

# Creates a pattern from generated variables.
def decode_model(model):
    pattern = {}
    for i, j in product(range(N), repeat=2):
        for k in range(len(ALPHABET)):
            if var_id(i, j, k) in model:
                pattern[(i, j)] = ALPHABET[k]
    return pattern

# Encodes the nearest-neighbor shift constraints into a SAT problem. 
def encode_sft():
    solver = Solver()
    # Exactly-one constraints - at each position, there is exactly one letter of the alphabet.
    for i, j in product(range(N), repeat=2):
        vars_ij = [var_id(i, j, k) for k in range(len(ALPHABET))]
        solver.append_formula(CardEnc.atleast(lits=vars_ij, bound=1, encoding=EncType.pairwise))
        solver.append_formula(CardEnc.atmost(lits=vars_ij, bound=1, encoding=EncType.pairwise))

    # Forbidden pairs
    for ((a1, a2), direction) in FORBIDDEN_PAIRS:
        idx1 = ALPHABET.index(a1)
        idx2 = ALPHABET.index(a2)
        if direction == 'horizontal':
            for i in range(N):
                for j in range(N - 1):
                    solver.add_clause([-var_id(i, j, idx1), -var_id(i, j + 1, idx2)])
        elif direction == 'vertical':
            for i in range(N - 1):
                for j in range(N):
                    solver.add_clause([-var_id(i, j, idx1), -var_id(i + 1, j, idx2)])
    return solver

# Generate at most a fixed number of patterns
all_patterns = []
solver = encode_sft()

while len(all_patterns) < MAX_PATTERNS and solver.solve():
    model = solver.get_model()
    pattern = decode_model(model)
    all_patterns.append(pattern)

    # Add blocking clause to exclude the current model
    block_clause = [-l for l in model if abs(l) <= N * N * len(ALPHABET)]
    solver.add_clause(block_clause)

print(f"{len(all_patterns)} patterns generated.")

import os

OUTPUT_DIR = os.path.join("patterns", NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, pattern in enumerate(all_patterns):
    filename = os.path.join(OUTPUT_DIR, f"pattern_{idx:03}.txt")
    with open(filename, "w") as f:
        for i in range(N):
            row = [pattern[(i, j)] for j in range(N)]
            f.write(' '.join(row) + '\n')

print(f"{len(all_patterns)} patterns saved in folder '{OUTPUT_DIR}'")