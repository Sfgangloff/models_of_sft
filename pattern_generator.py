from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from itertools import product

# ID of variables, depending on the position (i,j) and the index of the letter in the alphabet (a_idx).
def var_id(i, j, a_idx,alphabet,n):
    return i * n * len(alphabet) + j * len(alphabet) + a_idx + 1

# Creates a pattern from generated variables.
def decode_model(model,alphabet,n):
    pattern = {}
    for i, j in product(range(n), repeat=2):
        for k in range(len(alphabet)):
            if var_id(i, j, k,alphabet,n) in model:
                pattern[(i, j)] = alphabet[k]
    return pattern

# Encodes the nearest-neighbor shift constraints into a SAT problem. 
def encode_sft(n,alphabet,forbidden_pairs):
    solver = Solver()
    # Exactly-one constraints - at each position, there is exactly one letter of the alphabet.
    for i, j in product(range(n), repeat=2):
        vars_ij = [var_id(i, j, k,alphabet,n) for k in range(len(alphabet))]
        solver.append_formula(CardEnc.atleast(lits=vars_ij, bound=1, encoding=EncType.pairwise))
        solver.append_formula(CardEnc.atmost(lits=vars_ij, bound=1, encoding=EncType.pairwise))

    # Forbidden pairs
    for ((a1, a2), direction) in forbidden_pairs:
        idx1 = alphabet.index(a1)
        idx2 = alphabet.index(a2)
        if direction == 'horizontal':
            for i in range(n):
                for j in range(n - 1):
                    solver.add_clause([-var_id(i, j, idx1,alphabet,n), -var_id(i, j + 1, idx2,alphabet,n)])
        elif direction == 'vertical':
            for i in range(n - 1):
                for j in range(n):
                    solver.add_clause([-var_id(i, j, idx1,alphabet,n), -var_id(i + 1, j, idx2,alphabet,n)])
    return solver

# Generate at most a fixed number of patterns
def generate_patterns(n,alphabet,max_patterns,name,forbidden_pairs):
    all_patterns = []
    solver = encode_sft(n,alphabet,forbidden_pairs)

    while len(all_patterns) < max_patterns and solver.solve():
        model = solver.get_model()
        pattern = decode_model(model,alphabet,n)
        all_patterns.append(pattern)

        # Add blocking clause to exclude the current model
        block_clause = [-l for l in model if abs(l) <= n * n * len(alphabet)]
        solver.add_clause(block_clause)

    print(f"{len(all_patterns)} patterns generated.")

    import os

    output_dir = os.path.join("patterns", name)
    os.makedirs(output_dir, exist_ok=True)

    for idx, pattern in enumerate(all_patterns):
        filename = os.path.join(output_dir, f"pattern_{idx:03}.txt")
        with open(filename, "w") as f:
            for i in range(n):
                row = [pattern[(i, j)] for j in range(n)]
                f.write(' '.join(row) + '\n')

    print(f"{len(all_patterns)} patterns saved in folder '{output_dir}'")


if __name__ == "__main__":
    # Parameters
    N = 19  # box size: B_n = {0,...,n-1}^2
    ALPHABET = ['0', '1'] # alphabet of the shift
    NAME = 'full_shift' # name of the shift
    FORBIDDEN_PAIRS = [
    ] # forbidden dominos - the shift is assume to be nearest-neighbor.
    MAX_PATTERNS = 10000 # limit on the number of generated patterns.
    generate_patterns(N,ALPHABET,MAX_PATTERNS,NAME,FORBIDDEN_PAIRS)