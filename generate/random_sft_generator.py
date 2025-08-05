"""
random_sft_generator.py

This module provides functionality to generate a random nearest-neighbor subshift
of finite type (SFT) over a given alphabet, by probabilistically forbidding local 
2-symbol patterns (dominoes) in horizontal and vertical directions.

Each SFT is specified by a list of forbidden dominoes, which are pairs of symbols 
with a direction (either 'horizontal' or 'vertical'). Forbidden pairs are sampled 
independently with a given probability.

This tool is useful for experimental or empirical investigations of generic SFTs.

Typical usage:
    forbidden_pairs = generate_random_sft(['0','1'], forbid_prob=0.2)
"""

import random
from itertools import product

def generate_random_sft(alphabet, forbid_prob=0.1, seed=None):
    """
    Generates a random list of forbidden nearest-neighbor dominoes defining a 2D SFT.

    For each ordered pair of symbols (a, b) from the alphabet, and for each direction
    (horizontal or vertical), a random number in [0,1) is drawn. If it is less than 
    `forbid_prob`, the domino ((a, b), direction) is added to the list of forbidden pairs.

    Parameters:
        alphabet (List[str]): Finite set of symbols used in the configuration
        forbid_prob (float): Probability with which each domino is forbidden (default: 0.1)
        seed (Optional[int]): Optional random seed for reproducibility (default: None)

    Returns:
        List[((str, str), str)]: A list of forbidden dominoes, where each element is a tuple 
        of the form ((a, b), direction) with a, b ∈ alphabet and direction ∈ {'horizontal', 'vertical'}
    """
    if seed is not None:
        random.seed(seed)
    
    directions = ['horizontal', 'vertical']
    forbidden_pairs = []

    for a, b in product(alphabet, repeat=2):
        for direction in directions:
            rand_number = random.random()
            if rand_number < forbid_prob:
                forbidden_pairs.append(((a, b), direction))
    
    return forbidden_pairs

# -----------------------------------------------------------------------------
# Example usage and debug
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    ALPHABET = ['0', '1']        # Binary alphabet
    FORBID_PROB = 0.25           # 25% probability for each domino to be forbidden
    FORBIDDEN_PAIRS = generate_random_sft(ALPHABET, forbid_prob=FORBID_PROB)

    print("Randomly generated forbidden pairs:")
    for pair in FORBIDDEN_PAIRS:
        print(pair)