import random
from itertools import product

def generate_random_sft(alphabet, forbid_prob=0.1, seed=None):
    """
    Randomly generates a nearest-neighbor SFT by forbidding dominoes
    with a given probability independently in each direction.
    
    Returns a list of forbidden pairs: [ ((a, b), direction), ... ]
    """
    if seed is not None:
        random.seed(seed)
    
    directions = ['horizontal', 'vertical']
    forbidden_pairs = []

    for a, b in product(alphabet, repeat=2):
        for direction in directions:
            if random.random() < forbid_prob:
                forbidden_pairs.append(((a, b), direction))
    
    return forbidden_pairs



if __name__ == "__main__":
    ALPHABET = ['0','1','2','3']
    FORBID_PROB = 0.3
    FORBIDDEN_PAIRS = generate_random_sft(ALPHABET, forbid_prob=FORBID_PROB, seed=42)

    print("Randomly generated forbidden pairs:")
    for pair in FORBIDDEN_PAIRS:
        print(pair)