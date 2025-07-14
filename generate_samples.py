"""
generate_samples.py

This script generates a specified number of nearest-neighbor subshifts of finite type (SFTs)
by randomly forbidding local domino constraints (2-cell patterns in horizontal or vertical direction)
over a given finite alphabet. For each such subshift, it creates a corresponding folder with 
sample patterns generated using the `generate_patterns` function, and records the forbidden 
pairs in a global JSON file for future reference.

The subshifts are named `subshift_k`, where `k` is an integer chosen to avoid overwriting 
existing subshift folders. The list of forbidden pairs for each subshift is saved in 
`samples.json` in the root directory.

Dependencies:
- `random_sft_generator.py`: provides `generate_random_sft`
- `pattern_generator.py`: provides `generate_patterns`
"""

import os
import re
import json
from random_sft_generator import generate_random_sft
from pattern_generator import generate_patterns

FOLDER_PATH = "patterns"
SAMPLES_PATH = "samples.json"

def get_max_subshift_index(folder_path):
    """
    Scans the given folder and returns the maximum integer index `k` such that 
    a subfolder named `subshift_k` exists. If no such folders exist, returns -1.

    Parameters:
        folder_path (str): Path to the folder containing subshift directories.

    Returns:
        int: Maximum index k such that 'subshift_k' exists.
    """
    max_index = -1
    pattern = re.compile(r"subshift_(\d+)")
    
    for name in os.listdir(folder_path):
        match = pattern.fullmatch(name)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    
    return max_index

def generate_sample(alphabet, forbid_prob, n, max_patterns, num_samples):
    """
    Generates a given number of random SFTs by forbidding local patterns with 
    independent probability, and stores their descriptions and example patterns.

    Each new subshift is saved in a folder `patterns/subshift_k`, where `k` is 
    chosen sequentially. Forbidden pairs are recorded in `samples.json`.

    Parameters:
        alphabet (List[str]): The finite alphabet from which symbols are drawn.
        forbid_prob (float): Probability that each 2-symbol pattern (domino) is forbidden.
        n (int): Size of the square box B_n = {0, ..., n-1}^2 for pattern generation.
        max_patterns (int): Maximum number of patterns to generate per subshift.
        num_samples (int): Number of new subshifts to generate.
    """
    max_idx = get_max_subshift_index(FOLDER_PATH)

    # Load or initialize the JSON dictionary storing forbidden pairs
    if os.path.exists(SAMPLES_PATH) and os.path.getsize(SAMPLES_PATH) > 0:
        with open(SAMPLES_PATH, 'r') as f:
            forbidden_dict = json.load(f)
    else:
        forbidden_dict = {}

    for k in range(num_samples):
        # Generate a new set of forbidden pairs for a random SFT
        forbidden_pairs = generate_random_sft(alphabet=alphabet, forbid_prob=forbid_prob)
        
        # Compute index and name for this new subshift
        idx = max_idx + k + 1
        name = f"subshift_{idx}"

        # Generate and save example patterns consistent with the SFT
        generate_patterns(n, alphabet, max_patterns, name, forbidden_pairs)

        # Store forbidden pairs in the JSON structure (as-is; ensure serializable format upstream)
        forbidden_dict[name] = forbidden_pairs

    # Save updated forbidden pair records to JSON file
    with open(SAMPLES_PATH, 'w') as f:
        json.dump(forbidden_dict, f, indent=2)

if __name__ == "__main__":
    # Example configuration: binary alphabet, 30% forbid probability
    ALPHABET = ['0', '1']
    FORBID_PROB = 0.3
    N = 19  # Size of square box for pattern generation: B_n = {0,...,n-1}^2
    MAX_PATTERNS = 4  # Max number of patterns to generate for each SFT
    NUM_SAMPLES = 1   # Number of SFTs to generate

    generate_sample(ALPHABET, FORBID_PROB, N, MAX_PATTERNS, NUM_SAMPLES)