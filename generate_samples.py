import os
import re
import json
from random_sft_generator import generate_random_sft
from pattern_generator import generate_patterns

FOLDER_PATH = "patterns"
SAMPLES_PATH = "samples.json"

def get_max_subshift_index(folder_path):
    max_index = -1
    pattern = re.compile(r"subshift_(\d+)")
    
    for name in os.listdir(folder_path):
        match = pattern.fullmatch(name)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    
    return max_index

def generate_sample(alphabet,forbid_prob,n,max_patterns,num_samples):
    max_idx = get_max_subshift_index(FOLDER_PATH)
    if os.path.exists(SAMPLES_PATH):
        with open(SAMPLES_PATH, 'r') as f:
            forbidden_dict = json.load(f)
    else:
        forbidden_dict = {}
    for k in range(num_samples):
        forbidden_pairs = generate_random_sft(alphabet=alphabet,forbid_prob=forbid_prob)
        idx = max_idx + k + 1
        name = f"subshift_{idx}"
        generate_patterns(n,alphabet,max_patterns,name,forbidden_pairs)
        forbidden_dict[name] = forbidden_pairs

    with open(SAMPLES_PATH, 'w') as f:
        json.dump(forbidden_dict, f, indent=2)

if __name__ == "__main__":
    ALPHABET = ['0','1']
    FORBID_PROB = 0.3
    N = 19  # box size: B_n = {0,...,n-1}^2
    MAX_PATTERNS = 4 # limit on the number of generated patterns.
    NUM_SAMPLES = 1
    generate_sample(ALPHABET,FORBID_PROB,N,MAX_PATTERNS,NUM_SAMPLES)