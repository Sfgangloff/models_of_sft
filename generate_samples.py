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
from subpattern_extractor import keep_box_in_patterns, mask_crown_in_patterns, mask_subbox_in_patterns,keep_box_in_numpy_stack,mask_subbox_in_numpy_stack
from utils import empty_folder

FOLDER_PATH = "patterns"
SAMPLES_PATH = "samples.json"

def get_max_subshift_index(folder_path:str):
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

def generate_sample(alphabet:list[str], 
                    forbid_prob:float, 
                    n:int,
                    max_patterns:int, 
                    num_samples:int,
                    subbox_size:int,
                    save_as_txt:bool,
                    remove_previous_samples:bool):
    """
    Generates a given number of random SFTs by forbidding local patterns with 
    independent probability, and stores their descriptions and example patterns, as well as 
    extracted subpatterns.

    Each new subshift is saved in a folder `patterns/subshift_k`, where `k` is 
    chosen sequentially. Forbidden pairs are recorded in `samples.json`.

    Parameters:
        alphabet (List[str]): The finite alphabet from which symbols are drawn.
        forbid_prob (float): Probability that each 2-symbol pattern (domino) is forbidden.
        n (int): Size of the square box B_n = {0, ..., n-1}^2 for pattern generation.
        max_patterns (int): Maximum number of patterns to generate per subshift.
        num_samples (int): Number of new subshifts to generate.
        subbox_size (int): Size of the subbox used to extract subpatterns.
        save_as_txt (bool): if True, save the patterns as separate .txt files. Otherwise, save as a numpy array.
        remove_previous_samples (bool): if True, removes the previously created samples.
    """
    if remove_previous_samples:
        with open("samples.json", "w") as f:
            f.write("{}")
        empty_folder("patterns")
        empty_folder("outside_subbox_masked_patterns")
        empty_folder("subbox_masked_patterns")
        empty_folder("crown_masked_patterns")
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
        generate_patterns(n, alphabet, max_patterns, name, forbidden_pairs,save_as_txt)

        # Store forbidden pairs in the JSON structure (as-is; ensure serializable format upstream)
        forbidden_dict[name] = {}
        forbidden_dict[name]["forbidden_pairs"] = forbidden_pairs
        forbidden_dict[name]["alphabet"] = alphabet

        input_dir = os.path.join(FOLDER_PATH, name)
        output_dir = os.path.join("subbox_masked_patterns",name)

        input_path = os.path.join(input_dir,"all_patterns.npy")
        output_path = os.path.join(output_dir,"all_patterns.npy")
        if save_as_txt:
            mask_subbox_in_patterns(
                input_dir=input_dir,
                output_dir=output_dir,
                box_size=subbox_size
            )
        
        else: 
            mask_subbox_in_numpy_stack(input_path, output_path, subbox_size)

        output_dir = os.path.join("outside_subbox_masked_patterns",name)
        output_path = os.path.join(output_dir,"all_patterns.npy")
        if save_as_txt:
            keep_box_in_patterns(
                input_dir=input_dir,
                output_dir=output_dir,
                box_size=subbox_size
            )
        else: 
            keep_box_in_numpy_stack(input_path, output_path, subbox_size)



    # Save updated forbidden pair records to JSON file
    with open(SAMPLES_PATH, 'w') as f:
        json.dump(forbidden_dict, f, indent=2)

if __name__ == "__main__":
    import yaml

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Access parameters
    ALPHABET     = config["alphabet"]
    FORBID_PROB  = config["forbidden_propability"]
    BOX_SIZE     = config["box_size"]
    MAX_PATTERNS = config["max_patterns"]
    NUM_SAMPLES  = config["num_samples"]
    SUBBOX_SIZE  = config["subbox_size"]

    generate_sample(alphabet=ALPHABET, 
                    forbid_prob=FORBID_PROB, 
                    n=BOX_SIZE, 
                    max_patterns=MAX_PATTERNS, 
                    num_samples=NUM_SAMPLES,
                    subbox_size=SUBBOX_SIZE,
                    save_as_txt=False,
                    remove_previous_samples=True)