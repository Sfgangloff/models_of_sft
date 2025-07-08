# 🧩 Pattern Generator for Nearest-Neighbor 2D Shifts of Finite Type

This Python script uses a SAT solver to generate locally admissible patterns on a finite \( N 	imes N \) grid for a 2D **nearest-neighbor shift of finite type (SFT)**. Each generated pattern satisfies the shift constraints locally, and is saved in a structured format for inspection or further processing.

---

## 📦 Features

- Customizable alphabet and shift definition via **forbidden neighbor pairs**
- Supports **nearest-neighbor SFTs** (horizontal and vertical adjacency)
- Uses a **SAT solver** to efficiently generate admissible configurations
- Saves patterns in individual `.txt` files as clean 2D grids
- Limits the number of generated patterns to avoid combinatorial explosion

---

## 🚀 Getting Started

### 1. Install the required dependency

```bash
pip install python-sat
```

### 2. Run the script

```bash
python pattern_generator.py
```

This will create a directory named `patterns/<SHIFT_NAME>/` containing up to `MAX_PATTERNS` pattern files.

---

## 🛠 Configuration

Edit the top of `main.py` to configure the following parameters:

```python
N = 19                     # Size of the box: B_n = {0,...,n-1}^2
ALPHABET = ['0', '1']      # Alphabet for the SFT
NAME = 'full_shift'        # Name used for the output folder
FORBIDDEN_PAIRS = [        # Forbidden adjacent pairs (nearest-neighbor)
    # Example: (('0', '1'), 'horizontal'),
    #          (('1', '1'), 'vertical'),
]
MAX_PATTERNS = 3           # Number of patterns to generate
```

### Forbidden Pair Format

Each forbidden pattern is a tuple of the form:

```python
((symbol1, symbol2), direction)
```

Where `direction` is either `'horizontal'` or `'vertical'`. For example:

```python
FORBIDDEN_PAIRS = [
    (('1', '1'), 'horizontal'),
    (('1', '1'), 'vertical')
]
```

Forbids two adjacent `1`s in both directions.

---

## 📂 Output

The generated patterns are saved in:

```
patterns/
└── <SHIFT_NAME>/
    ├── pattern_000.txt
    ├── pattern_001.txt
    ├── ...
```

Each file contains a readable \( N 	imes N \) grid such as:

```
0 0 1 1
1 0 0 1
...
```

---

## 🧬 Random SFT Generator

This project includes a simple way to generate random nearest-neighbor shifts of finite type (SFTs) by sampling forbidden adjacent symbol pairs (also called dominoes) from the set of all possible pairs over a finite alphabet. Each domino (e.g., ('1', '0')) may be randomly forbidden in either the horizontal or vertical direction, or both. This randomized process defines a locally constrained 2D shift space. Once such a shift is defined by its forbidden pairs, it can be passed to the SAT-based pattern generator, which searches for locally admissible patterns on finite boxes. You can generate a random SFT by calling the provided function generate_random_sft(...), then use its output as the FORBIDDEN_PAIRS input to the main pattern generation script.


## ⚠️ Notes

- Generated patterns are **locally admissible**: they satisfy all constraints inside the finite box. Global extendibility is not guaranteed.
- No symmetry reduction is applied: patterns may be equivalent up to translation or rotation.
- Pattern blocking is handled via **blocking clauses**, ensuring no repeats among the generated outputs.
