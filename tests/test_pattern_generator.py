import unittest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pattern_generator import var_id, decode_model,encode_forbidden_clauses

# === Unit tests ===

class TestPatternGenerator(unittest.TestCase):

    def test_var_id_uniqueness(self):
        alphabet = ['a', 'b', 'c']
        n = 3
        ids = set()
        for i in range(n):
            for j in range(n):
                for k in range(len(alphabet)):
                    vid = var_id(i, j, k, alphabet, n)
                    self.assertIsInstance(vid, int)
                    self.assertGreater(vid, 0)
                    self.assertNotIn(vid, ids)
                    ids.add(vid)

    def test_var_id_known_case(self):
        alphabet = ['a', 'b']
        n = 2
        # Manually verify computation: var_id(1,1,1) = 1*2*2 + 1*2 + 1 + 1 = 8
        self.assertEqual(var_id(1, 1, 1, alphabet, n), 8)

    def test_decode_model_simple(self):
        alphabet = ['x', 'y']
        n = 2
        model = [
            var_id(0, 0, 0, alphabet, n),  # (0,0) → 'x'
            var_id(0, 1, 1, alphabet, n),  # (0,1) → 'y'
            var_id(1, 0, 1, alphabet, n),  # (1,0) → 'y'
            var_id(1, 1, 0, alphabet, n),  # (1,1) → 'x'
        ]
        expected = {
            (0, 0): 'x',
            (0, 1): 'y',
            (1, 0): 'y',
            (1, 1): 'x'
        }
        decoded = decode_model(model, alphabet, n)
        self.assertEqual(decoded, expected)

    def test_decode_model_ignores_unset_variables(self):
        alphabet = ['a', 'b']
        n = 1
        # Only one variable set: (0,0) → 'b'
        model = [var_id(0, 0, 1, alphabet, n)]
        decoded = decode_model(model, alphabet, n)
        self.assertEqual(decoded, {(0, 0): 'b'})
        self.assertNotIn((0, 0), [k for k, v in decoded.items() if v == 'a'])

class TestForbiddenClauses(unittest.TestCase):

    def setUp(self):
        self.alphabet = ['0', '1']
        self.n = 2  # 2x2 grid

    def test_horizontal_forbidden_pair(self):
        forbidden = [(('0', '1'), 'horizontal')]
        clauses = encode_forbidden_clauses(self.n, self.alphabet, forbidden)

        expected = [
            [-var_id(0, 0, 0, self.alphabet, self.n), -var_id(0, 1, 1, self.alphabet, self.n)],
            [-var_id(1, 0, 0, self.alphabet, self.n), -var_id(1, 1, 1, self.alphabet, self.n)],
        ]
        self.assertEqual(len(clauses), 2)
        self.assertCountEqual(clauses, expected)

    def test_vertical_forbidden_pair(self):
        forbidden = [(('1', '0'), 'vertical')]
        clauses = encode_forbidden_clauses(self.n, self.alphabet, forbidden)

        expected = [
            [-var_id(0, 0, 1, self.alphabet, self.n), -var_id(1, 0, 0, self.alphabet, self.n)],
            [-var_id(0, 1, 1, self.alphabet, self.n), -var_id(1, 1, 0, self.alphabet, self.n)],
        ]
        self.assertEqual(len(clauses), 2)
        self.assertCountEqual(clauses, expected)

    def test_horizontal_and_vertical(self):
        forbidden = [
            (('0', '1'), 'horizontal'),
            (('1', '0'), 'vertical'),
        ]
        clauses = encode_forbidden_clauses(self.n, self.alphabet, forbidden)
        self.assertEqual(len(clauses), 4)  # 2 horizontal + 2 vertical

    def test_no_forbidden(self):
        clauses = encode_forbidden_clauses(self.n, self.alphabet, [])
        self.assertEqual(clauses, [])

if __name__ == '__main__':
    unittest.main()