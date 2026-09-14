"""Tiny exact-arithmetic calibration oracles; test code, not a KV fallback.

These cases distinguish the paper's global-token/per-head normalization from
plausible but different chunk/rank averaging algorithms. No model tensors are
loaded and no reference module is imported or modified.
"""
from fractions import Fraction as F
from math import sqrt
import unittest


def zeros(d):
    return [[F(0) for _ in range(d)] for _ in range(d)]


def outer(x):
    return [[F(a) * F(b) for b in x] for a in x]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(a, factor):
    return [[x * factor for x in row] for row in a]


def gram(rows):
    total = zeros(len(rows[0]))
    for row in rows:
        total = add(total, outer(row))
    return scale(total, F(1, len(rows)))


def quadratic(row, matrix):
    return sum(F(row[i]) * matrix[i][j] * F(row[j])
               for i in range(len(row)) for j in range(len(row)))


def paper_covariances(q, k, v):
    tokens, heads, dim = len(q), len(k[0]), len(k[0][0])
    gqa = len(q[0]) // heads
    key_cov, value_cov = zeros(dim), zeros(dim)
    for head in range(heads):
        query_rows = [q[t][h] for t in range(tokens)
                      for h in range(head * gqa, (head + 1) * gqa)]
        qcov = gram(query_rows)
        key_cov = add(key_cov, qcov)
        weights = [quadratic(k[t][head], qcov) for t in range(tokens)]
        denominator = max(sum(weights), F(1, 10**12))
        contribution = zeros(dim)
        for token, weight in enumerate(weights):
            contribution = add(contribution, scale(outer(v[token][head]), weight / denominator))
        value_cov = add(value_cov, contribution)
    return scale(key_cov, F(1, heads)), scale(value_cov, F(1, heads))


def matrix_multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def hadamard(d):
    if d == 1:
        return [[1.0]]
    h = hadamard(d // 2)
    return [[x / sqrt(2) for x in row + row] for row in h] + [
        [x / sqrt(2) for x in row + [-v for v in row]] for row in h]


def bit_reverse(d):
    width = d.bit_length() - 1
    return [int(format(i, f"0{width}b")[::-1], 2) for i in range(d)]


def permutation(eigenvalues):
    d = len(eigenvalues)
    order = sorted(range(d), key=lambda i: eigenvalues[i], reverse=True)
    perm = [None] * d
    for i, destination in enumerate(bit_reverse(d)):
        perm[destination] = order[i]
    return perm, [[F(int(row == col)) for col in perm] for row in range(d)]


class CalibrationMathContractTest(unittest.TestCase):
    def setUp(self):
        self.q = [[[1, 0], [2, 0]], [[0, 1], [0, 2]]]
        self.k = [[[1, 0], [2, 0]], [[0, 3], [0, 1]]]
        self.v = [[[1, 0], [3, 0]], [[0, 1], [0, 2]]]

    def test_global_qqt_and_per_head_normalized_sst_exact_values(self):
        key, value = paper_covariances(self.q, self.k, self.v)
        self.assertEqual(key, [[F(5, 4), 0], [0, F(5, 4)]])
        self.assertEqual(value, [[F(73, 20), 0], [0, F(17, 20)]])
        # Pooling raw weights across heads weights head1 twice as much as head0.
        pooled_wrong = [[F(29, 6), 0], [0, F(5, 6)]]
        self.assertNotEqual(value, pooled_wrong)

    def test_sst_chunk_normalization_cannot_precede_global_merge(self):
        _, full = paper_covariances(self.q, self.k, self.v)
        _, chunk0 = paper_covariances(self.q[:1], self.k[:1], self.v[:1])
        _, chunk1 = paper_covariances(self.q[1:], self.k[1:], self.v[1:])
        wrong = scale(add(chunk0, chunk1), F(1, 2))
        self.assertEqual(wrong, [[F(5, 2), 0], [0, F(5, 4)]])
        self.assertNotEqual(full, wrong)

    def test_qqt_unequal_chunk_lengths_require_token_weighting(self):
        rows = [[1, 0], [0, 2], [0, 2]]
        full = gram(rows)
        correct = scale(add(scale(gram(rows[:1]), 1), scale(gram(rows[1:]), 2)), F(1, 3))
        wrong = scale(add(gram(rows[:1]), gram(rows[1:])), F(1, 2))
        self.assertEqual(full, [[F(1, 3), 0], [0, F(8, 3)]])
        self.assertEqual(full, correct)
        self.assertNotEqual(full, wrong)

    def test_gqa_normalizes_token_times_query_group_count(self):
        # Duplicate each query head within its matching GQA group. Dividing by
        # T alone would double QQT; the paper divides by T*G and is unchanged.
        q4 = [[token[0], token[0], token[1], token[1]] for token in self.q]
        self.assertEqual(paper_covariances(q4, self.k, self.v),
                         paper_covariances(self.q, self.k, self.v))

    def test_two_pass_sst_merges_raw_moments_before_normalization(self):
        _, expected = paper_covariances(self.q, self.k, self.v)
        result = zeros(2)
        for head in range(2):
            global_qcov = gram([token[head] for token in self.q])
            partial_numerators, partial_denominators = [], []
            for token in range(2):
                weight = quadratic(self.k[token][head], global_qcov)
                partial_numerators.append(scale(outer(self.v[token][head]), weight))
                partial_denominators.append(weight)
            merged = add(*partial_numerators)
            result = add(result, scale(merged, F(1, sum(partial_denominators))))
        self.assertEqual(scale(result, F(1, 2)), expected)

    def test_zero_score_head_yields_zero_sst_without_identity_substitution(self):
        q = [[[0, 0]], [[0, 0]]]
        k = [[[7, 4]], [[1, 2]]]
        v = [[[3, 8]], [[4, 1]]]
        key, value = paper_covariances(q, k, v)
        self.assertEqual(key, zeros(2))
        self.assertEqual(value, zeros(2))

    def test_permutation_is_columns_with_bit_reversed_destinations(self):
        self.assertEqual(bit_reverse(8), [0, 4, 2, 6, 1, 5, 3, 7])
        perm, matrix = permutation([3, 1, 8, 2, 7, 0, 6, 4])
        self.assertEqual(perm, [2, 0, 6, 1, 4, 3, 7, 5])
        self.assertEqual(matrix_multiply([list(range(8))], matrix)[0], perm)
        # Script sorts signed eigenvalues, not their absolute values.
        self.assertEqual(permutation([-F(1, 10**7), 0, 1, 2])[0], [3, 1, 2, 0])

    def test_normalized_hadamard_and_composition_order(self):
        h = hadamard(4)
        product = matrix_multiply(h, list(zip(*h)))
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(product[i][j], float(i == j), places=12)
        _, p = permutation([4, 1, 3, 2])
        rhp = matrix_multiply(h, p)  # U=I; exact recommended U@H@P order
        rph = matrix_multiply(p, h)
        self.assertNotEqual(rhp, rph)


if __name__ == "__main__":
    unittest.main()
