"""Confidence formulas against the worked examples in docs/SPEC.md, section 6."""

import unittest

from nex.confidence import choice_confidence, noul_confidence, score_confidence


class NoulConfidenceTest(unittest.TestCase):
    def test_distance_from_half(self):
        for p, expected in [(0.5, 0.0), (1.0, 1.0), (0.0, 1.0), (0.75, 0.5), (0.25, 0.5), (0.9, 0.8), (0.1, 0.8)]:
            self.assertAlmostEqual(noul_confidence(p), expected)

    def test_matches_abs_2p_minus_1(self):
        for i in range(101):
            p = i / 100
            self.assertAlmostEqual(noul_confidence(p), abs(2 * p - 1))


class ChoiceConfidenceTest(unittest.TestCase):
    def test_docs_examples(self):
        # Both have confidence 0.4 because only the top option matters.
        self.assertAlmostEqual(choice_confidence([0.6, 0.3, 0.1]), 0.4)
        self.assertAlmostEqual(choice_confidence([0.6, 0.2, 0.2]), 0.4)

    def test_order_does_not_matter(self):
        self.assertAlmostEqual(choice_confidence([0.1, 0.3, 0.6]), 0.4)

    def test_uniform_is_zero(self):
        for n in range(2, 21):
            self.assertAlmostEqual(choice_confidence([1 / n] * n), 0.0)

    def test_one_hot_is_one(self):
        for n in range(2, 21):
            for k in range(n):
                probs = [0.0] * n
                probs[k] = 1.0
                self.assertAlmostEqual(choice_confidence(probs), 1.0)

    def test_two_options(self):
        self.assertAlmostEqual(choice_confidence([0.75, 0.25]), 0.5)

    def test_single_option_is_certain(self):
        self.assertEqual(choice_confidence([1.0]), 1.0)


class ScoreConfidenceTest(unittest.TestCase):
    def test_docs_examples(self):
        self.assertAlmostEqual(score_confidence([0.0, 0.5, 0.5]), 0.25)
        self.assertAlmostEqual(score_confidence([0.5, 0.0, 0.5]), 0.0)
        self.assertAlmostEqual(score_confidence([0.0, 0.57, 0.43]), 0.355, places=6)

    def test_neighbours_cost_less_than_distant_levels(self):
        near = score_confidence([0.0, 0.0, 0.6, 0.4, 0.0])
        far = score_confidence([0.4, 0.0, 0.6, 0.0, 0.0])
        self.assertGreater(near, far)

    def test_uniform_is_zero(self):
        for n in range(2, 11):
            self.assertEqual(score_confidence([1 / n] * n), 0.0)

    def test_one_hot_is_one(self):
        for n in range(2, 11):
            for k in range(n):
                probs = [0.0] * n
                probs[k] = 1.0
                self.assertAlmostEqual(score_confidence(probs), 1.0)

    def test_ten_levels(self):
        # MAD of a uniform spread over 0..9 around 4.5 is 2.5.
        probs = [0.0] * 10
        probs[3], probs[4], probs[5] = 0.25, 0.5, 0.25
        self.assertAlmostEqual(score_confidence(probs), 1 - 0.5 / 2.5)

    def test_never_negative(self):
        self.assertEqual(score_confidence([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]), 0.0)

    def test_two_levels(self):
        # MAD of a uniform spread over 0..1 is 0.5.
        self.assertAlmostEqual(score_confidence([0.8, 0.2]), 1 - 0.2 / 0.5)

    def test_single_level_is_certain(self):
        self.assertEqual(score_confidence([1.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
