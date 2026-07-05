import unittest

from chaoslang.categorization.js import cluster_by_js, js_divergence


class JensenShannonClusteringTests(unittest.TestCase):
    def test_js_divergence_is_symmetric_and_zero_for_identical_histograms(self):
        a = {"left:a/right:b": 2, "left:c/right:d": 1}
        b = {"left:a/right:b": 2, "left:c/right:d": 1}
        c = {"other": 3}
        self.assertAlmostEqual(js_divergence(a, b), 0.0)
        self.assertAlmostEqual(js_divergence(a, c), js_divergence(c, a))
        self.assertGreater(js_divergence(a, c), 0.0)

    def test_cluster_by_js_is_deterministic_threshold_seam(self):
        histograms = {
            "x": {"a_b": 10, "a_c": 1},
            "y": {"a_b": 9, "a_c": 1},
            "z": {"q_r": 8},
        }
        clusters = cluster_by_js(histograms, threshold=0.01)
        self.assertEqual(clusters, (("x", "y"), ("z",)))


if __name__ == "__main__":
    unittest.main()
