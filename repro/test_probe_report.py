import unittest
from probe_report import analyze, stats


class ProbeReportTests(unittest.TestCase):
    def fixture(self):
        groups = [{"qid": "q", "completions": [{"exceeds_g": x} for x in [True, True, False, False]]}]
        q = [{"qid": "q", "completion_index": i, "q": v} for i, v in [(-1, .4), (0, .2), (1, .8)]]
        j = [{"qid": "q", "completion_index": i, "score": v, "judge_http_error": 0,
              "judge_parse_failed": 0, "judge_truncated": 0} for i, v in enumerate([0, 1, .5, .5])]
        return groups, q, j

    def test_four_sample_arithmetic_and_advantages(self):
        result = analyze(*self.fixture(), n=4)
        self.assertEqual(result["groups_complete_valid"], 1)
        self.assertEqual(result["pairs"]["matched_cut_vs_reward"], [[.5, .5]])
        self.assertEqual(result["pairs"]["individual_cut_vs_reward"], [[0, .2], [1, .8]])
        self.assertEqual(result["pairs"]["advantages"], [[-.5, -.3], [.5, .30000000000000004], [0, 0], [0, 0]])
        self.assertAlmostEqual(result["statistics"]["advantages"]["mae"], .1)
        self.assertIsNone(result["statistics"]["prefix_vs_reward"]["pearson"])

    def test_invalid_q_excludes_whole_group(self):
        g, q, j = self.fixture()
        q[1]["q"] = None
        result = analyze(g, q, j, 4)
        self.assertEqual(result["groups_complete_valid"], 0)
        self.assertEqual(result["excluded_invalid_q_groups"], [{"qid": "q", "invalid_q_indices": [0]}])
        self.assertEqual(result["statistics"]["advantages"]["n"], 0)
        self.assertEqual(result["pairs"]["hybrid_all_vs_reward"], [])

    def test_missing_duplicate_and_failed_grades_rejected(self):
        g, q, j = self.fixture()
        for bad in (j[:-1], j + [j[0]], [dict(j[0], judge_truncated=1)] + j[1:]):
            with self.assertRaises(AssertionError):
                analyze(g, q, bad, 4)

    def test_no_substitution_is_separate_and_constant_is_undefined(self):
        g, q, j = self.fixture()
        for c in g[0]["completions"]:
            c["exceeds_g"] = False
        result = analyze(g, q[:1], j, 4)
        self.assertEqual(result["no_substitution_groups"], ["q"])
        self.assertEqual(result["pairs"]["hybrid_all_vs_reward"], [[.5, .5]])
        self.assertEqual(result["statistics"]["advantages"]["n"], 0)
        self.assertIsNone(stats([[0, 1]] * 4)["pearson"])
        self.assertIsNone(stats([[.2, .2]] * 3)["pearson"])
        self.assertIsNone(stats([[.2, 0], [.2, 1], [.2, .5]])["pearson"])
        self.assertAlmostEqual(stats([[.2, .2], [.4, .4], [.8, .8]])["pearson"], 1)


if __name__ == "__main__":
    unittest.main()
