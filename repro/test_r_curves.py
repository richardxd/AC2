import copy
import unittest
from r_curves import validation


class CurveValidationTests(unittest.TestCase):
    def rows(self):
        return [{"input": group, "acc": value, "judge_parse_failed": 0,
                 "judge_http_error": 0, "judge_truncated": 0}
                for group, values in [('a', [0, 0, 1, 1]), ('b', [0, 1, 1, 1])] for value in values]

    def test_matched_group_mean_and_count(self):
        result = validation(self.rows(), 4, 2, .625, ["a", "b"])
        self.assertEqual(result["mean_score"], .625)
        self.assertEqual(sorted(result["problem_means"].values()), [.5, .75])
        self.assertEqual(result["nonzero_fraction"], .625)

    def test_missing_unbalanced_failed_and_log_mismatch_rejected(self):
        rows = self.rows()
        unbalanced = copy.deepcopy(rows)
        unbalanced[0]["input"] = 'b'
        failed = copy.deepcopy(rows)
        failed[0]["judge_http_error"] = 1
        for bad, mean in [(rows[:-1], .625), (unbalanced, .625), (failed, .625), (rows, .7)]:
            with self.assertRaises(AssertionError):
                validation(bad, 4, 2, mean)
        with self.assertRaises(AssertionError):
            validation(rows, 4, 2, .625, ["wrong", "problems"])


if __name__ == '__main__':
    unittest.main()
