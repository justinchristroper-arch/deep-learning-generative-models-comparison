import unittest

import numpy as np
import torch

from evaluate_anomaly import comparison, defects


class AnomalyTests(unittest.TestCase):
    def test_perfect_and_tied_scores(self):
        controls = np.arange(20, dtype=float)
        perfect = comparison(controls, controls + 20, controls)
        for key in ("auroc", "auprc_average_precision", "test_tpr_at_test_fpr_at_most_5pct"):
            self.assertEqual(perfect[key], 1.0)
        self.assertEqual(perfect["validation_fpr_at_threshold"], 0.05)
        ties = comparison(np.zeros(20), np.zeros(20), np.zeros(20))
        self.assertEqual(ties["auroc"], 0.5)
        self.assertEqual(ties["auprc_average_precision"], 0.5)
        self.assertEqual(ties["test_tpr_at_test_fpr_at_most_5pct"], 0.0)

    def test_defects_are_seeded_local_and_preserve_clean_inputs(self):
        clean = torch.full((12, 1, 28, 28), 0.6)
        a, labels = defects(clean, 42)
        b, repeated = defects(clean, 42)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(labels, repeated)
        self.assertEqual(torch.bincount(labels).tolist(), [4, 4, 4])
        self.assertTrue(torch.all(clean == 0.6))
        changed = (a != clean).sum(dim=(1, 2, 3))
        self.assertTrue(torch.all((changed > 0) & (changed <= 49)))
