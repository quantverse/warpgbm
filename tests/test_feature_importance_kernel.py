import unittest

import numpy as np


class FeatureImportanceKernelTests(unittest.TestCase):
    def test_accumulates_selected_split_gain_per_era(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for feature importance kernel tests")

        gradient_histogram = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, -0.5, 2.0]],
                [[2.0, -1.0, 0.5, 1.0], [1.0, -2.0, 3.0, 0.5]],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        hessian_histogram = torch.tensor(
            [
                [[3.0, 2.0, 2.0, 1.0], [2.0, 2.0, 3.0, 4.0]],
                [[4.0, 3.0, 2.0, 1.0], [1.5, 2.5, 3.5, 4.5]],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        importance = torch.ones((2, 5), device="cuda", dtype=torch.float32)
        local_feature = 1
        best_bin = 1
        global_feature = 3
        min_child_weight = 1.0
        eps = 1e-6

        node_kernel.accumulate_feature_importance(
            gradient_histogram,
            hessian_histogram,
            local_feature,
            best_bin,
            global_feature,
            min_child_weight,
            eps,
            importance,
            128,
        )
        torch.cuda.synchronize()

        g = gradient_histogram.cpu().numpy()
        h = hessian_histogram.cpu().numpy()
        expected = np.ones((2, 5), dtype=np.float32)
        for era in range(g.shape[0]):
            g_hist = g[era, local_feature]
            h_hist = h[era, local_feature]
            g_l = g_hist[: best_bin + 1].sum()
            h_l = h_hist[: best_bin + 1].sum()
            g_total = g_hist.sum()
            h_total = h_hist.sum()
            g_r = g_total - g_l
            h_r = h_total - h_l
            gain = 0.0
            if h_l >= min_child_weight and h_r >= min_child_weight:
                gain = (
                    (g_l * g_l) / (h_l + eps)
                    + (g_r * g_r) / (h_r + eps)
                    - (g_total * g_total) / (h_total + eps)
                )
            expected[era, global_feature] += gain

        np.testing.assert_allclose(importance.cpu().numpy(), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
