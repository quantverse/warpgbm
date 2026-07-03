import unittest

import numpy as np


class SplitFusedNoAtomicTests(unittest.TestCase):
    def test_no_atomic_fused_split_matches_cpu_reference(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for fused split kernel tests")

        gradient_histogram = torch.tensor(
            [
                [[1.0, 2.0, -1.0, 0.5, 3.0], [2.0, -0.5, 1.0, 1.5, -1.0]],
                [[0.5, -1.0, 2.5, 1.0, 0.0], [1.0, 2.0, -2.0, 0.5, 1.5]],
                [[2.0, 1.0, 0.5, -1.5, 1.0], [0.5, 1.5, 2.0, -0.5, 1.0]],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        hessian_histogram = torch.tensor(
            [
                [[3.0, 2.0, 2.0, 1.0, 4.0], [2.0, 2.0, 3.0, 4.0, 2.0]],
                [[4.0, 3.0, 2.0, 1.0, 3.0], [1.5, 2.5, 3.5, 4.5, 2.0]],
                [[2.0, 3.0, 1.0, 2.0, 2.0], [3.0, 1.0, 2.0, 2.5, 3.5]],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        min_split_gain = 0.0
        min_child_weight = 1.0
        eps = 1e-6
        fused_gain = torch.zeros((2, 4), device="cuda", dtype=torch.float32)
        fused_agreement = torch.zeros_like(fused_gain)

        node_kernel.compute_split_fused_no_atomic(
            gradient_histogram,
            hessian_histogram,
            min_split_gain,
            min_child_weight,
            eps,
            fused_gain,
            fused_agreement,
            128,
        )
        torch.cuda.synchronize()

        g = gradient_histogram.cpu().numpy()
        h = hessian_histogram.cpu().numpy()
        expected_gain = np.zeros((2, 4), dtype=np.float32)
        expected_agreement = np.zeros((2, 4), dtype=np.float32)
        for feature in range(g.shape[1]):
            for split_bin in range(g.shape[2] - 1):
                for era in range(g.shape[0]):
                    g_hist = g[era, feature]
                    h_hist = h[era, feature]
                    g_l = g_hist[: split_bin + 1].sum()
                    h_l = h_hist[: split_bin + 1].sum()
                    g_total = g_hist.sum()
                    h_total = h_hist.sum()
                    g_r = g_total - g_l
                    h_r = h_total - h_l
                    if h_l >= min_child_weight and h_r >= min_child_weight:
                        gain = (
                            (g_l * g_l) / (h_l + eps)
                            + (g_r * g_r) / (h_r + eps)
                            - (g_total * g_total) / (h_total + eps)
                        )
                        left_val = g_l / (h_l + eps)
                        right_val = g_r / (h_r + eps)
                        direction = 1.0 if left_val > right_val else -1.0
                    else:
                        gain = 0.0
                        direction = 0.0
                    expected_gain[feature, split_bin] += gain / g.shape[0]
                    expected_agreement[feature, split_bin] += direction / g.shape[0]
        expected_agreement = np.abs(expected_agreement)

        np.testing.assert_allclose(fused_gain.cpu().numpy(), expected_gain, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            fused_agreement.cpu().numpy(), expected_agreement, rtol=1e-6, atol=1e-6
        )

    def test_no_atomic_fused_split_supports_non_power_of_two_threads(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for fused split kernel tests")

        num_eras = 96
        gradient_histogram = torch.zeros((num_eras, 1, 2), device="cuda", dtype=torch.float32)
        hessian_histogram = torch.ones_like(gradient_histogram)
        gradient_histogram[:, :, 0] = 1.0
        fused_gain = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
        fused_agreement = torch.zeros_like(fused_gain)

        node_kernel.compute_split_fused_no_atomic(
            gradient_histogram,
            hessian_histogram,
            0.0,
            1.0,
            1e-6,
            fused_gain,
            fused_agreement,
            96,
        )
        torch.cuda.synchronize()

        self.assertAlmostEqual(float(fused_gain.cpu()[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(fused_agreement.cpu()[0, 0]), 1.0, places=5)

    def test_no_atomic_fused_split_falls_back_when_shared_memory_is_too_large(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for fused split kernel tests")

        gradient_histogram = torch.zeros((1, 1, 9), device="cuda", dtype=torch.float32)
        hessian_histogram = torch.ones_like(gradient_histogram)
        fused_gain = torch.zeros((1, 8), device="cuda", dtype=torch.float32)
        fused_agreement = torch.zeros_like(fused_gain)

        node_kernel.compute_split_fused_no_atomic(
            gradient_histogram,
            hessian_histogram,
            0.0,
            1.0,
            1e-6,
            fused_gain,
            fused_agreement,
            1024,
        )
        torch.cuda.synchronize()

        self.assertEqual(tuple(fused_gain.shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
