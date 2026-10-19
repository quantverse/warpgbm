import unittest

import numpy as np


def _cpu_split_reference(g, h, min_child_weight, eps):
    """Float64 reference for the fused era reduction (mean gain, |mean direction|)."""
    expected_gain = np.zeros((g.shape[1], g.shape[2] - 1), dtype=np.float64)
    expected_agreement = np.zeros_like(expected_gain)
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
    return expected_gain, np.abs(expected_agreement)


def _run_fused_split(g, h, threads, min_split_gain=0.0, min_child_weight=1.0, eps=1e-6):
    import torch
    from warpgbm.cuda import node_kernel

    fused_gain = torch.zeros((g.shape[1], g.shape[2] - 1), device="cuda", dtype=torch.float32)
    fused_agreement = torch.zeros_like(fused_gain)
    node_kernel.compute_split_fused(
        g,
        h,
        min_split_gain,
        min_child_weight,
        eps,
        fused_gain,
        fused_agreement,
        threads,
    )
    torch.cuda.synchronize()
    return fused_gain, fused_agreement


class SplitFusedKernelTests(unittest.TestCase):
    def _require_cuda(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for fused split kernel tests")

    def test_fused_split_matches_cpu_reference(self):
        import torch

        self._require_cuda()

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
        min_child_weight = 1.0
        eps = 1e-6

        fused_gain, fused_agreement = _run_fused_split(
            gradient_histogram, hessian_histogram, 128, min_child_weight=min_child_weight, eps=eps
        )

        g = gradient_histogram.cpu().numpy()
        h = hessian_histogram.cpu().numpy()
        expected_gain, expected_agreement = _cpu_split_reference(g, h, min_child_weight, eps)

        np.testing.assert_allclose(fused_gain.cpu().numpy(), expected_gain, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            fused_agreement.cpu().numpy(), expected_agreement, rtol=1e-6, atol=1e-6
        )

    def test_fused_split_supports_non_power_of_two_threads(self):
        import torch

        self._require_cuda()

        num_eras = 96
        gradient_histogram = torch.zeros((num_eras, 1, 2), device="cuda", dtype=torch.float32)
        hessian_histogram = torch.ones_like(gradient_histogram)
        gradient_histogram[:, :, 0] = 1.0

        fused_gain, fused_agreement = _run_fused_split(gradient_histogram, hessian_histogram, 96)

        self.assertAlmostEqual(float(fused_gain.cpu()[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(fused_agreement.cpu()[0, 0]), 1.0, places=5)

    def test_fused_split_shrinks_threads_when_shared_memory_would_overflow(self):
        """threads=1024 with B=9 needs 2*(B-1)*1024 floats of shared memory — more
        than the dynamic limit. The kernel must shrink the block and still compute
        correct gains (it previously fell back to the nondeterministic atomic path,
        and this test asserted only an output shape on zero inputs)."""
        import torch

        self._require_cuda()

        rng = np.random.default_rng(7)
        g = torch.tensor(
            rng.standard_normal((5, 3, 9)) * 2.0, device="cuda", dtype=torch.float32
        )
        h = torch.tensor(
            rng.uniform(0.5, 4.0, (5, 3, 9)), device="cuda", dtype=torch.float32
        )
        min_child_weight = 1.0
        eps = 1e-6

        fused_gain, fused_agreement = _run_fused_split(
            g, h, 1024, min_child_weight=min_child_weight, eps=eps
        )

        expected_gain, expected_agreement = _cpu_split_reference(
            g.cpu().numpy(), h.cpu().numpy(), min_child_weight, eps
        )
        np.testing.assert_allclose(
            fused_gain.cpu().numpy(), expected_gain, rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            fused_agreement.cpu().numpy(), expected_agreement, rtol=1e-4, atol=1e-4
        )

    def test_fused_split_is_bitwise_deterministic_and_breaks_ties_lowest_index(self):
        """Regression pin: float atomicAdd accumulation made fused_gain vary
        run-to-run, flipping the winner between tied features (issue F1)."""
        import torch
        from warpgbm.cuda import node_kernel

        self._require_cuda()

        eras, features, bins = 33, 4, 17
        # Features 0 and 3 carry zero gradients: their gain is 0 (excluded by the
        # gain gate) while their agreement is also 1.0. Features 1 and 2 are
        # identical in every era, so both have agreement exactly 1.0 and equal
        # gains: the winner must come from that pair, ties break to the lowest
        # flat index (feature 1).
        g_np = np.zeros((eras, features, bins), dtype=np.float32)
        h_np = np.ones((eras, features, bins), dtype=np.float32)
        base_g = np.zeros(bins, dtype=np.float32)
        base_g[0], base_g[1], base_g[2], base_g[3] = 8.0, -8.0, 0.5, 0.25
        g_np[:, 1, :] = base_g
        g_np[:, 2, :] = base_g
        g = torch.tensor(g_np, device="cuda")
        h = torch.tensor(h_np, device="cuda")

        results = []
        for _ in range(20):
            fused_gain, fused_agreement = _run_fused_split(g, h, 64)
            best_idx = torch.empty(1, dtype=torch.int32, device="cuda")
            node_kernel.select_split_fused(fused_gain, fused_agreement, 0.0, best_idx, 64)
            results.append(
                (fused_gain.cpu().clone(), fused_agreement.cpu().clone(), int(best_idx.item()))
            )

        ref_gain, ref_agree, ref_idx = results[0]
        for gain, agree, idx in results[1:]:
            self.assertTrue(torch.equal(gain, ref_gain), "fused_gain varied across identical runs")
            self.assertTrue(torch.equal(agree, ref_agree), "fused_agreement varied across runs")
            self.assertEqual(idx, ref_idx, "split selection flipped across identical runs")

        # The winner must be the tied pair and ties break to the lowest flat
        # index (feature 1).
        split_bins = bins - 1
        self.assertEqual(ref_idx // split_bins, 1)
        self.assertGreater(ref_gain[1, ref_idx % split_bins].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
