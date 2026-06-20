import unittest

import numpy as np


class FeatureMajorHistogramTests(unittest.TestCase):
    def test_feature_major_histogram_matches_cpu_reference(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for histogram extension tests")

        bin_rows = np.array(
            [
                [0, 1, 2],
                [3, 4, 0],
                [1, 2, 3],
                [4, 0, 1],
                [2, 3, 4],
                [0, 1, 2],
                [3, 4, 0],
                [1, 2, 3],
            ],
            dtype=np.int8,
        )
        residuals_np = np.array([0.5, -1.0, 2.0, 1.5, -0.25, 0.75, -1.25, 3.0], dtype=np.float32)
        era_np = np.array([0, 1, 0, 1, 2, 2, 0, 1], dtype=np.int32)
        sample_np = np.array([7, 0, 4, 2, 5, 1], dtype=np.int32)
        feature_np = np.array([2, 0], dtype=np.int32)
        num_bins = 5
        num_eras = 3

        bin_feature_major = torch.as_tensor(bin_rows.T.copy(), device="cuda")
        residuals = torch.as_tensor(residuals_np, device="cuda")
        era_indices = torch.as_tensor(era_np, device="cuda")
        sample_indices = torch.as_tensor(sample_np, device="cuda")
        feature_indices = torch.as_tensor(feature_np, device="cuda")
        grad_hist = torch.zeros((num_eras, len(feature_np), num_bins), device="cuda", dtype=torch.float32)
        hess_hist = torch.zeros_like(grad_hist)

        node_kernel.compute_histogram_feature_major(
            bin_feature_major,
            residuals,
            sample_indices,
            feature_indices,
            era_indices,
            grad_hist,
            hess_hist,
            num_bins,
            64,
            2,
        )
        torch.cuda.synchronize()

        expected_grad = np.zeros((num_eras, len(feature_np), num_bins), dtype=np.float32)
        expected_hess = np.zeros_like(expected_grad)
        for local_feature, feature in enumerate(feature_np):
            for sample in sample_np:
                era = era_np[sample]
                bin_idx = bin_rows[sample, feature]
                expected_grad[era, local_feature, bin_idx] += residuals_np[sample]
                expected_hess[era, local_feature, bin_idx] += 1.0

        np.testing.assert_allclose(grad_hist.cpu().numpy(), expected_grad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(hess_hist.cpu().numpy(), expected_hess, rtol=1e-6, atol=1e-6)

    def test_feature_tile_histogram_matches_cpu_reference(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for histogram extension tests")

        bin_rows = np.array(
            [
                [0, 1, 2, 3, 4, 0, 1, 2, 3],
                [3, 4, 0, 1, 2, 3, 4, 0, 1],
                [1, 2, 3, 4, 0, 1, 2, 3, 4],
                [4, 0, 1, 2, 3, 4, 0, 1, 2],
                [2, 3, 4, 0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 0, 1, 2, 3],
                [3, 4, 0, 1, 2, 3, 4, 0, 1],
                [1, 2, 3, 4, 0, 1, 2, 3, 4],
            ],
            dtype=np.int8,
        )
        residuals_np = np.array([0.5, -1.0, 2.0, 1.5, -0.25, 0.75, -1.25, 3.0], dtype=np.float32)
        era_np = np.array([0, 1, 0, 1, 2, 2, 0, 1], dtype=np.int32)
        sample_np = np.array([7, 0, 4, 2, 5, 1], dtype=np.int32)
        feature_np = np.array([8, 0, 6, 3, 1], dtype=np.int32)
        num_bins = 5
        num_eras = 3

        bin_indices = torch.as_tensor(bin_rows.copy(), device="cuda")
        residuals = torch.as_tensor(residuals_np, device="cuda")
        era_indices = torch.as_tensor(era_np, device="cuda")
        sample_indices = torch.as_tensor(sample_np, device="cuda")
        feature_indices = torch.as_tensor(feature_np, device="cuda")
        grad_hist = torch.zeros((num_eras, len(feature_np), num_bins), device="cuda", dtype=torch.float32)
        hess_hist = torch.zeros_like(grad_hist)

        node_kernel.compute_histogram_feature_tile(
            bin_indices,
            residuals,
            sample_indices,
            feature_indices,
            era_indices,
            grad_hist,
            hess_hist,
            num_bins,
            64,
            2,
        )
        torch.cuda.synchronize()

        expected_grad = np.zeros((num_eras, len(feature_np), num_bins), dtype=np.float32)
        expected_hess = np.zeros_like(expected_grad)
        for local_feature, feature in enumerate(feature_np):
            for sample in sample_np:
                era = era_np[sample]
                bin_idx = bin_rows[sample, feature]
                expected_grad[era, local_feature, bin_idx] += residuals_np[sample]
                expected_hess[era, local_feature, bin_idx] += 1.0

        np.testing.assert_allclose(grad_hist.cpu().numpy(), expected_grad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(hess_hist.cpu().numpy(), expected_hess, rtol=1e-6, atol=1e-6)

    def test_feature_tile_root_histogram_matches_cpu_reference(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for histogram extension tests")

        bin_rows = np.array(
            [
                [0, 1, 2, 3, 4, 0, 1, 2, 3],
                [3, 4, 0, 1, 2, 3, 4, 0, 1],
                [1, 2, 3, 4, 0, 1, 2, 3, 4],
                [4, 0, 1, 2, 3, 4, 0, 1, 2],
                [2, 3, 4, 0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 0, 1, 2, 3],
                [3, 4, 0, 1, 2, 3, 4, 0, 1],
                [1, 2, 3, 4, 0, 1, 2, 3, 4],
            ],
            dtype=np.int8,
        )
        residuals_np = np.array([0.5, -1.0, 2.0, 1.5, -0.25, 0.75, -1.25, 3.0], dtype=np.float32)
        era_np = np.array([0, 1, 0, 1, 2, 2, 0, 1], dtype=np.int32)
        num_bins = 5
        num_eras = 3

        bin_indices = torch.as_tensor(bin_rows.copy(), device="cuda")
        residuals = torch.as_tensor(residuals_np, device="cuda")
        era_indices = torch.as_tensor(era_np, device="cuda")
        grad_hist = torch.zeros((num_eras, bin_rows.shape[1], num_bins), device="cuda", dtype=torch.float32)
        hess_hist = torch.zeros_like(grad_hist)

        node_kernel.compute_histogram_feature_tile_root(
            bin_indices,
            residuals,
            era_indices,
            grad_hist,
            hess_hist,
            num_bins,
            64,
            2,
        )
        torch.cuda.synchronize()

        expected_grad = np.zeros((num_eras, bin_rows.shape[1], num_bins), dtype=np.float32)
        expected_hess = np.zeros_like(expected_grad)
        for sample in range(bin_rows.shape[0]):
            era = era_np[sample]
            for feature in range(bin_rows.shape[1]):
                bin_idx = bin_rows[sample, feature]
                expected_grad[era, feature, bin_idx] += residuals_np[sample]
                expected_hess[era, feature, bin_idx] += 1.0

        np.testing.assert_allclose(grad_hist.cpu().numpy(), expected_grad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(hess_hist.cpu().numpy(), expected_hess, rtol=1e-6, atol=1e-6)

    def test_feature_tile_grad_histogram_matches_cpu_reference(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for histogram extension tests")

        bin_rows = np.array(
            [
                [0, 1, 2, 3, 4, 0, 1, 2, 3],
                [3, 4, 0, 1, 2, 3, 4, 0, 1],
                [1, 2, 3, 4, 0, 1, 2, 3, 4],
                [4, 0, 1, 2, 3, 4, 0, 1, 2],
                [2, 3, 4, 0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 0, 1, 2, 3],
                [3, 4, 0, 1, 2, 3, 4, 0, 1],
                [1, 2, 3, 4, 0, 1, 2, 3, 4],
            ],
            dtype=np.int8,
        )
        residuals_np = np.array([0.5, -1.0, 2.0, 1.5, -0.25, 0.75, -1.25, 3.0], dtype=np.float32)
        era_np = np.array([0, 1, 0, 1, 2, 2, 0, 1], dtype=np.int32)
        sample_np = np.array([7, 0, 4, 2, 5, 1], dtype=np.int32)
        feature_np = np.array([8, 0, 6, 3, 1], dtype=np.int32)
        num_bins = 5
        num_eras = 3

        bin_indices = torch.as_tensor(bin_rows.copy(), device="cuda")
        residuals = torch.as_tensor(residuals_np, device="cuda")
        era_indices = torch.as_tensor(era_np, device="cuda")
        sample_indices = torch.as_tensor(sample_np, device="cuda")
        feature_indices = torch.as_tensor(feature_np, device="cuda")
        grad_hist = torch.zeros((num_eras, len(feature_np), num_bins), device="cuda", dtype=torch.float32)

        node_kernel.compute_histogram_feature_tile_grad(
            bin_indices,
            residuals,
            sample_indices,
            feature_indices,
            era_indices,
            grad_hist,
            num_bins,
            64,
            2,
        )
        torch.cuda.synchronize()

        expected_grad = np.zeros((num_eras, len(feature_np), num_bins), dtype=np.float32)
        for local_feature, feature in enumerate(feature_np):
            for sample in sample_np:
                era = era_np[sample]
                bin_idx = bin_rows[sample, feature]
                expected_grad[era, local_feature, bin_idx] += residuals_np[sample]

        np.testing.assert_allclose(grad_hist.cpu().numpy(), expected_grad, rtol=1e-6, atol=1e-6)

    def test_feature_tile_by_era_histogram_matches_cpu_reference(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for histogram extension tests")

        bin_rows = np.array(
            [
                [0, 1, 2, 3, 4],
                [3, 4, 0, 1, 2],
                [1, 2, 3, 4, 0],
                [4, 0, 1, 2, 3],
                [2, 3, 4, 0, 1],
                [0, 1, 2, 3, 4],
                [3, 4, 0, 1, 2],
                [1, 2, 3, 4, 0],
            ],
            dtype=np.int8,
        )
        residuals_np = np.array([0.5, -1.0, 2.0, 1.5, -0.25, 0.75, -1.25, 3.0], dtype=np.float32)
        era_np = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
        sample_np = np.array([0, 2, 3, 5, 6, 7], dtype=np.int32)
        feature_np = np.array([4, 0, 3], dtype=np.int32)
        era_boundaries = np.array([0, 3, 5, 8], dtype=np.int32)
        era_offsets_np = np.searchsorted(sample_np, era_boundaries).astype(np.int32)
        max_era_count = int(np.max(era_offsets_np[1:] - era_offsets_np[:-1]))
        num_bins = 5
        num_eras = 3

        bin_indices = torch.as_tensor(bin_rows.copy(), device="cuda")
        residuals = torch.as_tensor(residuals_np, device="cuda")
        sample_indices = torch.as_tensor(sample_np, device="cuda")
        feature_indices = torch.as_tensor(feature_np, device="cuda")
        era_offsets = torch.as_tensor(era_offsets_np, device="cuda")
        grad_hist = torch.zeros((num_eras, len(feature_np), num_bins), device="cuda", dtype=torch.float32)
        hess_hist = torch.zeros_like(grad_hist)

        node_kernel.compute_histogram_feature_tile_by_era(
            bin_indices,
            residuals,
            sample_indices,
            feature_indices,
            era_offsets,
            grad_hist,
            hess_hist,
            num_bins,
            max_era_count,
            64,
            2,
        )
        torch.cuda.synchronize()

        expected_grad = np.zeros((num_eras, len(feature_np), num_bins), dtype=np.float32)
        expected_hess = np.zeros_like(expected_grad)
        for local_feature, feature in enumerate(feature_np):
            for sample in sample_np:
                era = era_np[sample]
                bin_idx = bin_rows[sample, feature]
                expected_grad[era, local_feature, bin_idx] += residuals_np[sample]
                expected_hess[era, local_feature, bin_idx] += 1.0

        np.testing.assert_allclose(grad_hist.cpu().numpy(), expected_grad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(hess_hist.cpu().numpy(), expected_hess, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
