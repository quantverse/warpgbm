import os
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class CudaRegressionTests(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for WarpGBM CUDA regression tests")

    def test_feature_tile_histogram_supports_num_bins_12(self):
        import torch
        from warpgbm.cuda import node_kernel

        num_bins = 12
        bin_indices = torch.zeros((1, 1), device="cuda", dtype=torch.int8)
        residuals = torch.ones(1, device="cuda", dtype=torch.float32)
        sample_indices = torch.arange(1, device="cuda", dtype=torch.int32)
        feature_indices = torch.arange(1, device="cuda", dtype=torch.int32)
        era_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
        grad_hist = torch.zeros((1, 1, num_bins), device="cuda", dtype=torch.float32)
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
            56,
        )
        torch.cuda.synchronize()

        self.assertEqual(float(grad_hist.sum().item()), 1.0)
        self.assertEqual(float(hess_hist.sum().item()), 1.0)

    def test_root_feature_tile_histogram_supports_num_bins_12(self):
        import torch
        from warpgbm.cuda import node_kernel

        num_bins = 12
        bin_indices = torch.zeros((1, 1), device="cuda", dtype=torch.int8)
        residuals = torch.ones(1, device="cuda", dtype=torch.float32)
        era_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
        grad_hist = torch.zeros((1, 1, num_bins), device="cuda", dtype=torch.float32)
        hess_hist = torch.zeros_like(grad_hist)

        node_kernel.compute_histogram_feature_tile_root(
            bin_indices,
            residuals,
            era_indices,
            grad_hist,
            hess_hist,
            num_bins,
            64,
            56,
        )
        torch.cuda.synchronize()

        self.assertEqual(float(grad_hist.sum().item()), 1.0)
        self.assertEqual(float(hess_hist.sum().item()), 1.0)

    def test_grad_only_feature_tile_histogram_supports_num_bins_24(self):
        import torch
        from warpgbm.cuda import node_kernel

        num_bins = 24
        bin_indices = torch.zeros((1, 1), device="cuda", dtype=torch.int8)
        residuals = torch.ones(1, device="cuda", dtype=torch.float32)
        sample_indices = torch.arange(1, device="cuda", dtype=torch.int32)
        feature_indices = torch.arange(1, device="cuda", dtype=torch.int32)
        era_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
        grad_hist = torch.zeros((1, 1, num_bins), device="cuda", dtype=torch.float32)

        node_kernel.compute_histogram_feature_tile_grad(
            bin_indices,
            residuals,
            sample_indices,
            feature_indices,
            era_indices,
            grad_hist,
            num_bins,
            64,
            56,
        )
        torch.cuda.synchronize()

        self.assertEqual(float(grad_hist.sum().item()), 1.0)

    def test_feature_tile_histogram_supports_large_row_tile_grid(self):
        import torch
        from warpgbm.cuda import node_kernel

        rows_per_block = (64 // 32) * 56
        num_rows = 65535 * rows_per_block + 1
        num_bins = 4
        bin_indices = torch.zeros((num_rows, 1), device="cuda", dtype=torch.int8)
        residuals = torch.ones(num_rows, device="cuda", dtype=torch.float32)
        sample_indices = torch.arange(num_rows, device="cuda", dtype=torch.int32)
        feature_indices = torch.arange(1, device="cuda", dtype=torch.int32)
        era_indices = torch.zeros(num_rows, device="cuda", dtype=torch.int32)
        grad_hist = torch.zeros((1, 1, num_bins), device="cuda", dtype=torch.float32)
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
            56,
        )
        torch.cuda.synchronize()

        self.assertEqual(float(grad_hist.sum().item()), float(num_rows))
        self.assertEqual(float(hess_hist.sum().item()), float(num_rows))

    def test_classification_refit_resets_sorted_era_boundaries(self):
        import torch
        from warpgbm import WarpGBM

        class BoundaryProbe(WarpGBM):
            def grow_forest_multiclass(self):
                self.stop = False
                self.feat_indices_tree = self.feature_indices
                self.residual = torch.zeros(
                    self.num_samples, device=self.device, dtype=torch.float32
                )
                self.compute_histograms(self.root_node_indices, self.feature_indices)

        X_reg = np.array([[0], [1], [2], [3], [0], [1], [2], [3]], dtype=np.int8)
        y_reg = np.linspace(0.0, 1.0, X_reg.shape[0], dtype=np.float32)
        era_reg = np.repeat(np.arange(2, dtype=np.int32), 4)

        model = BoundaryProbe(
            objective="regression",
            num_bins=4,
            max_depth=1,
            n_estimators=1,
            min_child_weight=1,
            threads_per_block=64,
            rows_per_thread=2,
            device="cuda",
        )
        model.fit(X_reg, y_reg, era_id=era_reg)

        model.objective = "binary"
        X_cls = np.zeros((4000, 1), dtype=np.int8)
        y_cls = np.array([0, 1] * 2000, dtype=np.int32)
        era_cls = np.zeros(4000, dtype=np.int32)
        model.fit(X_cls, y_cls, era_id=era_cls)

        self.assertEqual(model.era_row_boundaries.detach().cpu().tolist(), [0, 4000])
        self.assertEqual(model.num_eras, 1)

    def test_warm_start_feature_importance_buffer_resizes_for_more_eras(self):
        from warpgbm import WarpGBM

        X = np.array([[0], [1], [2], [3], [0], [1], [2], [3]], dtype=np.int8)
        y = np.array([0.0, 0.2, 1.0, 1.2, 0.1, 0.3, 1.1, 1.3], dtype=np.float32)

        model = WarpGBM(
            objective="regression",
            num_bins=4,
            max_depth=1,
            n_estimators=1,
            min_child_weight=1,
            threads_per_block=64,
            rows_per_thread=2,
            warm_start=True,
            device="cuda",
        )
        model.fit(X, y, era_id=np.zeros(X.shape[0], dtype=np.int32))
        model.n_estimators = 2
        model.fit(X, y, era_id=np.repeat(np.arange(2, dtype=np.int32), 4))

        self.assertEqual(model.per_era_feature_importance_gpu.shape, (2, 1))
        self.assertEqual(model.per_era_feature_importance_.shape, (2, 1))

    def test_saved_model_state_can_be_unpickled_without_visible_cuda(self):
        from warpgbm import WarpGBM

        X = np.array([[0], [0], [1], [1]], dtype=np.int8)
        y = np.array([0.0, 0.1, 1.0, 1.1], dtype=np.float32)
        era = np.zeros(X.shape[0], dtype=np.int32)

        model = WarpGBM(
            objective="regression",
            num_bins=2,
            max_depth=1,
            n_estimators=1,
            min_child_weight=1,
            threads_per_block=64,
            rows_per_thread=2,
            device="cuda",
        )
        model.fit(X, y, era_id=era)

        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        try:
            model.save_model(path)
            code = (
                "import pickle, sys, torch\n"
                "print(torch.cuda.is_available())\n"
                "with open(sys.argv[1], 'rb') as f:\n"
                "    pickle.load(f)\n"
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            result = subprocess.run(
                [sys.executable, "-c", code, path],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("False", result.stdout)
            with open(path, "rb") as f:
                state = pickle.load(f)
            leaf_values = []

            def collect_leaf_values(node):
                if isinstance(node, list):
                    for child in node:
                        collect_leaf_values(child)
                elif "leaf_value" in node:
                    leaf_values.append(node["leaf_value"])
                else:
                    collect_leaf_values(node["left"])
                    collect_leaf_values(node["right"])

            collect_leaf_values(state["forest"])
            self.assertGreater(len(leaf_values), 0)
            self.assertTrue(all(isinstance(value, float) for value in leaf_values))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
