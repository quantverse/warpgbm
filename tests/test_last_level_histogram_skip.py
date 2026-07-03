import unittest

import numpy as np


class LastLevelHistogramSkipTests(unittest.TestCase):
    def test_max_depth_one_does_not_compute_unused_child_histograms(self):
        import torch
        from warpgbm import WarpGBM

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for WarpGBM histogram counting")

        class CountingWarpGBM(WarpGBM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.histogram_calls = 0

            def compute_histograms(self, sample_indices, feature_indices):
                self.histogram_calls += 1
                return super().compute_histograms(sample_indices, feature_indices)

        X = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [2, 0],
                [2, 1],
                [3, 0],
                [3, 1],
            ],
            dtype=np.int8,
        )
        y = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1], dtype=np.float32)
        era = np.zeros(X.shape[0], dtype=np.int32)

        model = CountingWarpGBM(
            objective="regression",
            num_bins=4,
            max_depth=1,
            learning_rate=0.005,
            n_estimators=1,
            min_child_weight=1,
            threads_per_block=64,
            rows_per_thread=2,
            device="cuda",
        )
        model.fit(X, y, era_id=era)

        self.assertEqual(model.histogram_calls, 1)

    def test_too_small_regression_node_skips_split_search(self):
        import torch
        from warpgbm import WarpGBM

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for WarpGBM split counting")

        class CountingWarpGBM(WarpGBM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.split_calls = 0

            def find_best_split(self, gradient_histogram, hessian_histogram):
                self.split_calls += 1
                return super().find_best_split(gradient_histogram, hessian_histogram)

        X = np.array(
            [
                [0],
                [0],
                [1],
                [1],
                [2],
                [2],
                [3],
                [3],
                [0],
                [1],
            ],
            dtype=np.int8,
        )
        y = np.arange(X.shape[0], dtype=np.float32)
        era = np.zeros(X.shape[0], dtype=np.int32)

        model = CountingWarpGBM(
            objective="regression",
            num_bins=4,
            max_depth=3,
            learning_rate=0.005,
            n_estimators=1,
            min_child_weight=6,
            threads_per_block=64,
            rows_per_thread=2,
            device="cuda",
        )
        model.fit(X, y, era_id=era)

        self.assertEqual(model.split_calls, 0)


if __name__ == "__main__":
    unittest.main()
