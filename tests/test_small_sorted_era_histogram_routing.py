import unittest

import numpy as np


class SmallSortedEraHistogramRoutingTests(unittest.TestCase):
    def test_small_sorted_era_nodes_use_generic_histogram_path(self):
        import torch
        from warpgbm import WarpGBM

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for WarpGBM training")

        class CountingWarpGBM(WarpGBM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.by_era_calls = 0

            def compute_histograms_by_era(self, sample_indices, feature_indices):
                self.by_era_calls += 1
                return super().compute_histograms_by_era(sample_indices, feature_indices)

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
                [0, 2],
                [1, 2],
                [2, 2],
                [3, 2],
            ],
            dtype=np.int8,
        )
        y = np.linspace(0.0, 1.0, X.shape[0], dtype=np.float32)
        era = np.repeat(np.arange(3, dtype=np.int32), 4)

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

        self.assertEqual(model.by_era_calls, 0)

    def test_by_era_histograms_reject_unsorted_node_subsets(self):
        import torch
        from warpgbm import WarpGBM

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for WarpGBM training")

        model = WarpGBM(
            objective="regression",
            num_bins=4,
            max_depth=1,
            n_estimators=1,
            device="cuda",
        )

        sorted_indices = torch.arange(8, dtype=torch.int32, device="cuda")
        model._require_sorted_samples(sorted_indices)  # must not raise

        # The by-era kernels segment rows with searchsorted offsets; an unsorted
        # node subset would silently mis-bin rows without this guard.
        unsorted_indices = torch.tensor([0, 5, 2, 7], dtype=torch.int32, device="cuda")
        with self.assertRaises(RuntimeError):
            model.compute_histograms_by_era(
                unsorted_indices, torch.arange(2, dtype=torch.int32, device="cuda")
            )


if __name__ == "__main__":
    unittest.main()
