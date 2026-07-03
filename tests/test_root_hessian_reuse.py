import unittest

import numpy as np


class RootHessianReuseTests(unittest.TestCase):
    def test_sorted_era_regression_uses_grad_only_root_after_first_tree(self):
        import torch
        from warpgbm import WarpGBM

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for WarpGBM training")

        class CountingWarpGBM(WarpGBM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.root_grad_only_calls = 0

            def compute_root_gradient_histogram_by_era(self):
                self.root_grad_only_calls += 1
                return super().compute_root_gradient_histogram_by_era()

        X = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [0, 2],
                [1, 3],
                [2, 0],
                [3, 1],
                [0, 3],
                [1, 0],
                [2, 1],
                [3, 2],
            ],
            dtype=np.int8,
        )
        y = np.linspace(0.0, 1.0, X.shape[0], dtype=np.float32)
        era = np.repeat(np.arange(3, dtype=np.int32), 4)

        model = CountingWarpGBM(
            objective="regression",
            num_bins=4,
            max_depth=2,
            learning_rate=0.005,
            n_estimators=3,
            min_child_weight=1,
            colsample_bytree=1.0,
            threads_per_block=64,
            rows_per_thread=2,
            device="cuda",
        )
        model.fit(X, y, era_id=era)

        self.assertEqual(model.root_grad_only_calls, 2)


if __name__ == "__main__":
    unittest.main()
