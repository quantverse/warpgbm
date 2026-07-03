import unittest

import numpy as np


class DeferredLeafValueTests(unittest.TestCase):
    def test_leaf_values_stay_on_device_after_fit_and_predict_still_works(self):
        import torch
        from warpgbm import WarpGBM

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for WarpGBM training")

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
            ],
            dtype=np.int8,
        )
        y = np.array([0.0, 0.2, 1.0, 1.2, 2.0, 2.2, 3.0, 3.2], dtype=np.float32)
        era = np.zeros(X.shape[0], dtype=np.int32)

        model = WarpGBM(
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

        self.assertIsInstance(model.forest[0]["left"]["leaf_value"], torch.Tensor)
        preds = model.predict(X)
        self.assertEqual(preds.shape, (X.shape[0],))


if __name__ == "__main__":
    unittest.main()
