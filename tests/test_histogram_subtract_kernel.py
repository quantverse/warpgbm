import unittest

import numpy as np


class HistogramSubtractKernelTests(unittest.TestCase):
    def test_subtracts_gradient_and_hessian_histograms_in_place(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for histogram subtraction kernel tests")

        parent_grad = torch.tensor(
            [[[5.0, 4.0, 3.0], [2.0, 1.0, 0.0]], [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]],
            device="cuda",
            dtype=torch.float32,
        )
        parent_hess = torch.tensor(
            [[[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]],
            device="cuda",
            dtype=torch.float32,
        )
        child_grad = torch.tensor(
            [[[1.0, 1.5, 2.0], [0.5, 0.25, 0.0]], [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]],
            device="cuda",
            dtype=torch.float32,
        )
        child_hess = torch.tensor(
            [[[2.0, 1.0, 0.5], [1.0, 0.5, 0.25]], [[0.5, 0.5, 1.0], [1.5, 2.0, 2.5]]],
            device="cuda",
            dtype=torch.float32,
        )
        expected_grad = parent_grad.cpu().numpy() - child_grad.cpu().numpy()
        expected_hess = parent_hess.cpu().numpy() - child_hess.cpu().numpy()

        node_kernel.subtract_histograms_inplace(parent_grad, parent_hess, child_grad, child_hess, 128)
        torch.cuda.synchronize()

        np.testing.assert_allclose(parent_grad.cpu().numpy(), expected_grad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(parent_hess.cpu().numpy(), expected_hess, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
