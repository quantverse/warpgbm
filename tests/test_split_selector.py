import unittest


class SplitSelectorTests(unittest.TestCase):
    def test_selects_highest_gain_among_max_agreement_splits(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for split selector tests")

        fused_gain = torch.tensor(
            [[0.1, 2.0, 3.0], [4.0, 5.0, 6.0]],
            device="cuda",
            dtype=torch.float32,
        )
        fused_agreement = torch.tensor(
            [[0.9, 0.4, 0.9], [0.9, 0.8, 0.1]],
            device="cuda",
            dtype=torch.float32,
        )
        best_idx = torch.empty(1, device="cuda", dtype=torch.int32)

        node_kernel.select_split_fused(fused_gain, fused_agreement, 0.0, best_idx, 128)
        torch.cuda.synchronize()

        self.assertEqual(int(best_idx.cpu().item()), 3)

    def test_returns_negative_one_when_no_gain_passes_threshold(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for split selector tests")

        fused_gain = torch.tensor([[0.1, 0.2]], device="cuda", dtype=torch.float32)
        fused_agreement = torch.tensor([[1.0, 1.0]], device="cuda", dtype=torch.float32)
        best_idx = torch.empty(1, device="cuda", dtype=torch.int32)

        node_kernel.select_split_fused(fused_gain, fused_agreement, 1.0, best_idx, 128)
        torch.cuda.synchronize()

        self.assertEqual(int(best_idx.cpu().item()), -1)

    def test_global_max_agreement_must_pass_gain_threshold(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for split selector tests")

        fused_gain = torch.tensor([[0.1, 2.0]], device="cuda", dtype=torch.float32)
        fused_agreement = torch.tensor([[1.0, 0.9]], device="cuda", dtype=torch.float32)
        best_idx = torch.empty(1, device="cuda", dtype=torch.int32)

        node_kernel.select_split_fused(fused_gain, fused_agreement, 1.0, best_idx, 128)
        torch.cuda.synchronize()

        self.assertEqual(int(best_idx.cpu().item()), -1)

    def test_non_power_of_two_thread_reduction_keeps_all_candidates(self):
        import torch
        from warpgbm.cuda import node_kernel

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for split selector tests")

        fused_gain = torch.ones((1, 96), device="cuda", dtype=torch.float32)
        fused_agreement = torch.zeros((1, 96), device="cuda", dtype=torch.float32)
        fused_agreement[0, 2] = 10.0
        fused_agreement[0, 0] = 1.0
        best_idx = torch.empty(1, device="cuda", dtype=torch.int32)

        node_kernel.select_split_fused(fused_gain, fused_agreement, 0.0, best_idx, 96)
        torch.cuda.synchronize()

        self.assertEqual(int(best_idx.cpu().item()), 2)


if __name__ == "__main__":
    unittest.main()
