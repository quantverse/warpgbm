"""Equivalence audit: era-splitting behavior, optimized branch vs pre-optimization logic.

The pre-optimization (main) multi-era split path was:

    per_era_gain, per_era_direction = compute_split(...)        # CUDA, per era
    agreement = per_era_direction.mean(dim=0).abs()             # torch
    criterion = per_era_gain.mean(dim=0)                        # torch
    mask = (agreement == agreement.max()) & (criterion > min_split_gain)
    best = argmax(criterion masked)                             # torch

The optimized path fuses the era reduction and the selection into CUDA kernels
(compute_split_fused + select_split_fused, deterministic for any num_bins).
The old `compute_split` kernel is still compiled into the extension and is
byte-identical to main, so the legacy pipeline can be reconstructed exactly
and compared in-process.

Likewise the old `compute_histogram3` kernel (mathematically unchanged on this
branch: only shared-memory era tiling was added) lets us rebuild the legacy
histogram path and run a full legacy-vs-optimized training comparison.
"""

import unittest

import numpy as np


def _cuda_or_skip(test):
    import torch

    if not torch.cuda.is_available():
        test.skipTest("CUDA is required for era split equivalence tests")


def legacy_find_best_split(grad_hist, hess_hist, min_split_gain, min_child_weight, L2_reg, threads=64):
    """Exact reimplementation of main's WarpGBM.find_best_split (multi- and single-era)."""
    import torch
    from warpgbm.cuda import node_kernel

    E, F, B = grad_hist.shape
    per_era_gain = torch.zeros(E, F, B - 1, device=grad_hist.device, dtype=torch.float32)
    per_era_direction = torch.zeros_like(per_era_gain)

    node_kernel.compute_split(
        grad_hist,
        hess_hist,
        min_split_gain,
        min_child_weight,
        L2_reg,
        per_era_gain,
        per_era_direction,
        threads,
    )

    if E == 1:
        criterion = per_era_gain[0, :, :]
        mask = criterion > min_split_gain
    else:
        agreement = per_era_direction.mean(dim=0).abs()
        criterion = per_era_gain.mean(dim=0)
        mask = (agreement == agreement.max()) & (criterion > min_split_gain)

    if not mask.any():
        return -1, -1, criterion, None

    criterion = criterion.clone()
    criterion[mask == 0] = float("-inf")
    best_idx = torch.argmax(criterion)
    split_bins = B - 1
    if E == 1:
        return int(best_idx // split_bins), int(best_idx % split_bins), per_era_gain.mean(dim=0), None
    return (
        int(best_idx // split_bins),
        int(best_idx % split_bins),
        per_era_gain.mean(dim=0),
        per_era_direction.mean(dim=0).abs(),
    )


def fused_find_best_split(grad_hist, hess_hist, min_split_gain, min_child_weight, L2_reg, threads=64):
    """Mirror of WarpGBM._find_best_split_fused."""
    import torch
    from warpgbm.cuda import node_kernel

    E, F, B = grad_hist.shape
    fused_gain = torch.zeros(F, B - 1, device=grad_hist.device, dtype=torch.float32)
    fused_agreement = torch.zeros_like(fused_gain)
    best_split_idx = torch.empty(1, device=grad_hist.device, dtype=torch.int32)

    compute_split = node_kernel.compute_split_fused
    compute_split(
        grad_hist,
        hess_hist,
        min_split_gain,
        min_child_weight,
        L2_reg,
        fused_gain,
        fused_agreement,
        threads,
    )
    node_kernel.select_split_fused(fused_gain, fused_agreement, min_split_gain, best_split_idx, threads)

    best_idx = int(best_split_idx.item())
    if best_idx < 0:
        return -1, -1, fused_gain, fused_agreement
    split_bins = B - 1
    return best_idx // split_bins, best_idx % split_bins, fused_gain, fused_agreement


def make_histograms(rng, E, F, B, device, row_scale=200.0, sparse_eras=False):
    """Random but histogram-shaped inputs: hessians are non-negative counts, grads are sums."""
    import torch

    counts = rng.gamma(shape=2.0, scale=row_scale / (2.0 * B), size=(E, F, B))
    if sparse_eras:
        dead = rng.random((E, 1, 1)) < 0.4
        counts = counts * np.where(dead, 0.02, 1.0)
    hess = np.round(counts).astype(np.float32)
    grad = (rng.standard_normal((E, F, B)) * np.sqrt(hess + 0.25)).astype(np.float32)
    return (
        torch.tensor(grad, device=device, dtype=torch.float32),
        torch.tensor(hess, device=device, dtype=torch.float32),
    )


class FusedSplitDecisionEquivalence(unittest.TestCase):
    """The fused GPU era reduction + selection must reproduce main's split decisions."""

    def _run_config(self, seed, E, F, B, min_child_weight, min_split_gain, sparse_eras):
        import torch

        rng = np.random.default_rng(seed)
        grad, hess = make_histograms(rng, E, F, B, "cuda", sparse_eras=sparse_eras)
        L2_reg = 1e-6

        lf, lb, l_crit, l_agree = legacy_find_best_split(
            grad, hess, min_split_gain, min_child_weight, L2_reg
        )
        ff, fb, f_gain, f_agree = fused_find_best_split(
            grad, hess, min_split_gain, min_child_weight, L2_reg
        )
        torch.cuda.synchronize()

        ctx = f"seed={seed} E={E} F={F} B={B} mcw={min_child_weight} msg={min_split_gain} sparse={sparse_eras}"

        # Reduced tensors must match the legacy torch reduction (float-order tolerance only).
        np.testing.assert_allclose(
            f_gain.cpu().numpy(),
            l_crit.cpu().numpy(),
            rtol=2e-5,
            atol=2e-5,
            err_msg=f"fused_gain != legacy mean gain [{ctx}]",
        )
        if l_agree is not None:
            np.testing.assert_allclose(
                f_agree.cpu().numpy(),
                l_agree.cpu().numpy(),
                rtol=2e-5,
                atol=2e-5,
                err_msg=f"fused_agreement != legacy |mean dir| [{ctx}]",
            )

        self.assertEqual(
            (lf, lb),
            (ff, fb),
            msg=(
                f"Split decision diverged [{ctx}]: legacy=({lf},{lb}) fused=({ff},{fb}). "
                f"legacy crit={l_crit[lf, lb].item() if lf >= 0 else None} "
                f"fused gain={f_gain[ff, fb].item() if ff >= 0 else None}"
            ),
        )

    def test_small_bin_configs_match_legacy_randomized(self):
        _cuda_or_skip(self)
        # Small-bin configs: the production (Numerai) regime.
        configs = []
        seed = 0
        for E in (2, 3, 5, 8, 31, 96, 324):
            for B in (3, 5, 9):
                for mcw, sparse in ((1.0, False), (20.0, False), (60.0, True)):
                    configs.append((seed, E, 17, B, mcw, 0.0, sparse))
                    seed += 1
        # min_split_gain > 0 edge
        configs += [(seed + i, 8, 17, 5, 20.0, 0.5, False) for i in range(3)]
        for cfg in configs:
            self._run_config(*cfg)

    def test_large_bin_configs_match_legacy_randomized(self):
        _cuda_or_skip(self)
        # Large-bin configs.
        configs = []
        seed = 1000
        for E in (2, 7, 33, 96):
            for B in (10, 17, 32):
                for mcw, sparse in ((1.0, False), (40.0, True)):
                    configs.append((seed, E, 11, B, mcw, 0.0, sparse))
                    seed += 1
        for cfg in configs:
            self._run_config(*cfg)

    def test_agreement_dominates_gain(self):
        """Directional era splitting semantics: an era-consistent low-gain split must beat
        an era-inconsistent high-gain split, in both the legacy and the fused path."""
        _cuda_or_skip(self)
        import torch

        E, F, B = 4, 2, 3
        hess = torch.full((E, F, B), 50.0, device="cuda")
        grad = torch.zeros((E, F, B), device="cuda")
        # Feature 0: huge gain, but direction flips across eras -> mean direction 0.
        grad[0, 0] = torch.tensor([90.0, -45.0, -45.0])
        grad[1, 0] = torch.tensor([-90.0, 45.0, 45.0])
        grad[2, 0] = torch.tensor([80.0, -40.0, -40.0])
        grad[3, 0] = torch.tensor([-80.0, 40.0, 40.0])
        # Feature 1: modest gain, same direction in every era -> agreement 1.
        for e in range(E):
            grad[e, 1] = torch.tensor([10.0, -5.0, -5.0])

        legacy = legacy_find_best_split(grad, hess, 0.0, 1.0, 1e-6)[:2]
        fused = fused_find_best_split(grad, hess, 0.0, 1.0, 1e-6)[:2]
        self.assertEqual(legacy, (1, 0), "legacy path no longer prefers the era-consistent split")
        self.assertEqual(fused, legacy)

    def test_no_valid_split_matches(self):
        """min_child_weight kills every candidate: both paths must return no-split."""
        _cuda_or_skip(self)
        import torch

        grad = torch.randn(3, 4, 5, device="cuda")
        hess = torch.full((3, 4, 5), 1.0, device="cuda")
        legacy = legacy_find_best_split(grad, hess, 0.0, 1000.0, 1e-6)[:2]
        fused = fused_find_best_split(grad, hess, 0.0, 1000.0, 1e-6)[:2]
        self.assertEqual(legacy, (-1, -1))
        self.assertEqual(fused, (-1, -1))

    def test_max_agreement_split_below_gain_threshold_matches(self):
        """When every max-agreement cell fails min_split_gain, both paths must refuse to
        fall back to lower-agreement cells (main's mask semantics)."""
        _cuda_or_skip(self)
        import torch

        E, F, B = 4, 2, 3
        hess = torch.full((E, F, B), 50.0, device="cuda")
        grad = torch.zeros((E, F, B), device="cuda")
        # Feature 0: perfect agreement, tiny gain.
        for e in range(E):
            grad[e, 0] = torch.tensor([0.6, -0.3, -0.3])
        # Feature 1: agreement 0.5 (3 up, 1 down), big gain.
        for e in range(3):
            grad[e, 1] = torch.tensor([60.0, -30.0, -30.0])
        grad[3, 1] = torch.tensor([-60.0, 30.0, 30.0])

        min_split_gain = 1.0  # feature 0's mean gain is far below this
        legacy = legacy_find_best_split(grad, hess, min_split_gain, 1.0, 1e-6)[:2]
        fused = fused_find_best_split(grad, hess, min_split_gain, 1.0, 1e-6)[:2]
        self.assertEqual(legacy, (-1, -1))
        self.assertEqual(fused, legacy)


class HistogramPathEquivalence(unittest.TestCase):
    """New feature-tile and by-era histogram kernels vs the pre-optimization kernel."""

    def _make_model_with_data(self, N, F, E, B, seed=7):
        import torch
        from warpgbm import WarpGBM

        rng = np.random.default_rng(seed)
        model = WarpGBM(num_bins=B, threads_per_block=64, rows_per_thread=4)
        model.num_eras = E
        model.num_bins = B
        model.bin_indices = torch.tensor(
            rng.integers(0, B, size=(N, F)), device="cuda", dtype=torch.int8
        ).contiguous()
        model.residual = torch.tensor(
            rng.standard_normal(N), device="cuda", dtype=torch.float32
        )
        era_sizes = rng.multinomial(N - E, np.full(E, 1.0 / E)) + 1
        era_id = np.repeat(np.arange(E), era_sizes)
        model.era_indices = torch.tensor(era_id, device="cuda", dtype=torch.int32)
        model._setup_sorted_era_histogram_boundaries(era_id)
        return model

    def test_feature_tile_and_by_era_match_legacy_histogram(self):
        _cuda_or_skip(self)
        import torch
        from warpgbm.cuda import node_kernel

        N, F, E, B = 50_000, 12, 6, 5
        model = self._make_model_with_data(N, F, E, B)
        feature_indices = torch.arange(F, device="cuda", dtype=torch.int32)

        # A non-trivial node: every third row, still > BY_ERA_HISTOGRAM_MIN_ROWS.
        node = torch.arange(0, N, 3, device="cuda", dtype=torch.int32)

        legacy_grad = torch.zeros((E, F, B), device="cuda", dtype=torch.float32)
        legacy_hess = torch.zeros_like(legacy_grad)
        node_kernel.compute_histogram3(
            model.bin_indices,
            model.residual,
            node,
            feature_indices,
            model.era_indices,
            legacy_grad,
            legacy_hess,
            B,
            model.threads_per_block,
            model.rows_per_thread,
        )

        # New generic path (era_sorted disabled -> feature-tile kernel).
        model.era_sorted_for_hist = False
        tile_grad, tile_hess = model.compute_histograms(node, feature_indices)

        # New by-era path (era_sorted enabled, node is large enough to route there).
        model.era_sorted_for_hist = True
        by_era_grad, by_era_hess = model.compute_histograms(node, feature_indices)
        torch.cuda.synchronize()

        # Hessians are counts (sums of 1.0f): exact regardless of accumulation order.
        np.testing.assert_array_equal(tile_hess.cpu().numpy(), legacy_hess.cpu().numpy())
        np.testing.assert_array_equal(by_era_hess.cpu().numpy(), legacy_hess.cpu().numpy())
        # Gradients differ only by float accumulation order.
        np.testing.assert_allclose(
            tile_grad.cpu().numpy(), legacy_grad.cpu().numpy(), rtol=1e-4, atol=5e-3
        )
        np.testing.assert_allclose(
            by_era_grad.cpu().numpy(), legacy_grad.cpu().numpy(), rtol=1e-4, atol=5e-3
        )

    def test_small_node_generic_path_matches_legacy(self):
        _cuda_or_skip(self)
        import torch
        from warpgbm.cuda import node_kernel

        N, F, E, B = 20_000, 8, 5, 5
        model = self._make_model_with_data(N, F, E, B, seed=11)
        feature_indices = torch.arange(F, device="cuda", dtype=torch.int32)
        node = torch.arange(0, 2000, device="cuda", dtype=torch.int32)  # below routing threshold

        legacy_grad = torch.zeros((E, F, B), device="cuda", dtype=torch.float32)
        legacy_hess = torch.zeros_like(legacy_grad)
        node_kernel.compute_histogram3(
            model.bin_indices,
            model.residual,
            node,
            feature_indices,
            model.era_indices,
            legacy_grad,
            legacy_hess,
            B,
            model.threads_per_block,
            model.rows_per_thread,
        )
        grad, hess = model.compute_histograms(node, feature_indices)
        torch.cuda.synchronize()

        np.testing.assert_array_equal(hess.cpu().numpy(), legacy_hess.cpu().numpy())
        np.testing.assert_allclose(grad.cpu().numpy(), legacy_grad.cpu().numpy(), rtol=1e-4, atol=5e-3)


class EndToEndLegacyVsOptimized(unittest.TestCase):
    """Full training comparison: legacy histogram + legacy split reduction vs the
    optimized default pipeline (by-era histograms, root hessian reuse, fused split)."""

    @staticmethod
    def _make_legacy_model_class():
        import torch
        from warpgbm import WarpGBM
        from warpgbm.cuda import node_kernel

        class LegacyPathWarpGBM(WarpGBM):
            def _setup_sorted_era_histogram_boundaries(self, era_id):
                # Disables by-era histogram routing and root-hessian reuse.
                self.era_sorted_for_hist = False
                self.era_row_boundaries = None
                self.max_root_era_count = 0

            def compute_histograms(self, sample_indices, feature_indices):
                grad_hist = torch.zeros(
                    (self.num_eras, len(feature_indices), self.num_bins),
                    device=self.device,
                    dtype=torch.float32,
                )
                hess_hist = torch.zeros_like(grad_hist)
                node_kernel.compute_histogram3(
                    self.bin_indices,
                    self.residual,
                    sample_indices,
                    feature_indices,
                    self.era_indices,
                    grad_hist,
                    hess_hist,
                    self.num_bins,
                    self.threads_per_block,
                    self.rows_per_thread,
                )
                return grad_hist, hess_hist

            def find_best_split(self, gradient_histogram, hessian_histogram):
                f, b, _, _ = legacy_find_best_split(
                    gradient_histogram,
                    hessian_histogram,
                    self.min_split_gain,
                    self.min_child_weight,
                    self.L2_reg,
                    self.threads_per_block,
                )
                return f, b

        return LegacyPathWarpGBM

    @staticmethod
    def _make_data(seed=123):
        rng = np.random.default_rng(seed)
        N, F, E, B = 48_000, 16, 8, 5
        era_id = np.repeat(np.arange(E), N // E)
        X = rng.integers(0, B, size=(N, F)).astype(np.int8)
        # Era-consistent signal on features 0 and 3, era-flipping signal on feature 1
        # (era splitting should distrust feature 1), plus noise.
        w0 = 0.5 + 0.1 * rng.standard_normal(E)
        w1 = 0.6 * np.sign(rng.standard_normal(E))
        y = (
            w0[era_id] * X[:, 0]
            - 0.35 * X[:, 3]
            + w1[era_id] * X[:, 1]
            + 0.15 * X[:, 7]
            + rng.normal(0, 0.5, N)
        ).astype(np.float32)
        return X, y, era_id

    @staticmethod
    def _collect_structure(node, out):
        if "leaf_value" in node:
            value = node["leaf_value"]
            value = float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
            out.append(("leaf", node["samples"], value))
            return
        feature = node["feature"]
        feature = int(feature.item()) if hasattr(feature, "item") else int(feature)
        out.append(("split", feature, int(node["bin"])))
        EndToEndLegacyVsOptimized._collect_structure(node["left"], out)
        EndToEndLegacyVsOptimized._collect_structure(node["right"], out)

    def test_training_matches_legacy_pipeline(self):
        _cuda_or_skip(self)
        from warpgbm import WarpGBM

        X, y, era_id = self._make_data()
        params = dict(
            objective="regression",
            num_bins=5,
            max_depth=4,
            n_estimators=12,
            learning_rate=0.1,
            min_child_weight=20,
            min_split_gain=0.0,
            threads_per_block=64,
            rows_per_thread=4,
        )

        LegacyPathWarpGBM = self._make_legacy_model_class()
        legacy_model = LegacyPathWarpGBM(**params)
        legacy_model.fit(X, y, era_id=era_id)
        legacy_pred = legacy_model.predict(X)

        opt_model = WarpGBM(**params)
        opt_model.fit(X, y, era_id=era_id)
        opt_pred = opt_model.predict(X)

        # 1) Identical tree structure: same feature/bin at every node, same node sizes.
        self.assertEqual(len(legacy_model.forest), len(opt_model.forest))
        for t, (lt, ot) in enumerate(zip(legacy_model.forest, opt_model.forest)):
            legacy_nodes, opt_nodes = [], []
            self._collect_structure(lt, legacy_nodes)
            self._collect_structure(ot, opt_nodes)
            self.assertEqual(len(legacy_nodes), len(opt_nodes), f"tree {t}: node count differs")
            for i, (ln, on) in enumerate(zip(legacy_nodes, opt_nodes)):
                self.assertEqual(
                    ln[:2] if ln[0] == "leaf" else ln,
                    on[:2] if on[0] == "leaf" else on,
                    f"tree {t}, node {i}: structure diverged (legacy={ln}, optimized={on})",
                )
                if ln[0] == "leaf":
                    self.assertAlmostEqual(
                        ln[2], on[2], places=4, msg=f"tree {t}, node {i}: leaf value diverged"
                    )

        # 2) Predictions agree to float-accumulation tolerance.
        np.testing.assert_allclose(legacy_pred, opt_pred, rtol=0, atol=1e-4)

        # 3) Per-era feature importance agrees.
        np.testing.assert_allclose(
            legacy_model.per_era_feature_importance_,
            opt_model.per_era_feature_importance_,
            rtol=1e-3,
            atol=1e-3,
        )

        # 4) Sanity: era splitting distrusts the era-flipping feature 1.
        importance = opt_model.feature_importance_
        self.assertLess(
            importance[1],
            importance[0],
            "era-inconsistent feature outranked era-consistent feature — era splitting broken",
        )


if __name__ == "__main__":
    unittest.main()
