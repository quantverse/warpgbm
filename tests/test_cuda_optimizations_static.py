import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text()


class CudaOptimizationStaticTests(unittest.TestCase):
    def test_cuda_launch_errors_raise_to_python(self):
        histogram = read_source("warpgbm/cuda/histogram_kernel.cu")
        split = read_source("warpgbm/cuda/best_split_kernel.cu")

        self.assertIn("TORCH_CHECK", histogram)
        self.assertIn("TORCH_CHECK", split)
        self.assertNotIn('printf("CUDA histogram kernel launch failed', histogram)
        self.assertNotIn('printf("Fused directional split kernel launch failed', split)

    def test_multi_era_feature_importance_respects_child_weight(self):
        split = read_source("warpgbm/cuda/best_split_kernel.cu")
        start = split.index("__global__ void accumulate_feature_importance_kernel")
        end = split.index("void launch_accumulate_feature_importance_kernel")
        importance_kernel = split[start:end]

        self.assertIn("H_L >= min_child_samples && H_R >= min_child_samples", importance_kernel)
        self.assertIn("gain = (G_L * G_L)", importance_kernel)

    def test_fused_split_parallelizes_over_eras_with_shared_reduction(self):
        split = read_source("warpgbm/cuda/best_split_kernel.cu")
        start = split.index("__global__ void directional_split_fused_kernel")
        end = split.index("void launch_directional_split_fused_kernel")
        fused_kernel = split[start:end]

        self.assertRegex(fused_kernel, r"int\s+f\s*=\s*blockIdx\.x\s*;")
        self.assertRegex(
            fused_kernel,
            r"for\s*\(\s*int\s+e\s*=\s*threadIdx\.x\s*;\s*e\s*<\s*E\s*;"
            r"\s*e\s*\+=\s*blockDim\.x\s*\)",
        )
        self.assertIn("extern __shared__ float shmem[]", fused_kernel)
        self.assertIn("atomicAdd(&sh_gain", fused_kernel)
        self.assertNotRegex(
            fused_kernel,
            r"int\s+f\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x",
        )

    def test_large_era_histogram_uses_single_scan_global_fallback(self):
        histogram = read_source("warpgbm/cuda/histogram_kernel.cu")

        self.assertIn("histogram_global_atomic_kernel", histogram)
        self.assertIn("if (shared_mem_bytes <= DEFAULT_DYNAMIC_SHARED_LIMIT)", histogram)
        self.assertRegex(
            histogram,
            r"histogram_global_atomic_kernel<<<.*blocks,\s*threads>>>",
        )
        self.assertNotRegex(histogram, r"for\s*\(\s*int\s+era_start\s*=\s*0;")

    def test_by_era_histogram_skips_empty_row_tiles(self):
        histogram = read_source("warpgbm/cuda/histogram_kernel.cu")
        start = histogram.index("__global__ void histogram_feature_tile_by_era_kernel")
        end = histogram.index("__global__ void histogram_feature_tile_grad_kernel")
        by_era_kernel = histogram[start:end]

        self.assertIn("if (row_tile_start >= era_end)", by_era_kernel)


if __name__ == "__main__":
    unittest.main()
