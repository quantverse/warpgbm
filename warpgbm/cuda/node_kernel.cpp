#include <torch/extension.h>
#include <vector>

// Declare the function from histogram_kernel.cu

void launch_directional_split_kernel(
    const at::Tensor &G, // [E, F, B]
    const at::Tensor &H, // [E, F, B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &per_era_gain,       // [E, F, B]
    at::Tensor &per_era_direction,  // [E, F, B]
    int threads = 128);

void launch_histogram_kernel_cuda_configurable(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_histogram_feature_major_kernel(
    const at::Tensor &bin_indices_feature_major,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_histogram_feature_tile_kernel(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_histogram_feature_tile_by_era_kernel(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_offsets,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int max_era_count,
    int active_era_start,
    int active_era_end,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_histogram_feature_tile_grad_kernel(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_histogram_feature_tile_by_era_grad_kernel(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_offsets,
    at::Tensor &grad_hist,
    int num_bins,
    int max_era_count,
    int active_era_start,
    int active_era_end,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_histogram_feature_tile_root_kernel(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_subtract_histograms_inplace_kernel(
    at::Tensor &parent_grad,
    at::Tensor &parent_hess,
    const at::Tensor &child_grad,
    const at::Tensor &child_hess,
    int threads = 256);

void launch_directional_split_fused_kernel(
    const at::Tensor &G, // [E, F, B]
    const at::Tensor &H, // [E, F, B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &fused_gain,       // [F, B-1]
    at::Tensor &fused_agreement,  // [F, B-1]
    int threads = 128);

void launch_directional_split_fused_no_atomic_kernel(
    const at::Tensor &G,
    const at::Tensor &H,
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &fused_gain,
    at::Tensor &fused_agreement,
    int threads = 128);

void launch_select_split_fused_kernel(
    const at::Tensor &fused_gain,
    const at::Tensor &fused_agreement,
    float min_split_gain,
    at::Tensor &best_idx,
    int threads = 128);

void launch_accumulate_feature_importance_kernel(
    const at::Tensor &G,
    const at::Tensor &H,
    int local_feature,
    int best_bin,
    int global_feature,
    float min_child_samples,
    float eps,
    at::Tensor &per_era_feature_importance,
    int threads = 128);

void launch_bin_column_kernel(
    at::Tensor X,
    at::Tensor bin_edges,
    at::Tensor bin_indices);

void predict_with_forest(
    const at::Tensor &bin_indices, // [N x F], int8
    const at::Tensor &tree_tensor, // [T x max_nodes x 6], float32
    float learning_rate,
    at::Tensor &out // [N], float32
);

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_histogram3", &launch_histogram_kernel_cuda_configurable, "Histogram Feature Shared Mem");
    m.def("compute_histogram_feature_major", &launch_histogram_feature_major_kernel, "Histogram Feature-Major Layout");
    m.def("compute_histogram_feature_tile", &launch_histogram_feature_tile_kernel, "Histogram Row-Major Feature Tile");
    m.def("compute_histogram_feature_tile_by_era", &launch_histogram_feature_tile_by_era_kernel, "By-Era Row-Major Feature Tile");
    m.def("compute_histogram_feature_tile_grad", &launch_histogram_feature_tile_grad_kernel, "Grad-Only Row-Major Feature Tile");
    m.def("compute_histogram_feature_tile_by_era_grad", &launch_histogram_feature_tile_by_era_grad_kernel, "By-Era Grad-Only Row-Major Feature Tile");
    m.def("compute_histogram_feature_tile_root", &launch_histogram_feature_tile_root_kernel, "Root Histogram Row-Major Feature Tile");
    m.def("subtract_histograms_inplace", &launch_subtract_histograms_inplace_kernel, "Subtract Grad/Hess Histograms In-Place (CUDA)");
    m.def("compute_split", &launch_directional_split_kernel, "Best Split (CUDA)");
    m.def("compute_split_fused", &launch_directional_split_fused_kernel, "Best Split Fused (CUDA)");
    m.def("compute_split_fused_no_atomic", &launch_directional_split_fused_no_atomic_kernel, "Best Split Fused No-Atomic (CUDA)");
    m.def("select_split_fused", &launch_select_split_fused_kernel, "Select Best Fused Split (CUDA)");
    m.def("accumulate_feature_importance", &launch_accumulate_feature_importance_kernel, "Accumulate Selected Split Feature Importance (CUDA)");
    m.def("custom_cuda_binner", &launch_bin_column_kernel, "Custom CUDA binning kernel");
    m.def("predict_forest", &predict_with_forest, "CUDA Predictions");
}
