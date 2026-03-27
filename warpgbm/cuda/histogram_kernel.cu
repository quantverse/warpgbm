#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void histogram_tiled_configurable_kernel(
    const int8_t *__restrict__ bin_indices, // [N, F_master]
    const float *__restrict__ residuals,    // [N]
    const int32_t *__restrict__ sample_indices, // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_indices, // [N]
    float *__restrict__ grad_hist,          // [E * F * B]
    float *__restrict__ hess_hist,          // [E * F * B]
    int64_t N, int64_t F_master, int64_t F, int64_t B, int64_t num_eras,
    int era_start, int era_end,
    int rows_per_thread)
{
    int hist_feat_idx = blockIdx.x;
    int feat = feature_indices[ hist_feat_idx ]; // 1 block per feature
    int row_start = (blockIdx.y * blockDim.x + threadIdx.x) * rows_per_thread;

    int era_tile_size = era_end - era_start;

    extern __shared__ float shmem[];
    float *sh_grad = shmem;                             // [era_tile_size * B]
    float *sh_hess = &sh_grad[era_tile_size * B];       // [era_tile_size * B]

    // Initialize shared memory histograms
    for (int i = threadIdx.x; i < era_tile_size * B; i += blockDim.x) {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
    }

    __syncthreads();

    // Each thread processes multiple rows
    for (int r = 0; r < rows_per_thread; ++r)
    {
        int row = row_start + r;
        if (row < N)
        {
            int sample = sample_indices[row];
            int32_t era = era_indices[sample];
            // Only process samples belonging to eras in this tile
            if (era >= era_start && era < era_end)
            {
                int8_t bin = bin_indices[sample * F_master + feat];
                if (bin >= 0 && bin < B)
                {
                    int local_era = era - era_start;
                    atomicAdd(&sh_grad[local_era * B + bin], residuals[sample]);
                    atomicAdd(&sh_hess[local_era * B + bin], 1.0f);
                }
            }
        }
    }
    __syncthreads();

    // Write results back to global memory
    for (int b = threadIdx.x; b < era_tile_size * B; b += blockDim.x)
    {
        int local_e = b / B;
        int bin = b % B;
        int global_e = era_start + local_e;
        int64_t idx = global_e * F * B + hist_feat_idx * B + bin;

        atomicAdd(&grad_hist[idx], sh_grad[b]);
        atomicAdd(&hess_hist[idx], sh_hess[b]);
    }

}

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
    int rows_per_thread = 1)
{

    int64_t N = sample_indices.size(0);
    int64_t F = feature_indices.size(0);
    int num_features_master = bin_indices.size(1);

    int64_t rows_per_block = threads_per_block * rows_per_thread;
    int64_t row_tiles = (N + rows_per_block - 1) / rows_per_block;

    dim3 blocks(F, row_tiles); // grid.x = F, grid.y = row_tiles
    dim3 threads(threads_per_block);
    int num_eras = grad_hist.size(0); // inferred from output tensor

    // Compute max eras per tile that fit in shared memory (48KB default limit)
    int max_shared_bytes = 49152; // conservative default
    int max_eras_per_tile = max_shared_bytes / (2 * num_bins * (int)sizeof(float));
    if (max_eras_per_tile < 1) max_eras_per_tile = 1;

    // Launch one kernel per era tile
    for (int era_start = 0; era_start < num_eras; era_start += max_eras_per_tile) {
        int era_end = era_start + max_eras_per_tile;
        if (era_end > num_eras) era_end = num_eras;
        int era_tile_size = era_end - era_start;
        int shared_mem_bytes = 2 * era_tile_size * num_bins * (int)sizeof(float);

        histogram_tiled_configurable_kernel<<<blocks, threads, shared_mem_bytes>>>(
            bin_indices.data_ptr<int8_t>(),
            residuals.data_ptr<float>(),
            sample_indices.data_ptr<int32_t>(),
            feature_indices.data_ptr<int32_t>(),
            era_indices.data_ptr<int32_t>(),
            grad_hist.data_ptr<float>(),
            hess_hist.data_ptr<float>(),
            N, num_features_master, F, num_bins, num_eras,
            era_start, era_end,
            rows_per_thread);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA histogram kernel launch failed (eras %d-%d): %s\n",
                   era_start, era_end, cudaGetErrorString(err));
            return;
        }
    }
}
