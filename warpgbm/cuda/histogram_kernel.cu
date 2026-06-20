#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <climits>

constexpr int ERA_TILE_CAP = 16;
constexpr int FEATURE_TILE = 32;

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
    int64_t row_start = (blockIdx.y * blockDim.x + threadIdx.x) * rows_per_thread;

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
        int64_t row = row_start + r;
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

__global__ void histogram_global_atomic_kernel(
    const int8_t *__restrict__ bin_indices, // [N, F_master]
    const float *__restrict__ residuals,    // [N]
    const int32_t *__restrict__ sample_indices, // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_indices, // [N]
    float *__restrict__ grad_hist,          // [E * F * B]
    float *__restrict__ hess_hist,          // [E * F * B]
    int64_t N, int64_t F_master, int64_t F, int64_t B, int64_t num_eras,
    int rows_per_thread)
{
    int hist_feat_idx = blockIdx.x;
    int feat = feature_indices[hist_feat_idx];
    int64_t row_start = (blockIdx.y * blockDim.x + threadIdx.x) * rows_per_thread;

    for (int r = 0; r < rows_per_thread; ++r)
    {
        int64_t row = row_start + r;
        if (row < N)
        {
            int sample = sample_indices[row];
            int8_t bin = bin_indices[sample * F_master + feat];
            int32_t era = era_indices[sample];
            if (bin >= 0 && bin < B && era >= 0 && era < num_eras)
            {
                int64_t idx = era * F * B + hist_feat_idx * B + bin;
                atomicAdd(&grad_hist[idx], residuals[sample]);
                atomicAdd(&hess_hist[idx], 1.0f);
            }
        }
    }
}

__global__ void histogram_feature_major_shared_kernel(
    const int8_t *__restrict__ bin_indices_feature_major, // [F_master, N_master]
    const float *__restrict__ residuals,    // [N_master]
    const int32_t *__restrict__ sample_indices, // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_indices, // [N_master]
    float *__restrict__ grad_hist,          // [E * F * B]
    float *__restrict__ hess_hist,          // [E * F * B]
    int64_t N, int64_t N_master, int64_t F, int64_t B, int64_t num_eras,
    int rows_per_thread)
{
    int hist_feat_idx = blockIdx.x;
    int feat = feature_indices[hist_feat_idx];
    int64_t row_start = (blockIdx.y * blockDim.x + threadIdx.x) * rows_per_thread;

    extern __shared__ float shmem_feature_major[];
    float *sh_grad = shmem_feature_major;                              // [ERA_TILE_CAP * B]
    float *sh_hess = &sh_grad[ERA_TILE_CAP * B];                       // [ERA_TILE_CAP * B]
    int *sh_meta = reinterpret_cast<int *>(&sh_hess[ERA_TILE_CAP * B]); // [min_era, max_era]

    if (threadIdx.x == 0) {
        sh_meta[0] = INT_MAX;
        sh_meta[1] = -1;
    }
    __syncthreads();

    int64_t feature_offset = (int64_t)feat * N_master;

    for (int r = 0; r < rows_per_thread; ++r)
    {
        int64_t row = row_start + r;
        if (row < N)
        {
            int sample = sample_indices[row];
            int32_t era = era_indices[sample];
            if (era >= 0 && era < num_eras)
            {
                atomicMin(&sh_meta[0], (int)era);
                atomicMax(&sh_meta[1], (int)era);
            }
        }
    }
    __syncthreads();

    int era_base = sh_meta[0];
    int era_last = sh_meta[1];
    int era_span = era_last - era_base + 1;

    if (era_span <= 0) {
        return;
    }

    if (era_span > ERA_TILE_CAP) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int64_t row = row_start + r;
            if (row < N)
            {
                int sample = sample_indices[row];
                int8_t bin = bin_indices_feature_major[feature_offset + sample];
                int32_t era = era_indices[sample];
                if (bin >= 0 && bin < B && era >= 0 && era < num_eras)
                {
                    int64_t idx = era * F * B + hist_feat_idx * B + bin;
                    atomicAdd(&grad_hist[idx], residuals[sample]);
                    atomicAdd(&hess_hist[idx], 1.0f);
                }
            }
        }
        return;
    }

    for (int i = threadIdx.x; i < era_span * B; i += blockDim.x) {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
    }
    __syncthreads();

    for (int r = 0; r < rows_per_thread; ++r)
    {
        int64_t row = row_start + r;
        if (row < N)
        {
            int sample = sample_indices[row];
            int32_t era = era_indices[sample];
            if (era >= 0 && era < num_eras)
            {
                int8_t bin = bin_indices_feature_major[feature_offset + sample];
                int local_era = era - era_base;
                if (bin >= 0 && bin < B && local_era >= 0 && local_era < era_span)
                {
                    atomicAdd(&sh_grad[local_era * B + bin], residuals[sample]);
                    atomicAdd(&sh_hess[local_era * B + bin], 1.0f);
                }
            }
        }
    }
    __syncthreads();

    for (int b = threadIdx.x; b < era_span * B; b += blockDim.x)
    {
        int e = era_base + b / B;
        int bin = b % B;
        int64_t idx = e * F * B + hist_feat_idx * B + bin;
        atomicAdd(&grad_hist[idx], sh_grad[b]);
        atomicAdd(&hess_hist[idx], sh_hess[b]);
    }
}

__global__ void histogram_feature_major_global_kernel(
    const int8_t *__restrict__ bin_indices_feature_major, // [F_master, N_master]
    const float *__restrict__ residuals,    // [N_master]
    const int32_t *__restrict__ sample_indices, // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_indices, // [N_master]
    float *__restrict__ grad_hist,          // [E * F * B]
    float *__restrict__ hess_hist,          // [E * F * B]
    int64_t N, int64_t N_master, int64_t F, int64_t B, int64_t num_eras,
    int rows_per_thread)
{
    int hist_feat_idx = blockIdx.x;
    int feat = feature_indices[hist_feat_idx];
    int64_t row_start = (blockIdx.y * blockDim.x + threadIdx.x) * rows_per_thread;
    int64_t feature_offset = (int64_t)feat * N_master;

    for (int r = 0; r < rows_per_thread; ++r)
    {
        int64_t row = row_start + r;
        if (row < N)
        {
            int sample = sample_indices[row];
            int8_t bin = bin_indices_feature_major[feature_offset + sample];
            int32_t era = era_indices[sample];
            if (bin >= 0 && bin < B && era >= 0 && era < num_eras)
            {
                int64_t idx = era * F * B + hist_feat_idx * B + bin;
                atomicAdd(&grad_hist[idx], residuals[sample]);
                atomicAdd(&hess_hist[idx], 1.0f);
            }
        }
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

    // Keep the shared-memory path when it fits. For large era counts, use one
    // global-atomic scan instead of rereading the same rows once per era tile.
    int64_t max_shared_bytes = 49152; // conservative default
    int64_t shared_mem_bytes = 2LL * num_eras * num_bins * (int64_t)sizeof(float);

    if (shared_mem_bytes <= max_shared_bytes) {
        histogram_tiled_configurable_kernel<<<blocks, threads, shared_mem_bytes>>>(
            bin_indices.data_ptr<int8_t>(),
            residuals.data_ptr<float>(),
            sample_indices.data_ptr<int32_t>(),
            feature_indices.data_ptr<int32_t>(),
            era_indices.data_ptr<int32_t>(),
            grad_hist.data_ptr<float>(),
            hess_hist.data_ptr<float>(),
            N, num_features_master, F, num_bins, num_eras,
            0, num_eras,
            rows_per_thread);

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "CUDA histogram shared-memory kernel launch failed: ",
                    cudaGetErrorString(err));
        return;
    }

    histogram_global_atomic_kernel<<<blocks, threads>>>(
        bin_indices.data_ptr<int8_t>(),
        residuals.data_ptr<float>(),
        sample_indices.data_ptr<int32_t>(),
        feature_indices.data_ptr<int32_t>(),
        era_indices.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        N, num_features_master, F, num_bins, num_eras,
        rows_per_thread);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA histogram global-atomic kernel launch failed: ",
                cudaGetErrorString(err));
}

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
    int rows_per_thread = 1)
{
    int64_t N = sample_indices.size(0);
    int64_t N_master = bin_indices_feature_major.size(1);
    int64_t F = feature_indices.size(0);

    int64_t rows_per_block = threads_per_block * rows_per_thread;
    int64_t row_tiles = (N + rows_per_block - 1) / rows_per_block;

    dim3 blocks(F, row_tiles);
    dim3 threads(threads_per_block);
    int num_eras = grad_hist.size(0);

    int shared_mem_bytes = 2 * ERA_TILE_CAP * num_bins * (int)sizeof(float) + 2 * (int)sizeof(int);

    histogram_feature_major_shared_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices_feature_major.data_ptr<int8_t>(),
        residuals.data_ptr<float>(),
        sample_indices.data_ptr<int32_t>(),
        feature_indices.data_ptr<int32_t>(),
        era_indices.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        N, N_master, F, num_bins, num_eras,
        rows_per_thread);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA feature-major sparse histogram kernel launch failed: ",
                cudaGetErrorString(err));
}

__global__ void histogram_feature_tile_kernel(
    const int8_t *__restrict__ bin_indices, // [N_master, F_master]
    const float *__restrict__ residuals,    // [N_master]
    const int32_t *__restrict__ sample_indices, // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_indices, // [N_master]
    float *__restrict__ grad_hist,          // [E * F * B]
    float *__restrict__ hess_hist,          // [E * F * B]
    int64_t N, int64_t F_master, int64_t F, int64_t B, int64_t num_eras,
    int rows_per_thread)
{
    int local_f = threadIdx.x % FEATURE_TILE;
    int row_lane = threadIdx.x / FEATURE_TILE;
    int row_lanes = blockDim.x / FEATURE_TILE;
    int rows_per_block = row_lanes * rows_per_thread;
    int hist_feat_idx = blockIdx.x * FEATURE_TILE + local_f;
    bool valid_feature = hist_feat_idx < F;
    int feat = valid_feature ? feature_indices[hist_feat_idx] : 0;
    int64_t row_tile_start = (int64_t)blockIdx.y * rows_per_block;

    extern __shared__ float shmem_feature_tile[];
    float *sh_grad = shmem_feature_tile;                                          // [FEATURE_TILE * ERA_TILE_CAP * B]
    float *sh_hess = &sh_grad[FEATURE_TILE * ERA_TILE_CAP * B];                   // [FEATURE_TILE * ERA_TILE_CAP * B]
    float *sh_residual = &sh_hess[FEATURE_TILE * ERA_TILE_CAP * B];               // [rows_per_block]
    int *sh_sample = reinterpret_cast<int *>(&sh_residual[rows_per_block]);       // [rows_per_block]
    int *sh_era = &sh_sample[rows_per_block];                                     // [rows_per_block]
    int *sh_meta = &sh_era[rows_per_block];                                       // [min_era, max_era]

    if (threadIdx.x == 0) {
        sh_meta[0] = INT_MAX;
        sh_meta[1] = -1;
    }
    __syncthreads();

    if (local_f == 0) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int local_row = r * row_lanes + row_lane;
            int64_t row = row_tile_start + local_row;
            if (row < N)
            {
                int sample = sample_indices[row];
                int32_t era = era_indices[sample];
                sh_sample[local_row] = sample;
                sh_era[local_row] = era;
                sh_residual[local_row] = residuals[sample];
                if (era >= 0 && era < num_eras)
                {
                    atomicMin(&sh_meta[0], (int)era);
                    atomicMax(&sh_meta[1], (int)era);
                }
            }
            else
            {
                sh_sample[local_row] = -1;
                sh_era[local_row] = -1;
                sh_residual[local_row] = 0.0f;
            }
        }
    }
    __syncthreads();

    int era_base = sh_meta[0];
    int era_last = sh_meta[1];
    int era_span = era_last - era_base + 1;
    if (era_span <= 0) {
        return;
    }

    if (era_span > ERA_TILE_CAP) {
        if (valid_feature) {
            for (int r = 0; r < rows_per_thread; ++r)
            {
                int local_row = r * row_lanes + row_lane;
                int sample = sh_sample[local_row];
                if (sample >= 0)
                {
                    int32_t era = sh_era[local_row];
                    int8_t bin = bin_indices[(int64_t)sample * F_master + feat];
                    if (bin >= 0 && bin < B && era >= 0 && era < num_eras)
                    {
                        int64_t idx = era * F * B + hist_feat_idx * B + bin;
                        atomicAdd(&grad_hist[idx], sh_residual[local_row]);
                        atomicAdd(&hess_hist[idx], 1.0f);
                    }
                }
            }
        }
        return;
    }

    int hist_values = FEATURE_TILE * era_span * B;
    for (int i = threadIdx.x; i < hist_values; i += blockDim.x) {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
    }
    __syncthreads();

    if (valid_feature) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int local_row = r * row_lanes + row_lane;
            int sample = sh_sample[local_row];
            if (sample >= 0)
            {
                int32_t era = sh_era[local_row];
                int local_era = era - era_base;
                int8_t bin = bin_indices[(int64_t)sample * F_master + feat];
                if (bin >= 0 && bin < B && local_era >= 0 && local_era < era_span)
                {
                    int sh_idx = (local_f * era_span + local_era) * B + bin;
                    atomicAdd(&sh_grad[sh_idx], sh_residual[local_row]);
                    atomicAdd(&sh_hess[sh_idx], 1.0f);
                }
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hist_values; i += blockDim.x)
    {
        int lf = i / (era_span * B);
        int rem = i % (era_span * B);
        int e = era_base + rem / B;
        int bin = rem % B;
        int out_feature = blockIdx.x * FEATURE_TILE + lf;
        if (out_feature < F)
        {
            int64_t idx = e * F * B + out_feature * B + bin;
            atomicAdd(&grad_hist[idx], sh_grad[i]);
            atomicAdd(&hess_hist[idx], sh_hess[i]);
        }
    }
}

__global__ void histogram_feature_tile_by_era_kernel(
    const int8_t *__restrict__ bin_indices, // [N_master, F_master]
    const float *__restrict__ residuals,    // [N_master]
    const int32_t *__restrict__ sample_indices, // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_offsets, // [E + 1], positions in sample_indices
    float *__restrict__ grad_hist,          // [E * F * B]
    float *__restrict__ hess_hist,          // [E * F * B]
    int64_t F_master, int64_t F, int64_t B,
    int active_era_start,
    int rows_per_thread)
{
    int local_f = threadIdx.x % FEATURE_TILE;
    int row_lane = threadIdx.x / FEATURE_TILE;
    int row_lanes = blockDim.x / FEATURE_TILE;
    int rows_per_block = row_lanes * rows_per_thread;
    int hist_feat_idx = blockIdx.x * FEATURE_TILE + local_f;
    int era = active_era_start + blockIdx.z;
    int era_start = era_offsets[era];
    int era_end = era_offsets[era + 1];
    int row_tile_start = era_start + blockIdx.y * rows_per_block;
    if (row_tile_start >= era_end) {
        return;
    }
    bool valid_feature = hist_feat_idx < F;
    int feat = valid_feature ? feature_indices[hist_feat_idx] : 0;

    extern __shared__ float shmem_feature_tile_by_era[];
    float *sh_grad = shmem_feature_tile_by_era;                         // [FEATURE_TILE * B]
    float *sh_hess = &sh_grad[FEATURE_TILE * B];                        // [FEATURE_TILE * B]
    float *sh_residual = &sh_hess[FEATURE_TILE * B];                    // [rows_per_block]
    int *sh_sample = reinterpret_cast<int *>(&sh_residual[rows_per_block]); // [rows_per_block]

    int hist_values = FEATURE_TILE * B;
    for (int i = threadIdx.x; i < hist_values; i += blockDim.x) {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
    }

    if (local_f == 0) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int local_row = r * row_lanes + row_lane;
            int row = row_tile_start + local_row;
            if (row < era_end)
            {
                int sample = sample_indices[row];
                sh_sample[local_row] = sample;
                sh_residual[local_row] = residuals[sample];
            }
            else
            {
                sh_sample[local_row] = -1;
                sh_residual[local_row] = 0.0f;
            }
        }
    }
    __syncthreads();

    if (valid_feature) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int local_row = r * row_lanes + row_lane;
            int sample = sh_sample[local_row];
            if (sample >= 0)
            {
                int8_t bin = bin_indices[(int64_t)sample * F_master + feat];
                if (bin >= 0 && bin < B)
                {
                    int sh_idx = local_f * B + bin;
                    atomicAdd(&sh_grad[sh_idx], sh_residual[local_row]);
                    atomicAdd(&sh_hess[sh_idx], 1.0f);
                }
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hist_values; i += blockDim.x)
    {
        int lf = i / B;
        int bin = i % B;
        int out_feature = blockIdx.x * FEATURE_TILE + lf;
        if (out_feature < F)
        {
            int64_t idx = (int64_t)era * F * B + out_feature * B + bin;
            atomicAdd(&grad_hist[idx], sh_grad[i]);
            atomicAdd(&hess_hist[idx], sh_hess[i]);
        }
    }
}

__global__ void histogram_feature_tile_grad_kernel(
    const int8_t *__restrict__ bin_indices, // [N_master, F_master]
    const float *__restrict__ residuals,    // [N_master]
    const int32_t *__restrict__ sample_indices, // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_indices, // [N_master]
    float *__restrict__ grad_hist,          // [E * F * B]
    int64_t N, int64_t F_master, int64_t F, int64_t B, int64_t num_eras,
    int rows_per_thread)
{
    int local_f = threadIdx.x % FEATURE_TILE;
    int row_lane = threadIdx.x / FEATURE_TILE;
    int row_lanes = blockDim.x / FEATURE_TILE;
    int rows_per_block = row_lanes * rows_per_thread;
    int hist_feat_idx = blockIdx.x * FEATURE_TILE + local_f;
    bool valid_feature = hist_feat_idx < F;
    int feat = valid_feature ? feature_indices[hist_feat_idx] : 0;
    int64_t row_tile_start = (int64_t)blockIdx.y * rows_per_block;

    extern __shared__ float shmem_feature_tile_grad[];
    float *sh_grad = shmem_feature_tile_grad;                                  // [FEATURE_TILE * ERA_TILE_CAP * B]
    float *sh_residual = &sh_grad[FEATURE_TILE * ERA_TILE_CAP * B];            // [rows_per_block]
    int *sh_sample = reinterpret_cast<int *>(&sh_residual[rows_per_block]);    // [rows_per_block]
    int *sh_era = &sh_sample[rows_per_block];                                  // [rows_per_block]
    int *sh_meta = &sh_era[rows_per_block];                                    // [min_era, max_era]

    if (threadIdx.x == 0) {
        sh_meta[0] = INT_MAX;
        sh_meta[1] = -1;
    }
    __syncthreads();

    if (local_f == 0) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int local_row = r * row_lanes + row_lane;
            int64_t row = row_tile_start + local_row;
            if (row < N)
            {
                int sample = sample_indices[row];
                int32_t era = era_indices[sample];
                sh_sample[local_row] = sample;
                sh_era[local_row] = era;
                sh_residual[local_row] = residuals[sample];
                if (era >= 0 && era < num_eras)
                {
                    atomicMin(&sh_meta[0], (int)era);
                    atomicMax(&sh_meta[1], (int)era);
                }
            }
            else
            {
                sh_sample[local_row] = -1;
                sh_era[local_row] = -1;
                sh_residual[local_row] = 0.0f;
            }
        }
    }
    __syncthreads();

    int era_base = sh_meta[0];
    int era_last = sh_meta[1];
    int era_span = era_last - era_base + 1;
    if (era_span <= 0) {
        return;
    }

    if (era_span > ERA_TILE_CAP) {
        if (valid_feature) {
            for (int r = 0; r < rows_per_thread; ++r)
            {
                int local_row = r * row_lanes + row_lane;
                int sample = sh_sample[local_row];
                if (sample >= 0)
                {
                    int32_t era = sh_era[local_row];
                    int8_t bin = bin_indices[(int64_t)sample * F_master + feat];
                    if (bin >= 0 && bin < B && era >= 0 && era < num_eras)
                    {
                        int64_t idx = era * F * B + hist_feat_idx * B + bin;
                        atomicAdd(&grad_hist[idx], sh_residual[local_row]);
                    }
                }
            }
        }
        return;
    }

    int hist_values = FEATURE_TILE * era_span * B;
    for (int i = threadIdx.x; i < hist_values; i += blockDim.x) {
        sh_grad[i] = 0.0f;
    }
    __syncthreads();

    if (valid_feature) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int local_row = r * row_lanes + row_lane;
            int sample = sh_sample[local_row];
            if (sample >= 0)
            {
                int32_t era = sh_era[local_row];
                int local_era = era - era_base;
                int8_t bin = bin_indices[(int64_t)sample * F_master + feat];
                if (bin >= 0 && bin < B && local_era >= 0 && local_era < era_span)
                {
                    int sh_idx = (local_f * era_span + local_era) * B + bin;
                    atomicAdd(&sh_grad[sh_idx], sh_residual[local_row]);
                }
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hist_values; i += blockDim.x)
    {
        int lf = i / (era_span * B);
        int rem = i % (era_span * B);
        int e = era_base + rem / B;
        int bin = rem % B;
        int out_feature = blockIdx.x * FEATURE_TILE + lf;
        if (out_feature < F)
        {
            int64_t idx = e * F * B + out_feature * B + bin;
            atomicAdd(&grad_hist[idx], sh_grad[i]);
        }
    }
}

__global__ void histogram_feature_tile_by_era_grad_kernel(
    const int8_t *__restrict__ bin_indices, // [N_master, F_master]
    const float *__restrict__ residuals,    // [N_master]
    const int32_t *__restrict__ sample_indices, // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_offsets, // [E + 1], positions in sample_indices
    float *__restrict__ grad_hist,          // [E * F * B]
    int64_t F_master, int64_t F, int64_t B,
    int active_era_start,
    int rows_per_thread)
{
    int local_f = threadIdx.x % FEATURE_TILE;
    int row_lane = threadIdx.x / FEATURE_TILE;
    int row_lanes = blockDim.x / FEATURE_TILE;
    int rows_per_block = row_lanes * rows_per_thread;
    int hist_feat_idx = blockIdx.x * FEATURE_TILE + local_f;
    int era = active_era_start + blockIdx.z;
    int era_start = era_offsets[era];
    int era_end = era_offsets[era + 1];
    int row_tile_start = era_start + blockIdx.y * rows_per_block;
    if (row_tile_start >= era_end) {
        return;
    }
    bool valid_feature = hist_feat_idx < F;
    int feat = valid_feature ? feature_indices[hist_feat_idx] : 0;

    extern __shared__ float shmem_feature_tile_by_era_grad[];
    float *sh_grad = shmem_feature_tile_by_era_grad;                // [FEATURE_TILE * B]
    float *sh_residual = &sh_grad[FEATURE_TILE * B];                // [rows_per_block]
    int *sh_sample = reinterpret_cast<int *>(&sh_residual[rows_per_block]); // [rows_per_block]

    int hist_values = FEATURE_TILE * B;
    for (int i = threadIdx.x; i < hist_values; i += blockDim.x) {
        sh_grad[i] = 0.0f;
    }

    if (local_f == 0) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int local_row = r * row_lanes + row_lane;
            int row = row_tile_start + local_row;
            if (row < era_end)
            {
                int sample = sample_indices[row];
                sh_sample[local_row] = sample;
                sh_residual[local_row] = residuals[sample];
            }
            else
            {
                sh_sample[local_row] = -1;
                sh_residual[local_row] = 0.0f;
            }
        }
    }
    __syncthreads();

    if (valid_feature) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int local_row = r * row_lanes + row_lane;
            int sample = sh_sample[local_row];
            if (sample >= 0)
            {
                int8_t bin = bin_indices[(int64_t)sample * F_master + feat];
                if (bin >= 0 && bin < B)
                {
                    int sh_idx = local_f * B + bin;
                    atomicAdd(&sh_grad[sh_idx], sh_residual[local_row]);
                }
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hist_values; i += blockDim.x)
    {
        int lf = i / B;
        int bin = i % B;
        int out_feature = blockIdx.x * FEATURE_TILE + lf;
        if (out_feature < F)
        {
            int64_t idx = (int64_t)era * F * B + out_feature * B + bin;
            atomicAdd(&grad_hist[idx], sh_grad[i]);
        }
    }
}

__global__ void histogram_feature_tile_root_kernel(
    const int8_t *__restrict__ bin_indices, // [N, F]
    const float *__restrict__ residuals,    // [N]
    const int32_t *__restrict__ era_indices, // [N]
    float *__restrict__ grad_hist,          // [E * F * B]
    float *__restrict__ hess_hist,          // [E * F * B]
    int64_t N, int64_t F, int64_t B, int64_t num_eras,
    int rows_per_thread)
{
    int local_f = threadIdx.x % FEATURE_TILE;
    int row_lane = threadIdx.x / FEATURE_TILE;
    int row_lanes = blockDim.x / FEATURE_TILE;
    int rows_per_block = row_lanes * rows_per_thread;
    int hist_feat_idx = blockIdx.x * FEATURE_TILE + local_f;
    bool valid_feature = hist_feat_idx < F;
    int64_t row_tile_start = (int64_t)blockIdx.y * rows_per_block;

    extern __shared__ float shmem_feature_tile_root[];
    float *sh_grad = shmem_feature_tile_root;                                    // [FEATURE_TILE * ERA_TILE_CAP * B]
    float *sh_hess = &sh_grad[FEATURE_TILE * ERA_TILE_CAP * B];                  // [FEATURE_TILE * ERA_TILE_CAP * B]
    float *sh_residual = &sh_hess[FEATURE_TILE * ERA_TILE_CAP * B];              // [rows_per_block]
    int *sh_era = reinterpret_cast<int *>(&sh_residual[rows_per_block]);         // [rows_per_block]
    int *sh_meta = &sh_era[rows_per_block];                                      // [min_era, max_era]

    if (threadIdx.x == 0) {
        sh_meta[0] = INT_MAX;
        sh_meta[1] = -1;
    }
    __syncthreads();

    if (local_f == 0) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int local_row = r * row_lanes + row_lane;
            int64_t row = row_tile_start + local_row;
            if (row < N)
            {
                int32_t era = era_indices[row];
                sh_era[local_row] = era;
                sh_residual[local_row] = residuals[row];
                if (era >= 0 && era < num_eras)
                {
                    atomicMin(&sh_meta[0], (int)era);
                    atomicMax(&sh_meta[1], (int)era);
                }
            }
            else
            {
                sh_era[local_row] = -1;
                sh_residual[local_row] = 0.0f;
            }
        }
    }
    __syncthreads();

    int era_base = sh_meta[0];
    int era_last = sh_meta[1];
    int era_span = era_last - era_base + 1;
    if (era_span <= 0) {
        return;
    }

    if (era_span > ERA_TILE_CAP) {
        if (valid_feature) {
            for (int r = 0; r < rows_per_thread; ++r)
            {
                int local_row = r * row_lanes + row_lane;
                int64_t row = row_tile_start + local_row;
                if (row < N)
                {
                    int32_t era = sh_era[local_row];
                    int8_t bin = bin_indices[row * F + hist_feat_idx];
                    if (bin >= 0 && bin < B && era >= 0 && era < num_eras)
                    {
                        int64_t idx = era * F * B + hist_feat_idx * B + bin;
                        atomicAdd(&grad_hist[idx], sh_residual[local_row]);
                        atomicAdd(&hess_hist[idx], 1.0f);
                    }
                }
            }
        }
        return;
    }

    int hist_values = FEATURE_TILE * era_span * B;
    for (int i = threadIdx.x; i < hist_values; i += blockDim.x) {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
    }
    __syncthreads();

    if (valid_feature) {
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int local_row = r * row_lanes + row_lane;
            int64_t row = row_tile_start + local_row;
            if (row < N)
            {
                int32_t era = sh_era[local_row];
                int local_era = era - era_base;
                int8_t bin = bin_indices[row * F + hist_feat_idx];
                if (bin >= 0 && bin < B && local_era >= 0 && local_era < era_span)
                {
                    int sh_idx = (local_f * era_span + local_era) * B + bin;
                    atomicAdd(&sh_grad[sh_idx], sh_residual[local_row]);
                    atomicAdd(&sh_hess[sh_idx], 1.0f);
                }
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hist_values; i += blockDim.x)
    {
        int lf = i / (era_span * B);
        int rem = i % (era_span * B);
        int e = era_base + rem / B;
        int bin = rem % B;
        int out_feature = blockIdx.x * FEATURE_TILE + lf;
        if (out_feature < F)
        {
            int64_t idx = e * F * B + out_feature * B + bin;
            atomicAdd(&grad_hist[idx], sh_grad[i]);
            atomicAdd(&hess_hist[idx], sh_hess[i]);
        }
    }
}

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
    int rows_per_thread = 1)
{
    TORCH_CHECK(threads_per_block % FEATURE_TILE == 0,
                "threads_per_block must be divisible by ", FEATURE_TILE,
                " for feature-tile histogram");

    int64_t N = sample_indices.size(0);
    int64_t F = feature_indices.size(0);
    int64_t F_master = bin_indices.size(1);
    int row_lanes = threads_per_block / FEATURE_TILE;
    int rows_per_block = row_lanes * rows_per_thread;
    int64_t row_tiles = (N + rows_per_block - 1) / rows_per_block;
    int64_t feature_tiles = (F + FEATURE_TILE - 1) / FEATURE_TILE;

    dim3 blocks(feature_tiles, row_tiles);
    dim3 threads(threads_per_block);

    int shared_float_count = 2 * FEATURE_TILE * ERA_TILE_CAP * num_bins + rows_per_block;
    int shared_int_count = 2 * rows_per_block + 2;
    int shared_mem_bytes = shared_float_count * (int)sizeof(float) + shared_int_count * (int)sizeof(int);

    histogram_feature_tile_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices.data_ptr<int8_t>(),
        residuals.data_ptr<float>(),
        sample_indices.data_ptr<int32_t>(),
        feature_indices.data_ptr<int32_t>(),
        era_indices.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        N, F_master, F, num_bins, grad_hist.size(0),
        rows_per_thread);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA row-major feature-tile histogram kernel launch failed: ",
                cudaGetErrorString(err));
}

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
    int rows_per_thread = 1)
{
    TORCH_CHECK(threads_per_block % FEATURE_TILE == 0,
                "threads_per_block must be divisible by ", FEATURE_TILE,
                " for by-era feature-tile histogram");
    TORCH_CHECK(max_era_count > 0, "max_era_count must be positive");

    int64_t F = feature_indices.size(0);
    int64_t F_master = bin_indices.size(1);
    int64_t num_eras = era_offsets.size(0) - 1;
    TORCH_CHECK(active_era_start >= 0 && active_era_end <= num_eras && active_era_start < active_era_end,
                "active era range must be non-empty and within era_offsets");
    int row_lanes = threads_per_block / FEATURE_TILE;
    int rows_per_block = row_lanes * rows_per_thread;
    int max_era_tiles = (max_era_count + rows_per_block - 1) / rows_per_block;
    int64_t feature_tiles = (F + FEATURE_TILE - 1) / FEATURE_TILE;

    dim3 blocks(feature_tiles, max_era_tiles, active_era_end - active_era_start);
    dim3 threads(threads_per_block);

    int shared_float_count = 2 * FEATURE_TILE * num_bins + rows_per_block;
    int shared_int_count = rows_per_block;
    int shared_mem_bytes = shared_float_count * (int)sizeof(float) + shared_int_count * (int)sizeof(int);

    histogram_feature_tile_by_era_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices.data_ptr<int8_t>(),
        residuals.data_ptr<float>(),
        sample_indices.data_ptr<int32_t>(),
        feature_indices.data_ptr<int32_t>(),
        era_offsets.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        F_master, F, num_bins,
        active_era_start,
        rows_per_thread);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA by-era feature-tile histogram kernel launch failed: ",
                cudaGetErrorString(err));
}

void launch_histogram_feature_tile_grad_kernel(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1)
{
    TORCH_CHECK(threads_per_block % FEATURE_TILE == 0,
                "threads_per_block must be divisible by ", FEATURE_TILE,
                " for grad-only feature-tile histogram");

    int64_t N = sample_indices.size(0);
    int64_t F = feature_indices.size(0);
    int64_t F_master = bin_indices.size(1);
    int row_lanes = threads_per_block / FEATURE_TILE;
    int rows_per_block = row_lanes * rows_per_thread;
    int64_t row_tiles = (N + rows_per_block - 1) / rows_per_block;
    int64_t feature_tiles = (F + FEATURE_TILE - 1) / FEATURE_TILE;

    dim3 blocks(feature_tiles, row_tiles);
    dim3 threads(threads_per_block);

    int shared_float_count = FEATURE_TILE * ERA_TILE_CAP * num_bins + rows_per_block;
    int shared_int_count = 2 * rows_per_block + 2;
    int shared_mem_bytes = shared_float_count * (int)sizeof(float) + shared_int_count * (int)sizeof(int);

    histogram_feature_tile_grad_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices.data_ptr<int8_t>(),
        residuals.data_ptr<float>(),
        sample_indices.data_ptr<int32_t>(),
        feature_indices.data_ptr<int32_t>(),
        era_indices.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        N, F_master, F, num_bins, grad_hist.size(0),
        rows_per_thread);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA grad-only feature-tile histogram kernel launch failed: ",
                cudaGetErrorString(err));
}

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
    int rows_per_thread = 1)
{
    TORCH_CHECK(threads_per_block % FEATURE_TILE == 0,
                "threads_per_block must be divisible by ", FEATURE_TILE,
                " for by-era grad-only feature-tile histogram");
    TORCH_CHECK(max_era_count > 0, "max_era_count must be positive");

    int64_t F = feature_indices.size(0);
    int64_t F_master = bin_indices.size(1);
    int64_t num_eras = era_offsets.size(0) - 1;
    TORCH_CHECK(active_era_start >= 0 && active_era_end <= num_eras && active_era_start < active_era_end,
                "active era range must be non-empty and within era_offsets");
    int row_lanes = threads_per_block / FEATURE_TILE;
    int rows_per_block = row_lanes * rows_per_thread;
    int max_era_tiles = (max_era_count + rows_per_block - 1) / rows_per_block;
    int64_t feature_tiles = (F + FEATURE_TILE - 1) / FEATURE_TILE;

    dim3 blocks(feature_tiles, max_era_tiles, active_era_end - active_era_start);
    dim3 threads(threads_per_block);

    int shared_float_count = FEATURE_TILE * num_bins + rows_per_block;
    int shared_int_count = rows_per_block;
    int shared_mem_bytes = shared_float_count * (int)sizeof(float) + shared_int_count * (int)sizeof(int);

    histogram_feature_tile_by_era_grad_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices.data_ptr<int8_t>(),
        residuals.data_ptr<float>(),
        sample_indices.data_ptr<int32_t>(),
        feature_indices.data_ptr<int32_t>(),
        era_offsets.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        F_master, F, num_bins,
        active_era_start,
        rows_per_thread);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA by-era grad-only feature-tile histogram kernel launch failed: ",
                cudaGetErrorString(err));
}

void launch_histogram_feature_tile_root_kernel(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1)
{
    TORCH_CHECK(threads_per_block % FEATURE_TILE == 0,
                "threads_per_block must be divisible by ", FEATURE_TILE,
                " for root feature-tile histogram");

    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);
    int row_lanes = threads_per_block / FEATURE_TILE;
    int rows_per_block = row_lanes * rows_per_thread;
    int64_t row_tiles = (N + rows_per_block - 1) / rows_per_block;
    int64_t feature_tiles = (F + FEATURE_TILE - 1) / FEATURE_TILE;

    dim3 blocks(feature_tiles, row_tiles);
    dim3 threads(threads_per_block);

    int shared_float_count = 2 * FEATURE_TILE * ERA_TILE_CAP * num_bins + rows_per_block;
    int shared_int_count = rows_per_block + 2;
    int shared_mem_bytes = shared_float_count * (int)sizeof(float) + shared_int_count * (int)sizeof(int);

    histogram_feature_tile_root_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices.data_ptr<int8_t>(),
        residuals.data_ptr<float>(),
        era_indices.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        N, F, num_bins, grad_hist.size(0),
        rows_per_thread);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA root feature-tile histogram kernel launch failed: ",
                cudaGetErrorString(err));
}

__global__ void subtract_histograms_inplace_kernel(
    float *__restrict__ parent_grad,
    float *__restrict__ parent_hess,
    const float *__restrict__ child_grad,
    const float *__restrict__ child_hess,
    int64_t N)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        parent_grad[i] -= child_grad[i];
        parent_hess[i] -= child_hess[i];
    }
}

void launch_subtract_histograms_inplace_kernel(
    at::Tensor &parent_grad,
    at::Tensor &parent_hess,
    const at::Tensor &child_grad,
    const at::Tensor &child_hess,
    int threads = 256)
{
    int64_t N = parent_grad.numel();
    TORCH_CHECK(parent_hess.numel() == N && child_grad.numel() == N && child_hess.numel() == N,
                "histogram tensors must have the same number of elements");
    dim3 blocks((N + threads - 1) / threads);
    dim3 thread_dims(threads);

    subtract_histograms_inplace_kernel<<<blocks, thread_dims>>>(
        parent_grad.data_ptr<float>(),
        parent_hess.data_ptr<float>(),
        child_grad.data_ptr<float>(),
        child_hess.data_ptr<float>(),
        N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "Histogram subtraction kernel launch failed: ",
                cudaGetErrorString(err));
}
