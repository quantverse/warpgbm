#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

constexpr int64_t DEFAULT_DYNAMIC_SHARED_LIMIT = 49152;

__global__ void directional_split_kernel(
    const float *__restrict__ G,                // [E * F * B]
    const float *__restrict__ H,                // [E * F * B]
    int E, int F, int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    float *__restrict__ per_era_gain,           // [E * F * (B-1)]
    float *__restrict__ per_era_direction       // [E * F * (B-1)]
)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x; // feature index
    int e = blockIdx.y;                            // era index

    if (f >= F || e >= E) return;

    // Access base offset for this (era, feature)
    int base = e * F * B + f * B;
    int base_gain = e * F * (B - 1) + f * (B - 1);

    float G_total = 0.0f, H_total = 0.0f;
    for (int b = 0; b < B; ++b) {
        G_total += G[base + b];
        H_total += H[base + b];
    }

    float G_L = 0.0f, H_L = 0.0f;
    for (int b = 0; b < B - 1; ++b) {
        G_L += G[base + b];
        H_L += H[base + b];

        float G_R = G_total - G_L;
        float H_R = H_total - H_L;

        float gain = 0.0f;
        float dir = 0.0f;

        if (H_L >= min_child_samples && H_R >= min_child_samples) {
            gain = (G_L * G_L) / (H_L + eps)
                 + (G_R * G_R) / (H_R + eps)
                 - (G_total * G_total) / (H_total + eps);

            float left_val = G_L / (H_L + eps);
            float right_val = G_R / (H_R + eps);
            dir = (left_val > right_val) ? 1.0f : -1.0f;
        }

        per_era_gain[base_gain + b] = gain;
        per_era_direction[base_gain + b] = dir;
    }
}

void launch_directional_split_kernel(
    const at::Tensor &G, // [E, F, B]
    const at::Tensor &H, // [E, F, B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &per_era_gain,       // [E, F, B]
    at::Tensor &per_era_direction,  // [E, F, B]
    int threads = 128)
{
    int E = G.size(0);
    int F = G.size(1);
    int B = G.size(2);

    dim3 blocks((F + threads - 1) / threads, E); // (feature blocks, era grid)
    dim3 thread_dims(threads);

    directional_split_kernel<<<blocks, thread_dims>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        E, F, B,
        min_split_gain,
        min_child_samples,
        eps,
        per_era_gain.data_ptr<float>(),
        per_era_direction.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "Directional split kernel launch failed: ",
                cudaGetErrorString(err));
}

// Fused kernel: reduces across eras inside the kernel, outputs [F, B-1] instead of [E, F, B-1].
// Eliminates E× global memory writes and the subsequent Python-side reduction.
__global__ void directional_split_fused_kernel(
    const float *__restrict__ G,                // [E * F * B]
    const float *__restrict__ H,                // [E * F * B]
    int E, int F, int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    float *__restrict__ fused_gain,             // [F * (B-1)]
    float *__restrict__ fused_agreement         // [F * (B-1)]
)
{
    int f = blockIdx.x;
    if (f >= F) return;

    int split_bins = B - 1;
    int out_base = f * split_bins;
    float inv_E = 1.0f / (float)E;

    extern __shared__ float shmem[];
    float *sh_gain = shmem;                                      // [split_bins * blockDim.x]
    float *sh_agreement = &sh_gain[split_bins * blockDim.x];     // [split_bins * blockDim.x]

    for (int b = 0; b < split_bins; ++b) {
        int idx = b * blockDim.x + threadIdx.x;
        sh_gain[idx] = 0.0f;
        sh_agreement[idx] = 0.0f;
    }
    __syncthreads();

    // Parallelize across eras within each feature block. Each thread owns one
    // slot per split bin, so accumulation needs no atomics and stays in a fixed
    // increasing-era order per thread — the reduction is bitwise reproducible.
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        int base = e * F * B + f * B;

        float G_total = 0.0f, H_total = 0.0f;
        for (int bb = 0; bb < B; ++bb) {
            G_total += G[base + bb];
            H_total += H[base + bb];
        }

        float G_L = 0.0f, H_L = 0.0f;
        for (int b = 0; b < split_bins; ++b) {
            G_L += G[base + b];
            H_L += H[base + b];

            float G_R = G_total - G_L;
            float H_R = H_total - H_L;

            float gain = 0.0f;
            float dir  = 0.0f;

            if (H_L >= min_child_samples && H_R >= min_child_samples) {
                gain = (G_L * G_L) / (H_L + eps)
                     + (G_R * G_R) / (H_R + eps)
                     - (G_total * G_total) / (H_total + eps);
                float left_val  = G_L / (H_L + eps);
                float right_val = G_R / (H_R + eps);
                dir = (left_val > right_val) ? 1.0f : -1.0f;
            }

            // Accumulate unscaled: direction sums stay exact integers, so cells with
            // the same net direction count get bit-identical agreement. Scaling
            // per-add would let the exact-equality agreement mask in
            // select_split_fused drop true tie members.
            int idx = b * blockDim.x + threadIdx.x;
            sh_gain[idx] += gain;
            sh_agreement[idx] += dir;
        }
    }
    __syncthreads();

    // Fixed pairwise tree reduction across thread slots: the combine order is a
    // function of the launch configuration only, so results are reproducible.
    for (int active = blockDim.x; active > 1; active = (active + 1) / 2) {
        int stride = (active + 1) / 2;
        if (threadIdx.x < stride && threadIdx.x + stride < active) {
            for (int b = 0; b < split_bins; ++b) {
                int idx = b * blockDim.x + threadIdx.x;
                sh_gain[idx] += sh_gain[idx + stride];
                sh_agreement[idx] += sh_agreement[idx + stride];
            }
        }
        __syncthreads();
    }

    for (int b = threadIdx.x; b < split_bins; b += blockDim.x) {
        fused_gain[out_base + b] = sh_gain[b * blockDim.x] * inv_E;
        fused_agreement[out_base + b] = fabsf(sh_agreement[b * blockDim.x] * inv_E);
    }
}

void launch_directional_split_fused_kernel(
    const at::Tensor &G, // [E, F, B]
    const at::Tensor &H, // [E, F, B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &fused_gain,       // [F, B-1]
    at::Tensor &fused_agreement,  // [F, B-1]
    int threads = 128)
{
    int E = G.size(0);
    int F = G.size(1);
    int B = G.size(2);
    TORCH_CHECK(B >= 2, "Fused split kernel requires at least 2 bins");

    dim3 blocks(F);
    int split_bins = B - 1;

    // The deterministic reduction keeps 2 * split_bins per-thread floats in
    // shared memory; shrink the block (whole warps) when the caller's thread
    // count would exceed the dynamic shared-memory limit.
    int64_t shared_per_thread = 2LL * split_bins * (int64_t)sizeof(float);
    int threads_used = std::min(threads, 1024);
    threads_used = std::min(threads_used, (int)(DEFAULT_DYNAMIC_SHARED_LIMIT / shared_per_thread));
    threads_used = std::max(32, (threads_used / 32) * 32);
    TORCH_CHECK(2LL * split_bins * threads_used * (int64_t)sizeof(float) <= DEFAULT_DYNAMIC_SHARED_LIMIT,
                "Fused split kernel: num_bins too large for the deterministic reduction (",
                split_bins, " split bins)");

    dim3 thread_dims(threads_used);
    int shared_mem_bytes = (int)(2LL * split_bins * threads_used * (int64_t)sizeof(float));

    directional_split_fused_kernel<<<blocks, thread_dims, shared_mem_bytes>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        E, F, B,
        min_split_gain,
        min_child_samples,
        eps,
        fused_gain.data_ptr<float>(),
        fused_agreement.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "Fused directional split kernel launch failed: ",
                cudaGetErrorString(err));
}

__global__ void select_split_fused_kernel(
    const float *__restrict__ fused_gain,       // [F * (B-1)]
    const float *__restrict__ fused_agreement,  // [F * (B-1)]
    int N,
    float min_split_gain,
    int *__restrict__ best_idx)
{
    extern __shared__ unsigned char shmem_raw[];
    float *sh_agreement = reinterpret_cast<float *>(shmem_raw);
    float *sh_gain = &sh_agreement[blockDim.x];
    int *sh_idx = reinterpret_cast<int *>(&sh_gain[blockDim.x]);

    float local_agreement = -FLT_MAX;
    float local_gain = -FLT_MAX;
    int local_idx = -1;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float agreement = fused_agreement[i];
        if (agreement > local_agreement) {
            local_agreement = agreement;
        }
    }

    sh_agreement[threadIdx.x] = local_agreement;
    __syncthreads();

    for (int active = blockDim.x; active > 1; active = (active + 1) / 2) {
        int stride = (active + 1) / 2;
        if (threadIdx.x < stride && threadIdx.x + stride < active) {
            float other_agreement = sh_agreement[threadIdx.x + stride];
            if (other_agreement > sh_agreement[threadIdx.x]) {
                sh_agreement[threadIdx.x] = other_agreement;
            }
        }
        __syncthreads();
    }

    float max_agreement = sh_agreement[0];
    local_gain = -FLT_MAX;
    local_idx = -1;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float gain = fused_gain[i];
        float agreement = fused_agreement[i];
        if (agreement == max_agreement && gain > min_split_gain &&
            (gain > local_gain || (gain == local_gain && (local_idx < 0 || i < local_idx)))) {
            local_gain = gain;
            local_idx = i;
        }
    }

    sh_gain[threadIdx.x] = local_gain;
    sh_idx[threadIdx.x] = local_idx;
    __syncthreads();

    for (int active = blockDim.x; active > 1; active = (active + 1) / 2) {
        int stride = (active + 1) / 2;
        if (threadIdx.x < stride && threadIdx.x + stride < active) {
            float other_gain = sh_gain[threadIdx.x + stride];
            int other_idx = sh_idx[threadIdx.x + stride];
            if (other_idx >= 0 &&
                (other_gain > sh_gain[threadIdx.x] ||
                 (other_gain == sh_gain[threadIdx.x] &&
                  (sh_idx[threadIdx.x] < 0 || other_idx < sh_idx[threadIdx.x])))) {
                sh_gain[threadIdx.x] = other_gain;
                sh_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        best_idx[0] = sh_idx[0];
    }
}

void launch_select_split_fused_kernel(
    const at::Tensor &fused_gain,
    const at::Tensor &fused_agreement,
    float min_split_gain,
    at::Tensor &best_idx,
    int threads = 128)
{
    int N = fused_gain.numel();
    int shared_mem_bytes = 2 * threads * (int)sizeof(float) + threads * (int)sizeof(int);

    select_split_fused_kernel<<<1, threads, shared_mem_bytes>>>(
        fused_gain.data_ptr<float>(),
        fused_agreement.data_ptr<float>(),
        N,
        min_split_gain,
        best_idx.data_ptr<int>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "Fused split selector kernel launch failed: ",
                cudaGetErrorString(err));
}

__global__ void accumulate_feature_importance_kernel(
    const float *__restrict__ G, // [E * F * B]
    const float *__restrict__ H, // [E * F * B]
    int E, int F, int B,
    int local_feature,
    int best_bin,
    int global_feature,
    float min_child_samples,
    float eps,
    float *__restrict__ per_era_feature_importance, // [E * num_features]
    int importance_eras,
    int num_features)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E || e >= importance_eras ||
        local_feature < 0 || local_feature >= F ||
        best_bin < 0 || best_bin >= B - 1 ||
        global_feature < 0 || global_feature >= num_features) {
        return;
    }

    int base = e * F * B + local_feature * B;

    float G_total = 0.0f;
    float H_total = 0.0f;
    float G_L = 0.0f;
    float H_L = 0.0f;
    for (int b = 0; b < B; ++b) {
        float g = G[base + b];
        float h = H[base + b];
        G_total += g;
        H_total += h;
        if (b <= best_bin) {
            G_L += g;
            H_L += h;
        }
    }

    float G_R = G_total - G_L;
    float H_R = H_total - H_L;
    float gain = 0.0f;
    if (H_L >= min_child_samples && H_R >= min_child_samples) {
        gain = (G_L * G_L) / (H_L + eps)
             + (G_R * G_R) / (H_R + eps)
             - (G_total * G_total) / (H_total + eps);
    }

    per_era_feature_importance[e * num_features + global_feature] += gain;
}

void launch_accumulate_feature_importance_kernel(
    const at::Tensor &G,
    const at::Tensor &H,
    int local_feature,
    int best_bin,
    int global_feature,
    float min_child_samples,
    float eps,
    at::Tensor &per_era_feature_importance,
    int threads = 128)
{
    int E = G.size(0);
    int F = G.size(1);
    int B = G.size(2);
    int importance_eras = per_era_feature_importance.size(0);
    int num_features = per_era_feature_importance.size(1);
    dim3 blocks((E + threads - 1) / threads);
    dim3 thread_dims(threads);

    accumulate_feature_importance_kernel<<<blocks, thread_dims>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        E, F, B,
        local_feature,
        best_bin,
        global_feature,
        min_child_samples,
        eps,
        per_era_feature_importance.data_ptr<float>(),
        importance_eras,
        num_features);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "Feature importance accumulation kernel launch failed: ",
                cudaGetErrorString(err));
}
