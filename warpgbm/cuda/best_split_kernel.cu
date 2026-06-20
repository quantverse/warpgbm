#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    float *sh_gain = shmem;                    // [B - 1]
    float *sh_agreement = &sh_gain[split_bins]; // [B - 1]

    for (int b = threadIdx.x; b < split_bins; b += blockDim.x) {
        sh_gain[b] = 0.0f;
        sh_agreement[b] = 0.0f;
    }
    __syncthreads();

    // Parallelize across eras within each feature block, reducing into shared memory.
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

            atomicAdd(&sh_gain[b], gain * inv_E);
            atomicAdd(&sh_agreement[b], dir * inv_E);
        }
    }
    __syncthreads();

    for (int b = threadIdx.x; b < split_bins; b += blockDim.x) {
        fused_gain[out_base + b] = sh_gain[b];
        fused_agreement[out_base + b] = fabsf(sh_agreement[b]);
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

    dim3 blocks(F);
    dim3 thread_dims(threads);
    int shared_mem_bytes = 2 * (B - 1) * (int)sizeof(float);

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
