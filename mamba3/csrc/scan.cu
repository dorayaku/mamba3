// Inter-chunk SSD scan — forward + backward CUDA kernels.
// h[0] = h_init, h[k+1] = decay[k] * h[k] + h_chunk[k]
// Register-based state, no global memory round-trips between steps.

#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int MAX_ELEM = 16;

__global__ void ssd_scan_fwd_kernel(
    const float* __restrict__ decay,
    const float* __restrict__ h_chunk,
    const float* __restrict__ h_init,
    float* __restrict__ h_states,
    int K, int H, int PN)
{
    const int bh = blockIdx.x;
    const int b = bh / H, h = bh % H;
    const long long base = ((long long)b * K * H + h) * PN;
    const long long ks = (long long)H * PN;
    const long long db = (long long)b * K * H + h;

    const int n_elem = (PN + blockDim.x - 1) / blockDim.x;

    float state[MAX_ELEM];
    for (int i = 0; i < n_elem && i < MAX_ELEM; i++) {
        const int j = threadIdx.x + i * blockDim.x;
        if (j < PN) {
            if (h_init != nullptr) {
                const long long init_idx = ((long long)b * H + h) * PN + j;
                state[i] = h_init[init_idx];
            } else {
                state[i] = 0.0f;
            }
            h_states[base + j] = state[i];
        }
    }

    for (int k = 0; k < K - 1; k++) {
        const float d = decay[db + (long long)k * H];
        for (int i = 0; i < n_elem && i < MAX_ELEM; i++) {
            const int j = threadIdx.x + i * blockDim.x;
            if (j < PN) {
                const long long chunk_idx = base + (long long)k * ks + j;
                state[i] = d * state[i] + h_chunk[chunk_idx];
                h_states[base + (long long)(k + 1) * ks + j] = state[i];
            }
        }
    }
}

__global__ void ssd_scan_bwd_kernel(
    const float* __restrict__ decay,
    const float* __restrict__ h_states,
    const float* __restrict__ grad_h_in,
    float* __restrict__ d_chunk,
    float* __restrict__ d_decay,
    float* __restrict__ d_h_init,
    int K, int H, int PN,
    bool has_h_init)
{
    const int bh = blockIdx.x;
    const int b = bh / H, h = bh % H;
    const long long base = ((long long)b * K * H + h) * PN;
    const long long ks = (long long)H * PN;
    const long long db = (long long)b * K * H + h;

    const int n_elem = (PN + blockDim.x - 1) / blockDim.x;

    __shared__ float shr[32];

    float dh_reg[MAX_ELEM];
    for (int i = 0; i < n_elem && i < MAX_ELEM; i++) {
        const int j = threadIdx.x + i * blockDim.x;
        if (j < PN) {
            dh_reg[i] = grad_h_in[base + (long long)(K - 1) * ks + j];
            d_chunk[base + (long long)(K - 1) * ks + j] = 0.0f;
        }
    }
    if (threadIdx.x == 0)
        d_decay[db + (long long)(K - 1) * H] = 0.0f;

    for (int k = K - 2; k >= 0; k--) {
        const float d = decay[db + (long long)k * H];
        const long long o0 = base + (long long)k * ks;
        const long long o1 = o0 + ks;

        float acc = 0.0f;
        for (int i = 0; i < n_elem && i < MAX_ELEM; i++) {
            const int j = threadIdx.x + i * blockDim.x;
            if (j < PN) {
                const float g = dh_reg[i];
                acc += g * h_states[o0 + j];
                d_chunk[o0 + j] = g;
                dh_reg[i] = grad_h_in[o0 + j] + d * g;
            }
        }

        // warp reduction for d_decay
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, off);

        const int w = threadIdx.x / 32, l = threadIdx.x % 32;
        if (l == 0) shr[w] = acc;
        __syncthreads();
        if (w == 0) {
            acc = (l < (blockDim.x + 31) / 32) ? shr[l] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_xor_sync(0xffffffff, acc, off);
        }
        if (threadIdx.x == 0)
            d_decay[db + (long long)k * H] = acc;

        if (k > 0)
            __syncthreads();
    }

    if (has_h_init && d_h_init != nullptr) {
        const long long init_base = ((long long)b * H + h) * PN;
        for (int i = 0; i < n_elem && i < MAX_ELEM; i++) {
            const int j = threadIdx.x + i * blockDim.x;
            if (j < PN)
                d_h_init[init_base + j] = dh_reg[i];
        }
    }
}

torch::Tensor scan_forward(
    torch::Tensor decay,
    torch::Tensor h_chunk,
    torch::optional<torch::Tensor> h_init)
{
    TORCH_CHECK(decay.is_cuda() && h_chunk.is_cuda(),
                "scan_forward: inputs must be on CUDA");
    TORCH_CHECK(decay.is_contiguous() && h_chunk.is_contiguous(),
                "scan_forward: inputs must be contiguous");

    const int B = h_chunk.size(0);
    const int K = h_chunk.size(1);
    const int H = h_chunk.size(2);
    const int PN = h_chunk.size(3);

    TORCH_CHECK(PN <= 1024 * MAX_ELEM,
                "scan_forward: PN (", PN, ") exceeds maximum supported (",
                1024 * MAX_ELEM, ")");

    const float* h_init_ptr = nullptr;
    if (h_init.has_value()) {
        TORCH_CHECK(h_init->is_cuda() && h_init->is_contiguous(),
                    "scan_forward: h_init must be contiguous CUDA tensor");
        h_init_ptr = h_init->data_ptr<float>();
    }

    auto h_states = torch::zeros_like(h_chunk);
    const int threads = std::min(PN, 1024);
    const int blocks = B * H;

    ssd_scan_fwd_kernel<<<blocks, threads>>>(
        decay.data_ptr<float>(),
        h_chunk.data_ptr<float>(),
        h_init_ptr,
        h_states.data_ptr<float>(),
        K, H, PN);

    return h_states;
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
scan_backward(
    torch::Tensor decay,
    torch::Tensor h_states,
    torch::Tensor grad_h,
    bool has_h_init)
{
    TORCH_CHECK(decay.is_cuda() && h_states.is_cuda() && grad_h.is_cuda(),
                "scan_backward: inputs must be on CUDA");
    TORCH_CHECK(decay.is_contiguous() && h_states.is_contiguous()
                && grad_h.is_contiguous(),
                "scan_backward: inputs must be contiguous");

    const int B = h_states.size(0);
    const int K = h_states.size(1);
    const int H = h_states.size(2);
    const int PN = h_states.size(3);

    auto d_chunk = torch::zeros_like(h_states);
    auto d_decay = torch::zeros_like(decay);

    torch::optional<torch::Tensor> d_h_init;
    float* d_h_init_ptr = nullptr;
    if (has_h_init) {
        d_h_init = torch::zeros({B, H, PN},
            torch::TensorOptions().dtype(torch::kFloat32).device(decay.device()));
        d_h_init_ptr = d_h_init->data_ptr<float>();
    }

    const int threads = std::min(PN, 1024);
    const int blocks = B * H;

    ssd_scan_bwd_kernel<<<blocks, threads>>>(
        decay.data_ptr<float>(),
        h_states.data_ptr<float>(),
        grad_h.data_ptr<float>(),
        d_chunk.data_ptr<float>(),
        d_decay.data_ptr<float>(),
        d_h_init_ptr,
        K, H, PN,
        has_h_init);

    return {d_decay, d_chunk, d_h_init};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_forward", &scan_forward,
          py::arg("decay"), py::arg("h_chunk"),
          py::arg("h_init") = py::none());
    m.def("scan_backward", &scan_backward,
          py::arg("decay"), py::arg("h_states"),
          py::arg("grad_h"), py::arg("has_h_init") = false);
}
