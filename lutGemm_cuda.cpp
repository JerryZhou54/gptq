#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>


void lutGemm_cuda(
    torch::Tensor W,
    torch::Tensor alpha,
    torch::Tensor input,
    torch::Tensor output,
    int NUM_BITS
);


void lutMatmul(
    torch::Tensor W,
    torch::Tensor alpha,
    torch::Tensor input,
    torch::Tensor output,
    int NUM_BITS
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  lutGemm_cuda(W, alpha, input, output, NUM_BITS);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lutMatmul", &lutMatmul, "lookup table gemm (CUDA)");
}