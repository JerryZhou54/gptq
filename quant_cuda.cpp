#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void vecquant3matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant3matmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
); 

void vecquant3matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_cuda(vec, mat, mul, scales, zeros);
}

void vecquant3matmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_faster_cuda(vec, mat, mul, scales, zeros);
}


// void vecquant3matmul(
//   torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
//   torch::Tensor scales, torch::Tensor zeros, int batch_size
// ) {
//   const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
//   for (int i = 0; i < batch_size; i++)
//     vecquant3matmul_cuda(vec[i], mat, mul[i], scales, zeros);
// }

// void vecquant3matmul_faster(
//   torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
//   torch::Tensor scales, torch::Tensor zeros, int batch_size
// ) {
//   const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
//   for (int i = 0; i < batch_size; i++)
//     vecquant3matmul_faster_cuda(vec[i], mat, mul[i], scales, zeros);
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant3matmul_faster", &vecquant3matmul_faster, "Vector 3-bit Quantized Matrix Multiplication (CUDA), faster version");
}
