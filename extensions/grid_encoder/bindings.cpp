/**
 * @File:   grid_encoder_ext_cuda.cpp
 * @Author: Jiaxiang Tang (@ashawkey)
 * @Date:   2023-04-15 10:39:17
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-07-12 15:14:35
 * @Email:  ashawkey1999@gmail.com
 * @Ref: https://github.com/ashawkey/torch-ngp
 */

#include <stdint.h>
#include <torch/extension.h>
#include <torch/torch.h>

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [B, L * C], float
// H: base resolution
void grid_encode_forward(const torch::tensor inputs, const torch::tensor embeddings,
                         const torch::tensor offsets, torch::tensor outputs,
                         const uint32_t B, const uint32_t D, const uint32_t C,
                         const uint32_t L, const float S, const uint32_t H,
                         const bool calc_grad_inputs, torch::tensor dy_dx,
                         const uint32_t gridtype, const bool align_corners);
void grid_encode_backward(const torch::tensor grad, const torch::tensor inputs,
                          const torch::tensor embeddings, const torch::tensor offsets,
                          torch::tensor grad_embeddings, const uint32_t B,
                          const uint32_t D, const uint32_t C, const uint32_t L,
                          const float S, const uint32_t H,
                          const bool calc_grad_inputs, const torch::tensor dy_dx,
                          torch::tensor grad_inputs, const uint32_t gridtype,
                          const bool align_corners);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &grid_encode_forward,
        "grid_encode_forward (CUDA)");
  m.def("backward", &grid_encode_backward,
        "grid_encode_backward (CUDA)");
}
