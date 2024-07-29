/**
 * @File:   bindings.cpp
 * @Author: Haozhe Xie
 * @Date:   2023-03-26 11:06:13
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-07-29 16:30:27
 * @Email:  root@haozhexie.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA footprint")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor extrude_footprint_ext_cuda_forward(
    torch::Tensor volume, torch::Tensor bev_ins_map, torch::Tensor hf_td,
    torch::Tensor hf_bu, int l1_height, int roof_height, int l1_id_offset,
    int roof_id_offset, int bldg_inst_min, int bldg_inst_max,
    cudaStream_t stream);

torch::Tensor extrude_footprint_ext_forward(
    torch::Tensor volume, torch::Tensor bev_ins_map, torch::Tensor hf_td,
    torch::Tensor hf_bu, int l1_height, int roof_height, int l1_id_offset,
    int roof_id_offset, int bldg_inst_min, int bldg_inst_max) {
  CHECK_INPUT(volume);
  CHECK_INPUT(bev_ins_map);
  CHECK_INPUT(hf_td);
  CHECK_INPUT(hf_bu);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return extrude_footprint_ext_cuda_forward(
      volume, bev_ins_map, hf_td, hf_bu, l1_height, roof_height, l1_id_offset,
      roof_id_offset, bldg_inst_min, bldg_inst_max, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &extrude_footprint_ext_forward,
        "Extrude Tensor Ext. Forward (CUDA)");
}
