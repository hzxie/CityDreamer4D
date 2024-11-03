/**
 * @File:   bindings.cpp
 * @Author: Haozhe Xie
 * @Date:   2023-03-26 11:06:13
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-11-03 18:07:16
 * @Email:  root@haozhexie.com
 */

#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor extrude_footprint_ext_cuda_forward(
    torch::Tensor volume, torch::Tensor bev_ins_map, torch::Tensor hf_td,
    torch::Tensor hf_bu, int l1_height, int roof_height, int l1_id_offset,
    int roof_id_offset, int bldg_inst_min, int bldg_inst_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("extrude_footprint", &extrude_footprint_ext_cuda_forward,
        "Extrude Tensor Ext. Forward (CUDA)");
}
