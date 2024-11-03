/**
 * @File:   bindings.cpp
 * @Author: Haozhe Xie
 * @Date:   2024-11-03 16:29:36
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-11-03 18:07:21
 * @Email:  root@haozhexie.com
 */

#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor detect_keypoints_ext_cuda_forward(torch::Tensor skeleton_map);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("detect_keypoints", &detect_keypoints_ext_cuda_forward,
        "Keypoint Detector Ext. Forward (CUDA)");
}
