/**
 * @File:   keypoint_detector_ext.cu
 * @Author: Haozhe Xie
 * @Date:   2024-11-03 16:42:51
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-11-04 11:09:25
 * @Email:  root@haozhexie.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA footprint")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

#define CUDA_NUM_THREADS 512
#define TILE_DIM 16

inline __device__ bool
get_skeleton_map_value(int x, int y, int width, int height,
                       const bool *__restrict__ skeleton_map) {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return false;
  }
  return skeleton_map[y * width + x];
}

__device__ short get_kpt_map_value(int x, int y, int width, int height,
                                   const bool *__restrict__ skeleton_map) {
  short value = 0;
  // x - 1, y - 1 -> 1
  if (get_skeleton_map_value(x - 1, y - 1, width, height, skeleton_map)) {
    value += 1;
  }
  // x, y - 1 -> 2
  if (get_skeleton_map_value(x, y - 1, width, height, skeleton_map)) {
    value += 2;
  }
  // x + 1, y - 1 -> 4
  if (get_skeleton_map_value(x + 1, y - 1, width, height, skeleton_map)) {
    value += 4;
  }
  // x - 1, y -> 8
  if (get_skeleton_map_value(x - 1, y, width, height, skeleton_map)) {
    value += 8;
  }
  // x + 1, y -> 16
  if (get_skeleton_map_value(x + 1, y, width, height, skeleton_map)) {
    value += 16;
  }
  // x - 1, y + 1 -> 32
  if (get_skeleton_map_value(x - 1, y + 1, width, height, skeleton_map)) {
    value += 32;
  }
  // x, y + 1 -> 64
  if (get_skeleton_map_value(x, y + 1, width, height, skeleton_map)) {
    value += 64;
  }
  // x + 1, y + 1 -> 128
  if (get_skeleton_map_value(x + 1, y + 1, width, height, skeleton_map)) {
    value += 128;
  }
  return value;
}

__global__ void keypoint_detection_kernel(int width, int height,
                                          const bool *__restrict__ skeleton_map,
                                          short *__restrict__ kpt_map) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x; // width
  size_t y = blockIdx.y * blockDim.y + threadIdx.y; // height

  int idx = y * width + x;
  if (x < width && y < height) {
    if (!skeleton_map[idx]) {
      return;
    }
    kpt_map[idx] = get_kpt_map_value(x, y, width, height, skeleton_map);
    // ngr_pts_collinear values: 1 + 128; 2 + 64; 4 + 32; 8 + 16
    if (kpt_map[idx] == 129 || kpt_map[idx] == 66 || kpt_map[idx] == 36 ||
        kpt_map[idx] == 24) {
      kpt_map[idx] = 0;
    }
  }
}

torch::Tensor detect_keypoints_ext_cuda_forward(torch::Tensor skeleton_map) {
  CHECK_INPUT(skeleton_map);

  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
  torch::Device device = skeleton_map.device();

  int height = skeleton_map.size(0);
  int width = skeleton_map.size(1);
  torch::Tensor kpt_map =
      torch::zeros({height, width},
                   torch::TensorOptions().dtype(torch::kShort).device(device));

  dim3 blockDim(TILE_DIM, TILE_DIM);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y);

  keypoint_detection_kernel<<<gridDim, blockDim, 0, stream>>>(
      width, height, skeleton_map.data_ptr<bool>(), kpt_map.data_ptr<short>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in detect_keypoints_ext_cuda_forward: %s\n",
           cudaGetErrorString(err));
  }
  return kpt_map;
}
