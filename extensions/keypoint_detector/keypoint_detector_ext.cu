/**
 * @File:   keypoint_detector_ext.cu
 * @Author: Haozhe Xie
 * @Date:   2024-11-03 16:42:51
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-11-03 20:35:28
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
get_local_skeleton_map_value(int x, int y, int width, int height,
                             const bool *__restrict__ skeleton_map) {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return false;
  }
  return skeleton_map[y * width + x];
}

__device__ void
get_local_skeleton_map_values(int x, int y, int width, int height,
                              const bool *__restrict__ skeleton_map,
                              bool *__restrict__ local_skeleton_map) {
  local_skeleton_map[0] =
      get_local_skeleton_map_value(x - 1, y - 1, width, height, skeleton_map);
  local_skeleton_map[1] =
      get_local_skeleton_map_value(x, y - 1, width, height, skeleton_map);
  local_skeleton_map[2] =
      get_local_skeleton_map_value(x + 1, y - 1, width, height, skeleton_map);
  local_skeleton_map[3] =
      get_local_skeleton_map_value(x - 1, y, width, height, skeleton_map);
  local_skeleton_map[4] =
      get_local_skeleton_map_value(x + 1, y, width, height, skeleton_map);
  local_skeleton_map[5] =
      get_local_skeleton_map_value(x - 1, y + 1, width, height, skeleton_map);
  local_skeleton_map[6] =
      get_local_skeleton_map_value(x, y + 1, width, height, skeleton_map);
  local_skeleton_map[7] =
      get_local_skeleton_map_value(x + 1, y + 1, width, height, skeleton_map);
}

inline __device__ bool
is_ngr_pts_collinear(const bool *__restrict__ local_skeleton_map) {
    if (local_skeleton_map[0] && local_skeleton_map[7]) {
        return true;
    } else if (local_skeleton_map[1] && local_skeleton_map[6]) {
        return true;
    } else if (local_skeleton_map[2] && local_skeleton_map[5]) {
        return true;
    } else if (local_skeleton_map[3] && local_skeleton_map[4]) {
        return true;
    }
    return false;
}

__global__ void keypoint_detection_kernel(int width, int height,
                                          const bool *__restrict__ skeleton_map,
                                          bool *__restrict__ kpts_map) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x; // width
  size_t y = blockIdx.y * blockDim.y + threadIdx.y; // height

  if (x < width && y < height) {
    if (skeleton_map[y * width + x]) {
      return;
    }
    bool *local_kpts_map = kpts_map + y * width + x * 9;
    get_local_skeleton_map_values(x, y, width, height, skeleton_map,
                                  local_kpts_map);
    int ngr_pts = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if (local_kpts_map[i]) {
        ++ngr_pts;
      }
    }
    if (ngr_pts != 2 || !is_ngr_pts_collinear(local_kpts_map)) {
      local_kpts_map[8] = true;
    } else {
      local_kpts_map[8] = false;
    }
  }
}

torch::Tensor detect_keypoints_ext_cuda_forward(torch::Tensor skeleton_map) {
  CHECK_INPUT(skeleton_map);

  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
  torch::Device device = skeleton_map.device();

  const int CHANNELS = 9;
  int height = skeleton_map.size(0);
  int width = skeleton_map.size(1);
  torch::Tensor kpts_map =
      torch::empty({height, width, CHANNELS},
                   torch::TensorOptions().dtype(torch::kBool).device(device));

  dim3 dimBlock(TILE_DIM, TILE_DIM);
  dim3 dimGrid((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y);

  keypoint_detection_kernel<<<dimGrid, dimBlock, 0, stream>>>(
      width, height, skeleton_map.data_ptr<bool>(), kpts_map.data_ptr<bool>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in detect_keypoints_ext_cuda_forward: %s\n",
           cudaGetErrorString(err));
  }
  return kpts_map;
}
