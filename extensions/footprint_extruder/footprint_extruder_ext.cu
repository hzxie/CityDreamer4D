/**
 * @File:   extrude_footprint_ext.cu
 * @Author: Haozhe Xie
 * @Date:   2023-03-26 11:06:18
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-11-03 18:19:02
 * @Email:  root@haozhexie.com
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>

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

template <typename scalar_t>
__global__ void extrude_footprint_ext_cuda_kernel(
    int height, int width, int depth, int l1_height, int roof_height,
    int l1_id_offset, int roof_id_offset, int bldg_inst_min, int bldg_inst_max,
    const scalar_t *__restrict__ bev_ins_map, const short *__restrict__ hf_td,
    const short *__restrict__ hf_bu, scalar_t *__restrict__ volume) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x; // width
  size_t j = blockIdx.y * blockDim.y + threadIdx.y; // height

  if (i < width && j < height) {
    short hgt_up = hf_td[j * width + i];
    short hgt_lw = hf_bu[j * width + i];
    scalar_t inst = bev_ins_map[j * width + i];
    int64_t vol_offset = j * width * depth + i * depth;
    for (int k = hgt_lw; k <= hgt_up; ++k) {
      volume[vol_offset + k] = inst;
      if (inst >= bldg_inst_min && inst < bldg_inst_max) {
        if (k >= hgt_lw && k < l1_height) {
          volume[vol_offset + k] = inst + l1_id_offset;
        }
        if (k > hgt_up - roof_height && k <= hgt_up) {
          volume[vol_offset + k] = inst + roof_id_offset;
        }
      }
    }
  }
}

torch::Tensor extrude_footprint_ext_cuda_forward(
    torch::Tensor volume, torch::Tensor bev_ins_map, torch::Tensor hf_td,
    torch::Tensor hf_bu, int l1_height, int roof_height, int l1_id_offset,
    int roof_id_offset, int bldg_inst_min, int bldg_inst_max) {
  CHECK_INPUT(volume);
  CHECK_INPUT(bev_ins_map);
  CHECK_INPUT(hf_td);
  CHECK_INPUT(hf_bu);

  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

  size_t height = volume.size(0);
  size_t width = volume.size(1);
  size_t depth = volume.size(2);

  dim3 blockDim(TILE_DIM, TILE_DIM);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y);

  AT_DISPATCH_INTEGRAL_TYPES(
      volume.scalar_type(), "extrude_footprint_ext_cuda", ([&] {
        extrude_footprint_ext_cuda_kernel<<<gridDim, blockDim, 0, stream>>>(
            height, width, depth, l1_height, roof_height, l1_id_offset,
            roof_id_offset, bldg_inst_min, bldg_inst_max,
            bev_ins_map.data_ptr<scalar_t>(), hf_td.data_ptr<short>(),
            hf_bu.data_ptr<short>(), volume.data_ptr<scalar_t>());
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in extrude_footprint_ext_cuda_forward: %s\n",
           cudaGetErrorString(err));
  }
  return volume;
}
