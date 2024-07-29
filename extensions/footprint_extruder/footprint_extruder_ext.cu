/**
 * @File:   extrude_footprint_ext.cu
 * @Author: Haozhe Xie
 * @Date:   2023-03-26 11:06:18
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-07-29 16:38:55
 * @Email:  root@haozhexie.com
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <torch/extension.h>

#define CUDA_NUM_THREADS 512

// Computer the number of threads needed in GPU
inline int get_n_threads(int n) {
  const int pow_2 = std::log(static_cast<float>(n)) / std::log(2.0);
  return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}

__global__ void extrude_footprint_ext_cuda_kernel(
    int height, int width, int depth, int l1_height, int roof_height,
    int l1_id_offset, int roof_id_offset, int bldg_inst_min, int bldg_inst_max,
    const short *__restrict__ bev_ins_map, const short *__restrict__ hf_td,
    const short *__restrict__ hf_bu, short *__restrict__ volume) {
  int blk_index = blockIdx.x;  // Height
  int thr_index = threadIdx.x; // Width (* Depth)
  int stride = blockDim.x;

  bev_ins_map += blk_index * width;
  hf_td += blk_index * width;
  hf_bu += blk_index * width;
  volume += blk_index * width * depth;

  for (int i = thr_index; i < width; i += stride) {
    short hgt_up = hf_td[i];
    short hgt_lw = hf_bu[i];
    short inst = bev_ins_map[i];

    for (int j = hgt_lw; j <= hgt_up; ++j) {
      int offset_3d = i * depth;

      volume[offset_3d + j] = inst;
      if (inst >= bldg_inst_min && inst < bldg_inst_max) {
        if (j >= hgt_lw && j < l1_height) {
          volume[offset_3d + j] = inst + l1_id_offset;
        }
        if (j > hgt_up - roof_height && j <= hgt_up) {
          volume[offset_3d + j] = inst + roof_id_offset;
        }
      }
    }
  }
}

torch::Tensor extrude_footprint_ext_cuda_forward(
    torch::Tensor volume, torch::Tensor bev_ins_map, torch::Tensor hf_td,
    torch::Tensor hf_bu, int l1_height, int roof_height, int l1_id_offset,
    int roof_id_offset, int bldg_inst_min, int bldg_inst_max,
    cudaStream_t stream) {
  size_t height = volume.size(0);
  size_t width = volume.size(1);
  size_t depth = volume.size(2);

  extrude_footprint_ext_cuda_kernel<<<
      height, int(CUDA_NUM_THREADS / get_n_threads(width)), 0, stream>>>(
      height, width, depth, l1_height, roof_height, l1_id_offset,
      roof_id_offset, bldg_inst_min, bldg_inst_max,
      bev_ins_map.data_ptr<short>(), hf_td.data_ptr<short>(),
      hf_bu.data_ptr<short>(), volume.data_ptr<short>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in extrude_footprint_ext_cuda_forward: %s\n",
           cudaGetErrorString(err));
  }
  return volume;
}
